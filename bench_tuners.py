"""Benchmark: coordinate descent vs abduction tuner on synthetic cost functions.

Synthetic functions simulate real Triton kernel tuning landscapes:
- Rastrigin-like saddle points (many local optima)
- Coupled parameters (BLOCK_M and num_warps interact)
- Plateau regions (large flat areas where coordesc stalls)

No CUDA required — tests the search algorithm, not the kernel.
"""
import copy
import math
import time
from dataclasses import dataclass, field
from collections.abc import Callable

from torch._inductor.runtime.coordinate_descent_tuner import CoordescTuner


# --- Minimal triton.Config shim (no Triton install needed) ---

class FakeConfig:
    """Drop-in for triton.Config with the fields CoordescTuner expects."""
    def __init__(self, kwargs: dict, num_warps=4, num_stages=2):
        self.kwargs = dict(kwargs)
        self.num_warps = num_warps
        self.num_stages = num_stages
        self.found_by_coordesc = False

    def __repr__(self):
        parts = [f"{k}={v}" for k, v in sorted(self.kwargs.items())]
        parts.append(f"num_warps={self.num_warps}")
        parts.append(f"num_stages={self.num_stages}")
        return f"Config({', '.join(parts)})"

    def __hash__(self):
        return hash(tuple(sorted(self.kwargs.items())) + (self.num_warps, self.num_stages))

    def __eq__(self, other):
        return (self.kwargs == other.kwargs and
                self.num_warps == other.num_warps and
                self.num_stages == other.num_stages)


# Patch triton_config_to_hashable for FakeConfig
import torch._inductor.runtime.coordinate_descent_tuner as cdt
_orig_hashable = cdt.triton_config_to_hashable
def _fake_hashable(cfg):
    if isinstance(cfg, FakeConfig):
        return hash(cfg)
    return _orig_hashable(cfg)
cdt.triton_config_to_hashable = _fake_hashable


# --- Abduction tuner (prototype) ---

TRANSITIONS: dict[str, set[str]] = {
    "BLOCK_M":    {"BLOCK_M", "BLOCK_K", "num_warps", "BLOCK_N", "num_stages"},
    "BLOCK_N":    {"BLOCK_N", "BLOCK_K", "num_warps", "BLOCK_M"},
    "BLOCK_K":    {"BLOCK_K", "BLOCK_M", "BLOCK_N", "num_stages"},
    "XBLOCK":     {"XBLOCK", "num_warps", "R0_BLOCK", "YBLOCK"},
    "YBLOCK":     {"YBLOCK", "XBLOCK", "num_warps", "ZBLOCK"},
    "ZBLOCK":     {"ZBLOCK", "YBLOCK", "num_warps"},
    "R0_BLOCK":   {"R0_BLOCK", "XBLOCK", "num_warps", "R1_BLOCK"},
    "R1_BLOCK":   {"R1_BLOCK", "R0_BLOCK", "num_warps"},
    "num_warps":  {"num_warps", "BLOCK_M", "BLOCK_N", "XBLOCK", "num_stages"},
    "num_stages": {"num_stages", "num_warps", "BLOCK_K", "BLOCK_M"},
}


class AbductionTuner:
    """Hypothesis-driven kernel tuning. Same interface as CoordescTuner."""

    def __init__(self, is_mm=False, name="unknown", size_hints=None,
                 inductor_meta=None, frozen_fields=None, **kwargs):
        self.is_mm = is_mm
        self.name = name
        self.size_hints = size_hints
        self.inductor_meta = inductor_meta or {}
        self.frozen_fields = set(frozen_fields) if frozen_fields else set()
        self.call_count = 0

    @property
    def tunable_fields(self) -> list[str]:
        out = ["XBLOCK", "YBLOCK", "ZBLOCK", "R0_BLOCK", "R1_BLOCK",
               "BLOCK_M", "BLOCK_N", "BLOCK_K", "num_warps"]
        if self.is_mm:
            out.append("num_stages")
        return [f for f in out if f not in self.frozen_fields]

    def _get_field(self, config, name):
        if name == "num_warps": return config.num_warps
        if name == "num_stages": return config.num_stages
        return config.kwargs.get(name, None)

    def _set_field(self, config, name, value):
        if name == "num_warps": config.num_warps = value
        elif name == "num_stages": config.num_stages = value
        else: config.kwargs[name] = value

    def _neighbours(self, config, field_name):
        """Generate neighbour configs for a single field."""
        val = self._get_field(config, field_name)
        if val is None:
            return []
        candidates = []
        for next_val in [val * 2, val // 2]:
            if next_val <= 0 or next_val > 65536:
                continue
            c = copy.deepcopy(config)
            self._set_field(c, field_name, next_val)
            candidates.append((field_name, next_val, c))
        return candidates

    def _try_one(self, func, config):
        """Benchmark a single config. Returns timing or inf on failure."""
        try:
            t = func(config)
            self.call_count += 1
            return t
        except Exception:
            return float("inf")

    def autotune(self, func: Callable, baseline_config, baseline_timing=None) -> object:
        if baseline_timing is None:
            baseline_timing = self._try_one(func, baseline_config)

        best = baseline_config
        best_time = baseline_timing

        improved = True
        while improved:
            improved = False

            # Try every field. Collect both improvements and regressions.
            improvements: list[tuple[str, object, float]] = []
            regressions: list[tuple[str, object, float]] = []

            for field_name in self.tunable_fields:
                for fname, fval, candidate in self._neighbours(best, field_name):
                    t = self._try_one(func, candidate)
                    if t < best_time * 0.999:
                        improvements.append((fname, candidate, t))
                    elif t > best_time * 1.001:
                        regressions.append((fname, candidate, t))

            # Accept the best direct improvement
            if improvements:
                improvements.sort(key=lambda x: x[2])
                winner_field, best, best_time = improvements[0]
                improved = True

                # Follow transition edges from the winner
                for _ in range(20):
                    edges = TRANSITIONS.get(winner_field, set())
                    active = [f for f in self.tunable_fields if f in edges]
                    if not active:
                        break
                    found = False
                    for field_name in active:
                        for fname, fval, candidate in self._neighbours(best, field_name):
                            t = self._try_one(func, candidate)
                            if t < best_time * 0.999:
                                best, best_time = candidate, t
                                winner_field = fname
                                found = True
                                break
                        if found:
                            break
                    if not found:
                        break
                continue

            # No direct improvements. Speculative depth: for each regression
            # or lateral move, tentatively apply it and check if a
            # transition-graph neighbour rescues the position.
            # Lateral moves (same cost, different config) are the key —
            # they signal two forces cancelling, not "nothing happened."
            laterals: list[tuple[str, object, float]] = []
            for field_name in self.tunable_fields:
                for fname, fval, candidate in self._neighbours(best, field_name):
                    t = self._try_one(func, candidate)
                    if abs(t - best_time) <= best_time * 0.01:
                        laterals.append((fname, candidate, t))

            speculative = regressions + laterals
            for spec_field, spec_config, spec_time in speculative:
                edges = TRANSITIONS.get(spec_field, set()) - {spec_field}
                rescued = False
                for rescue_field in edges:
                    if rescue_field in self.frozen_fields:
                        continue
                    for _, _, rescue_candidate in self._neighbours(spec_config, rescue_field):
                        t = self._try_one(func, rescue_candidate)
                        if t < best_time * 0.999:
                            best, best_time = rescue_candidate, t
                            improved = True
                            rescued = True
                            break
                    if rescued:
                        break
                if rescued:
                    break

        return best


# --- Synthetic cost functions ---

def make_coupled_landscape(optimal: dict, coupling: dict[tuple[str, str], float]):
    """Cost function where parameters interact.
    coupling[(A,B)] = strength means A and B have a joint optimum
    that coordesc can't find by walking one axis at a time.
    """
    call_count = [0]
    def cost(config):
        call_count[0] += 1
        total = 0.0
        for field, opt_val in optimal.items():
            val = config.kwargs.get(field, getattr(config, field, opt_val))
            if val > 0 and opt_val > 0:
                log_ratio = math.log2(val / opt_val)
                total += log_ratio ** 2
        for (a, b), strength in coupling.items():
            va = config.kwargs.get(a, getattr(config, a, 1))
            vb = config.kwargs.get(b, getattr(config, b, 1))
            oa = optimal.get(a, va)
            ob = optimal.get(b, vb)
            if va > 0 and vb > 0 and oa > 0 and ob > 0:
                total += strength * (math.log2(va/oa) + math.log2(vb/ob)) ** 2
        return 1.0 + total
    return cost, call_count


def make_saddle_landscape(optimal: dict, saddle_field: str, saddle_depth: float = 0.3):
    """Cost function with a saddle point: moving saddle_field alone looks bad,
    but moving it WITH the right companion field finds the global optimum.
    """
    call_count = [0]
    companions = list(TRANSITIONS.get(saddle_field, set()))[:2]

    def cost(config):
        call_count[0] += 1
        total = 0.0
        for field, opt_val in optimal.items():
            val = config.kwargs.get(field, getattr(config, field, opt_val))
            if val > 0 and opt_val > 0:
                log_ratio = math.log2(val / opt_val)
                total += log_ratio ** 2

        sf_val = config.kwargs.get(saddle_field, getattr(config, saddle_field, 1))
        sf_opt = optimal.get(saddle_field, sf_val)

        if sf_val != sf_opt and companions:
            companion_aligned = 0
            for comp in companions:
                cv = config.kwargs.get(comp, getattr(config, comp, 1))
                co = optimal.get(comp, cv)
                if cv == co:
                    companion_aligned += 1
            if companion_aligned == 0:
                total += saddle_depth
        return 1.0 + total
    return cost, call_count


def make_non_separable(field_a: str, field_b: str,
                       joint_optima: list[tuple[dict, float]]):
    """Non-separable cost: optimal value of field_a depends on field_b.

    joint_optima: list of (config_dict, basin_cost) pairs.
    Each basin attracts configs nearby. The lowest-cost basin is the global optimum.
    A single-axis move from one basin toward another crosses a ridge.
    """
    call_count = [0]

    def cost(config):
        call_count[0] += 1
        best_basin_cost = float("inf")
        for center, base in joint_optima:
            dist = 0.0
            for field, opt_val in center.items():
                val = config.kwargs.get(field, getattr(config, field, opt_val))
                if val > 0 and opt_val > 0:
                    dist += math.log2(val / opt_val) ** 2
            basin_cost = base + dist
            best_basin_cost = min(best_basin_cost, basin_cost)
        return best_basin_cost

    return cost, call_count


# --- Benchmark runner ---

def run_comparison(name, cost_fn, call_count, start_config, is_mm=False):
    """Run both tuners on the same cost function, compare results."""

    # Coordesc
    call_count[0] = 0
    coordesc = CoordescTuner(is_mm=is_mm, name=f"coordesc_{name}",
                             size_hints={"x": 65536, "y": 65536, "z": 65536,
                                         "r0": 65536, "r1": 65536})
    t0 = time.perf_counter()
    coordesc_result = coordesc.autotune(cost_fn, copy.deepcopy(start_config))
    coordesc_wall = time.perf_counter() - t0
    coordesc_trials = call_count[0]
    coordesc_cost = cost_fn(coordesc_result)

    # Abduction
    call_count[0] = 0
    abduction = AbductionTuner(is_mm=is_mm, name=f"abduct_{name}")
    t0 = time.perf_counter()
    abduction_result = abduction.autotune(cost_fn, copy.deepcopy(start_config))
    abduction_wall = time.perf_counter() - t0
    abduction_trials = call_count[0]
    abduction_cost = cost_fn(abduction_result)

    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    print(f"  {'':20s} {'Coordesc':>12s} {'Abduction':>12s} {'Winner':>10s}")
    print(f"  {'Trials':20s} {coordesc_trials:12d} {abduction_trials:12d} "
          f"{'ABDUCT' if abduction_trials < coordesc_trials else 'COORD':>10s}")
    print(f"  {'Final cost':20s} {coordesc_cost:12.4f} {abduction_cost:12.4f} "
          f"{'ABDUCT' if abduction_cost < coordesc_cost else 'COORD':>10s}")
    print(f"  {'Wall time (ms)':20s} {coordesc_wall*1e3:12.2f} {abduction_wall*1e3:12.2f}")
    print(f"  Coordesc result: {coordesc_result}")
    print(f"  Abduction result: {abduction_result}")

    return {
        "name": name,
        "coordesc_trials": coordesc_trials, "abduction_trials": abduction_trials,
        "coordesc_cost": coordesc_cost, "abduction_cost": abduction_cost,
    }


if __name__ == "__main__":
    results = []

    # Test 1: Coupled BLOCK_M and num_warps
    # Optimal: BLOCK_M=128, BLOCK_N=64, BLOCK_K=32, num_warps=8
    # Start:   BLOCK_M=32,  BLOCK_N=32, BLOCK_K=32, num_warps=4
    # Coupling: BLOCK_M and num_warps must move together
    optimal_mm = {"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32}
    coupling = {("BLOCK_M", "num_warps"): 2.0}
    cost_fn, cc = make_coupled_landscape(optimal_mm, coupling)
    start = FakeConfig({"BLOCK_M": 32, "BLOCK_N": 32, "BLOCK_K": 32},
                       num_warps=4, num_stages=2)
    results.append(run_comparison("coupled_mm", cost_fn, cc, start, is_mm=True))

    # Test 2: Saddle on BLOCK_M — moving BLOCK_M alone looks worse
    optimal_saddle = {"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 64}
    cost_fn, cc = make_saddle_landscape(optimal_saddle, "BLOCK_M", saddle_depth=0.5)
    start = FakeConfig({"BLOCK_M": 64, "BLOCK_N": 32, "BLOCK_K": 16},
                       num_warps=4, num_stages=2)
    results.append(run_comparison("saddle_BLOCK_M", cost_fn, cc, start, is_mm=True))

    # Test 2b: Non-separable — optimal BLOCK_M depends on BLOCK_K
    # Local: (M=64, K=64) cost=2.0. Global: (M=128, K=32) cost=1.0.
    # Moving M alone: (M=128, K=64) is far from both basins = expensive
    # Moving K alone: (M=64, K=32) is far from both basins = expensive
    # Moving BOTH: (M=128, K=32) lands in global basin = cheap
    cost_fn, cc = make_non_separable("BLOCK_M", "BLOCK_K", [
        ({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 64}, 2.0),   # local minimum
        ({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32}, 1.0),   # global minimum
    ])
    start = FakeConfig({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 64},
                       num_warps=4, num_stages=2)
    results.append(run_comparison("non_sep_M_K", cost_fn, cc, start, is_mm=True))

    # Test 2c: Non-separable — XBLOCK depends on num_warps
    # Local: (X=128, warps=4) cost=2.0. Global: (X=512, warps=8) cost=1.0.
    cost_fn, cc = make_non_separable("XBLOCK", "num_warps", [
        ({"XBLOCK": 128, "YBLOCK": 1}, 2.0),   # local minimum (warps=4 implicit)
        ({"XBLOCK": 512, "YBLOCK": 1}, 1.0),   # global minimum (warps=8)
    ])
    # num_warps in FakeConfig, not kwargs — need to include in optima
    cost_fn2, cc2 = make_non_separable("XBLOCK", "num_warps", [
        ({"XBLOCK": 128, "YBLOCK": 1, "num_warps": 4}, 2.0),
        ({"XBLOCK": 512, "YBLOCK": 1, "num_warps": 8}, 1.0),
    ])
    start = FakeConfig({"XBLOCK": 128, "YBLOCK": 1}, num_warps=4)
    results.append(run_comparison("non_sep_X_warps", cost_fn2, cc2, start))

    # Test 3: Pointwise kernel — XBLOCK and num_warps coupled
    optimal_pw = {"XBLOCK": 1024, "YBLOCK": 1}
    coupling_pw = {("XBLOCK", "num_warps"): 1.5}
    cost_fn, cc = make_coupled_landscape(optimal_pw, coupling_pw)
    start = FakeConfig({"XBLOCK": 128, "YBLOCK": 1}, num_warps=4)
    results.append(run_comparison("pointwise_coupled", cost_fn, cc, start))

    # Test 4: High-dimensional — 6 fields, all interacting
    optimal_hd = {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64,
                  "XBLOCK": 512, "YBLOCK": 4, "R0_BLOCK": 128}
    coupling_hd = {
        ("BLOCK_M", "BLOCK_K"): 1.0,
        ("BLOCK_N", "num_warps"): 1.5,
        ("XBLOCK", "R0_BLOCK"): 0.8,
    }
    cost_fn, cc = make_coupled_landscape(optimal_hd, coupling_hd)
    start = FakeConfig({"BLOCK_M": 32, "BLOCK_N": 32, "BLOCK_K": 16,
                        "XBLOCK": 64, "YBLOCK": 1, "R0_BLOCK": 32},
                       num_warps=4, num_stages=2)
    results.append(run_comparison("high_dim_coupled", cost_fn, cc, start, is_mm=True))

    # Summary
    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")
    abduct_wins_trials = sum(1 for r in results if r["abduction_trials"] < r["coordesc_trials"])
    abduct_wins_cost = sum(1 for r in results if r["abduction_cost"] < r["coordesc_cost"])
    total_coordesc = sum(r["coordesc_trials"] for r in results)
    total_abduct = sum(r["abduction_trials"] for r in results)
    print(f"  Abduction wins on trials: {abduct_wins_trials}/{len(results)}")
    print(f"  Abduction wins on cost:   {abduct_wins_cost}/{len(results)}")
    print(f"  Total trials — coordesc: {total_coordesc}, abduction: {total_abduct} "
          f"({total_abduct/total_coordesc:.1%})")
