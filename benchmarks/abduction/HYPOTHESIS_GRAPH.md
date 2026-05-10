# Hypothesis Graph: abduction port to PyTorch Inductor

Companion to [README.md](README.md). The README states what's being investigated and how. This file logs the actual evidence as it accrues.

---

## H0: Inductor's autotune is the right comparison surface for abduction — HYPOTHESIS

**Claim.** PyTorch Inductor's existing autotuners (`max_autotune`, `coordinate_descent_tuning`) operate over the same problem shape as speedygrad's abduction engine: pick the best configuration from a discrete-but-large search space using measurement.

**Why plausible.** Both search over (block sizes, num_warps, num_stages, layout choices). Both face the same compile-time vs quality tradeoff. Both have a "default heuristic" baseline they're trying to improve.

**Why it might be wrong.** Inductor's space is finite-grid (a small cartesian product of named knobs); abduction's tinygrad space is closer to a continuous DAG of opt actions. The shapes may be too different for the same search strategy to transfer.

**Falsifier.** If a port of abduction can't even be expressed against `AlgorithmSelectorCache` without violating the "two samples, one diff, follow the failure" loop, this is killed at the architectural level — no measurement needed.

**Status:** OPEN. Architectural read of `torch/_inductor/select_algorithm.py` not yet performed.

---

## H1: Abduction reaches coordinate_descent_tuning quality in fewer trials — HYPOTHESIS

**Pre-committed prediction (before any data).** On the frozen corpus, abduction reaches ≥95% of `coordinate_descent_tuning`'s chosen-kernel runtime in ≤30% of the trial count.

**Falsifier (strict, pre-committed).** Hypothesis killed if abduction needs ≥ coordinate_descent's trial count to reach within 5% of its quality.

**Why this matters even if killed.** A clean kill on a foreign codebase is itself evidence about what abduction *requires* — telling us whether the engine's wins depend on tinygrad's UOp shape or on the methodology itself. Either result is publishable.

**Status:** BLOCKED on baseline. Triton not available in current Windows + Python 3.14 environment. WSL move pending.

---

## Open frontier (in dependency order)

| # | Edge | Blocked by |
|---|---|---|
| 1 | Confirm WSL + CUDA passthrough + Triton work | user runs `wsl --install` |
| 2 | Re-run identity probe inside WSL | #1 |
| 3 | Read `torch/_inductor/select_algorithm.py`, decide if H0 survives architectural inspection | #2 |
| 4 | Freeze corpus from `benchmarks/dynamo/microbenchmarks/operator_inp_logs/` | #3 |
| 5 | Build baseline harness (default / max_autotune / coord_descent / oracle) | #4 |
| 6 | Run baseline, commit raw CSV + Pareto plot | #5 |
| 7 | Implement abduction as 4th selector in fork | #6 |
| 8 | Run abduction on same corpus, compare | #7 |

---

## Closed

### Identity probe caught the Triton blocker — CONFIRMED

Running `identity_probe.py` on Windows + Python 3.14 + torch 2.11.0+cu128 revealed Triton was not importable. Without Triton, Inductor autotune cannot run, and all three modes silently degrade to default ATen. If the baseline had been built first and run before the probe, the resulting CSV would have shown all three modes producing identical numbers — a confusing null result that would have wasted hours diagnosing.

Lesson: identity probes pay for themselves on first run. Always run before measurement.
