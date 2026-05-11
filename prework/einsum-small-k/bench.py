"""Bench: fast_einsum vs torch.einsum vs reference (manual) on cliff and
non-cliff shapes. Verifies the fast path is faster on cliff shapes and not
slower on non-cliff shapes (where it falls through to torch.einsum).
"""
import sys
import time
import torch
import torch.utils.benchmark as bench
sys.path.insert(0, '.')

from propose import fast_einsum
from shapes import CLIFF_SHAPES, NONCLIFF_SHAPES


def bench_one(eq, sa, sb, dtype=torch.float32, min_run_time=0.4):
    torch.manual_seed(0)
    a = torch.randn(sa, device='cuda', dtype=dtype)
    b = torch.randn(sb, device='cuda', dtype=dtype)

    # Warmup
    for _ in range(5):
        torch.einsum(eq, a, b)
        fast_einsum(eq, a, b)
        (a * b).sum(-1) if (eq == "Nc,Nc->N" or eq == "ij,ij->i") else None
    torch.cuda.synchronize()

    t_einsum = bench.Timer(stmt="torch.einsum(eq, a, b)",
                           globals={"torch": torch, "eq": eq, "a": a, "b": b}
                          ).blocked_autorange(min_run_time=min_run_time)
    t_fast = bench.Timer(stmt="fast_einsum(eq, a, b)",
                         globals={"fast_einsum": fast_einsum, "eq": eq, "a": a, "b": b}
                        ).blocked_autorange(min_run_time=min_run_time)
    return t_einsum.median * 1e6, t_fast.median * 1e6


def main():
    print(f"torch {torch.__version__}, device {torch.cuda.get_device_name()}\n")

    print("=== CLIFF SHAPES (fast path should win) ===")
    print(f"{'desc':<48s}{'einsum µs':>12s}{'fast µs':>12s}{'speedup':>10s}")
    cliff_speedups = []
    for eq, (sa, sb), desc in CLIFF_SHAPES:
        e_us, f_us = bench_one(eq, sa, sb)
        sp = e_us / f_us
        cliff_speedups.append(sp)
        marker = "WIN " if sp >= 1.5 else ("tie " if sp >= 0.9 else "LOSE")
        print(f"{desc:<48s}{e_us:>12.2f}{f_us:>12.2f}{sp:>10.2f}x [{marker}]")

    print(f"\n  cliff geomean speedup: {torch.tensor(cliff_speedups).log().mean().exp().item():.2f}x")
    print(f"  cliff min/max speedup: {min(cliff_speedups):.2f}x / {max(cliff_speedups):.2f}x")

    print("\n=== NON-CLIFF SHAPES (fast path should NOT fire — must not regress) ===")
    print(f"{'desc':<48s}{'einsum µs':>12s}{'fast µs':>12s}{'ratio':>10s}")
    noncliff_ratios = []
    for eq, (sa, sb), desc in NONCLIFF_SHAPES:
        e_us, f_us = bench_one(eq, sa, sb)
        r = f_us / e_us
        noncliff_ratios.append(r)
        marker = "ok  " if r <= 1.10 else ("MILD" if r <= 1.30 else "REGR")
        print(f"{desc:<48s}{e_us:>12.2f}{f_us:>12.2f}{r:>10.2f}x [{marker}]")

    print(f"\n  non-cliff worst-case fast/einsum: {max(noncliff_ratios):.3f}x (must be ≤ ~1.10)")


if __name__ == "__main__":
    main()
