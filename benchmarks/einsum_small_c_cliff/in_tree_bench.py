"""In-tree bench for the einsum small-K batched-dot fast path (issue #101249).

Run with the patched PyTorch build (the C++ branch in Linear.cpp::sumproduct_pair).
Compares torch.einsum('Nc,Nc->N', a, b) against the explicit (a*b).sum(-1)
lowering across the cliff shapes from the issue.

Pre-patch behavior (cu130 wheel): einsum is 5-30x slower at small K.
Post-patch behavior: einsum should match (a*b).sum(-1) within a few %.

Usage (from the pytorch repo root, after build):
    python benchmarks/einsum_small_c_cliff/in_tree_bench.py
"""
import json
import os
import sys
import torch
import torch.utils.benchmark as bench

assert torch.cuda.is_available(), "CUDA required for the in-tree bench"

print(f"torch {torch.__version__}, device {torch.cuda.get_device_name()}")
print(f"is_built_from_source: {'+git' in torch.__version__ or '+dev' in torch.__version__}")
print()


def manual(a, b):
    return (a * b).sum(-1)

def via_einsum(a, b):
    return torch.einsum('Nc,Nc->N', a, b)


def bench_one(N, c, dtype=torch.float32, min_run_time=0.4):
    torch.manual_seed(0)
    a = torch.randn(N, c, device='cuda', dtype=dtype)
    b = torch.randn(N, c, device='cuda', dtype=dtype)
    for _ in range(5):
        manual(a, b); via_einsum(a, b)
    torch.cuda.synchronize()
    tm = bench.Timer(stmt="manual(a, b)", globals={"manual": manual, "a": a, "b": b}).blocked_autorange(min_run_time=min_run_time)
    te = bench.Timer(stmt="via_einsum(a, b)", globals={"via_einsum": via_einsum, "a": a, "b": b}).blocked_autorange(min_run_time=min_run_time)
    return tm.median * 1e6, te.median * 1e6


cells = [
    (1048576, 2),
    (1048576, 4),
    (524288, 8),
    (262144, 16),
    (131072, 32),
    (131072, 64),    # above-threshold; should fall through to bmm
    (16384, 256),    # well-above; bmm still wins
]

print(f"{'N':>10s} {'c':>4s}  {'manual us':>12s}  {'einsum us':>12s}  {'einsum/manual':>13s}")
results = []
for N, c in cells:
    try:
        m_us, e_us = bench_one(N, c)
        ratio = e_us / m_us
        marker = "PARITY" if 0.95 <= ratio <= 1.10 else ("CLIFF" if ratio > 1.20 else "ok")
        print(f"{N:>10d} {c:>4d}  {m_us:>12.2f}  {e_us:>12.2f}  {ratio:>12.3f}x [{marker}]")
        results.append({"N": N, "c": c, "manual_us": m_us, "einsum_us": e_us, "ratio": ratio})
    except Exception as ex:
        print(f"{N:>10d} {c:>4d}  FAILED: {ex}")

print()
cliff_shapes = [r for r in results if r["c"] <= 32]
above = [r for r in results if r["c"] > 32]
if cliff_shapes:
    geomean = torch.tensor([r["ratio"] for r in cliff_shapes]).log().mean().exp().item()
    print(f"Cliff shapes (K<=32) geomean einsum/manual: {geomean:.3f}x")
    print(f"  Pre-patch: 5-30x (cliff). Post-patch target: 1.0-1.2x (within bench noise).")
if above:
    above_geomean = torch.tensor([r["ratio"] for r in above]).log().mean().exp().item()
    print(f"Above-threshold (K>=64) geomean: {above_geomean:.3f}x")
    print(f"  Should fall through to bmm and match pre-patch behavior (~0.7-1.0x).")

with open(os.path.join(os.path.dirname(__file__), "in_tree_bench_results.json"), "w") as f:
    json.dump(results, f, indent=2)
