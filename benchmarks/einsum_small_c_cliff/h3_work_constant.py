"""H3 work-constant sweep: N*c = 2^21 held constant.

Tests whether c=64 einsum-wins finding is artifact of manual becoming
bandwidth-bound at large c, or einsum genuinely amortizing GEMM overhead.
"""
import torch
import torch.utils.benchmark as bench

assert torch.cuda.is_available(), "CUDA required"
device = torch.device("cuda")
dtype = torch.float64  # 8 bytes; matches GB/s formula (8*N*c + 8*N)

cells = [
    (2097152, 1), (1048576, 2), (524288, 4), (262144, 8), (131072, 16),
    (65536, 32), (32768, 64), (16384, 128), (8192, 256), (4096, 512), (2048, 1024),
]

def bench_fn(stmt, globs, label):
    t = bench.Timer(stmt=stmt, globals=globs, label=label)
    m = t.blocked_autorange(min_run_time=0.4)
    return m.median * 1e6  # us

print(f"{'N':>10} {'c':>6} {'manual_us':>12} {'einsum_us':>12} {'manual_GBps':>13} {'einsum_GBps':>13} {'ratio(m/e)':>12} {'equiv':>6}")
print("-" * 100)

results = []
for N, c in cells:
    torch.manual_seed(0)
    a = torch.randn(N, c, device=device, dtype=dtype)
    b = torch.randn(N, c, device=device, dtype=dtype)

    # warm + correctness
    out_manual = (a * b).sum(-1)
    out_einsum = torch.einsum("Nc,Nc->N", a, b)
    equiv = torch.allclose(out_manual, out_einsum, rtol=1e-10, atol=1e-10)

    g = {"a": a, "b": b, "torch": torch}
    manual_us = bench_fn("(a*b).sum(-1)", g, f"manual N={N} c={c}")
    einsum_us = bench_fn("torch.einsum('Nc,Nc->N', a, b)", g, f"einsum N={N} c={c}")

    # bandwidth: read 2*N*c floats (8 bytes), write N floats (8 bytes)
    bytes_total = 8 * (2 * N * c + N)
    manual_GBps = bytes_total / (manual_us * 1e-6) / 1e9
    einsum_GBps = bytes_total / (einsum_us * 1e-6) / 1e9
    ratio = manual_us / einsum_us

    print(f"{N:>10} {c:>6} {manual_us:>12.2f} {einsum_us:>12.2f} {manual_GBps:>13.2f} {einsum_GBps:>13.2f} {ratio:>12.3f} {str(equiv):>6}")
    results.append((N, c, manual_us, einsum_us, manual_GBps, einsum_GBps, ratio, equiv))

# crossover: smallest c where ratio >= 1.0 (einsum <= manual)
print()
print("Crossover (smallest c where einsum <= manual):")
crossover = None
for N, c, mu, eu, mg, eg, r, eq in results:
    if r >= 1.0:
        crossover = c
        print(f"  c = {c}: manual={mu:.2f}us einsum={eu:.2f}us ratio={r:.3f}")
        break
if crossover is None:
    print("  no crossover found in sweep")

print()
print("Trend summary:")
print(f"  manual GB/s range: {min(r[4] for r in results):.2f} -> {max(r[4] for r in results):.2f}")
print(f"  einsum GB/s range: {min(r[5] for r in results):.2f} -> {max(r[5] for r in results):.2f}")
print(f"  manual us range:   {min(r[2] for r in results):.2f} -> {max(r[2] for r in results):.2f}")
print(f"  einsum us range:   {min(r[3] for r in results):.2f} -> {max(r[3] for r in results):.2f}")
