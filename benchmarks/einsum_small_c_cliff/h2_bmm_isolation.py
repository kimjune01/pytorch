"""
H2 isolation benchmark for PyTorch issue #101249.

Tests whether the einsum("Nc,Nc->N", a, b) cliff at small c lives in
cuBLAS (slow small-K GEMM) or in framework dispatch above it.

We bench:
  manual      : (a*b).sum(-1)                                          [fast reference]
  einsum      : torch.einsum("Nc,Nc->N", a, b)                          [the cliff]
  bmm_form    : torch.bmm(a.unsqueeze(1), b.unsqueeze(2)).squeeze()     [suspected lowering]
  matmul_form : (a.unsqueeze(1) @ b.unsqueeze(2)).squeeze()             [matmul lowering]

If bmm at K=c is ~30x slower than manual, cuBLAS is the culprit (H2 confirmed).
If bmm is fast (~manual), einsum framework overhead dominates (H2 refuted).
"""

import time
import torch
import torch.utils.benchmark as benchmark


SHAPES = [
    (1048576, 2),
    (1048576, 4),
    (524288, 8),
    (131072, 64),
]


def make_inputs(N, c, device="cuda", dtype=torch.float32):
    a = torch.randn(N, c, device=device, dtype=dtype)
    b = torch.randn(N, c, device=device, dtype=dtype)
    return a, b


def fn_manual(a, b):
    return (a * b).sum(-1)


def fn_einsum(a, b):
    return torch.einsum("Nc,Nc->N", a, b)


def fn_bmm(a, b):
    # a:(N,c) -> (N,1,c); b:(N,c) -> (N,c,1); bmm -> (N,1,1) -> (N,)
    return torch.bmm(a.unsqueeze(1), b.unsqueeze(2)).squeeze(-1).squeeze(-1)


def fn_matmul(a, b):
    # (N,1,c) @ (N,c,1) -> (N,1,1) -> (N,)
    return (a.unsqueeze(1) @ b.unsqueeze(2)).squeeze(-1).squeeze(-1)


FNS = [
    ("manual", fn_manual),
    ("einsum", fn_einsum),
    ("bmm_form", fn_bmm),
    ("matmul_form", fn_matmul),
]


def verify_equivalence(N, c):
    a, b = make_inputs(N, c)
    ref = fn_manual(a, b)
    for name, fn in FNS[1:]:
        out = fn(a, b)
        if out.shape != ref.shape:
            raise RuntimeError(f"shape mismatch {name}: {out.shape} vs {ref.shape}")
        diff = (out - ref).abs().max().item()
        if diff > 1e-4 * max(1.0, ref.abs().max().item()):
            raise RuntimeError(f"value mismatch {name}: max abs diff {diff}")
    print(f"  verified equivalence at (N={N}, c={c})")


def bench_one(fn, a, b, label):
    t = benchmark.Timer(
        stmt="fn(a, b); torch.cuda.synchronize()",
        globals={"fn": fn, "a": a, "b": b, "torch": torch},
        label=label,
    )
    m = t.blocked_autorange(min_run_time=0.4)
    return m.median  # seconds


def isolate_einsum_parse_overhead(N, c, n_iter=1000):
    """Measure pure Python-side einsum overhead by feeding tiny tensors so
    the actual GPU work is negligible — what's left is parsing + dispatch."""
    a = torch.randn(8, c, device="cuda")
    b = torch.randn(8, c, device="cuda")
    # warm
    for _ in range(10):
        torch.einsum("Nc,Nc->N", a, b)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(n_iter):
        torch.einsum("Nc,Nc->N", a, b)
    torch.cuda.synchronize()
    einsum_per_call = (time.perf_counter() - t0) / n_iter

    for _ in range(10):
        fn_bmm(a, b)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iter):
        fn_bmm(a, b)
    torch.cuda.synchronize()
    bmm_per_call = (time.perf_counter() - t0) / n_iter

    for _ in range(10):
        fn_manual(a, b)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iter):
        fn_manual(a, b)
    torch.cuda.synchronize()
    manual_per_call = (time.perf_counter() - t0) / n_iter

    return einsum_per_call, bmm_per_call, manual_per_call


def main():
    assert torch.cuda.is_available(), "CUDA required"
    print(f"torch {torch.__version__}  device={torch.cuda.get_device_name(0)}")
    print()

    print("Verifying numerical equivalence...")
    for N, c in SHAPES:
        verify_equivalence(N, c)
    print()

    # warmup
    a, b = make_inputs(1024, 8)
    for _ in range(5):
        for _, fn in FNS:
            fn(a, b)
    torch.cuda.synchronize()

    # Main table
    header = f"{'(N, c)':>16} | " + " | ".join(f"{name:>11}" for name, _ in FNS) + " | " + " | ".join(f"{name+'/manual':>14}" for name, _ in FNS[1:])
    print(header)
    print("-" * len(header))

    rows = []
    for N, c in SHAPES:
        a, b = make_inputs(N, c)
        # warmup these tensors
        for _, fn in FNS:
            for _ in range(3):
                fn(a, b)
        torch.cuda.synchronize()

        times = {}
        for name, fn in FNS:
            times[name] = bench_one(fn, a, b, f"{name}@N={N},c={c}")
        rows.append((N, c, times))

        manual_t = times["manual"]
        time_strs = " | ".join(f"{times[name]*1e6:>9.2f}us" for name, _ in FNS)
        ratio_strs = " | ".join(f"{times[name]/manual_t:>13.2f}x" for name, _ in FNS[1:])
        print(f"  (N={N:>7}, c={c:>2}) | {time_strs} | {ratio_strs}")

    print()
    print("Isolating einsum parse/dispatch overhead with tiny inputs (N=8)...")
    print("(GPU work is negligible at N=8, so per-call time approximates Python+dispatch overhead)")
    for N, c in SHAPES:
        e, bm, ma = isolate_einsum_parse_overhead(8, c)
        print(f"  c={c:>2}  einsum/call={e*1e6:>7.2f}us  bmm/call={bm*1e6:>7.2f}us  manual/call={ma*1e6:>7.2f}us  einsum-bmm overhead={(e-bm)*1e6:>6.2f}us")


if __name__ == "__main__":
    main()
