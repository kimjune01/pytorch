"""Verify the patch is in the binary by checking output equivalence + bench
across the cliff shapes. Bypasses test_linalg.py (which has torchvision
import issues in the container).
"""
import torch
import torch.utils.benchmark as bench

print(f"torch {torch.__version__}, file: {torch.__file__}")
print(f"cuda: {torch.cuda.is_available()}, version: {torch.version.cuda}")
print(f"device: {torch.cuda.get_device_name() if torch.cuda.is_available() else 'cpu'}")
print()


def manual(a, b):
    return (a * b).sum(-1)


def check_correctness(eq, a, b, dtype, name):
    fast = torch.einsum(eq, a, b)
    if eq == 'Nc,Nc->N' or eq == 'i,i->' or eq == 'ij,ij->i':
        ref = (a.float() * b.float()).sum(-1).to(dtype)
    elif eq == 'bhij,bhij->bhi':
        ref = (a.float() * b.float()).sum(-1).to(dtype)
    else:
        ref = torch.einsum(eq, a, b)
    if dtype in (torch.float16, torch.bfloat16):
        rtol, atol = 5e-2, 5e-2
    else:
        rtol, atol = 1e-4, 1e-5
    ok = torch.allclose(fast, ref, rtol=rtol, atol=atol)
    diff = (fast.float() - ref.float()).abs().max().item()
    return ok, diff


print("=== Correctness (patched torch.einsum vs reference) ===")
for dtype in [torch.float32, torch.float16, torch.bfloat16]:
    for N, c in [(1024, 2), (1024, 4), (1024, 8), (1024, 16), (1024, 32), (1024, 64)]:
        torch.manual_seed(0)
        a = torch.randn(N, c, device='cuda', dtype=dtype)
        b = torch.randn(N, c, device='cuda', dtype=dtype)
        ok, diff = check_correctness('Nc,Nc->N', a, b, dtype, f"N={N},c={c}")
        marker = "ok" if ok else "FAIL"
        print(f"  [{marker}] {dtype} N={N} c={c}: max_diff={diff:.3e}")

print()
print("=== Bench (cliff shapes) ===")
print(f"{'N':>10s} {'c':>4s} {'manual µs':>11s} {'einsum µs':>11s} {'ratio':>8s} {'verdict':>10s}")

for N, c in [(1048576, 2), (1048576, 4), (524288, 8), (262144, 16), (131072, 32),
             (131072, 64), (16384, 256)]:
    a = torch.randn(N, c, device='cuda', dtype=torch.float32)
    b = torch.randn(N, c, device='cuda', dtype=torch.float32)
    for _ in range(5):
        manual(a, b); torch.einsum('Nc,Nc->N', a, b)
    torch.cuda.synchronize()
    tm = bench.Timer(stmt="manual(a,b)", globals={"manual": manual, "a": a, "b": b}).blocked_autorange(min_run_time=0.4)
    te = bench.Timer(stmt="torch.einsum('Nc,Nc->N', a, b)",
                     globals={"torch": torch, "a": a, "b": b}).blocked_autorange(min_run_time=0.4)
    m_us, e_us = tm.median*1e6, te.median*1e6
    ratio = e_us / m_us
    if c <= 32:
        verdict = "PARITY" if 0.85 <= ratio <= 1.15 else ("STILL CLIFF" if ratio > 1.5 else "WIN")
    else:
        verdict = "ok" if ratio <= 1.10 else "regression"
    print(f"{N:>10d} {c:>4d} {m_us:>11.2f} {e_us:>11.2f} {ratio:>7.3f}x {verdict:>10s}")

print()
print("=== Diag-of-attention pattern ===")
for B, H, T, d in [(2, 8, 64, 4), (4, 16, 32, 8), (1, 16, 128, 16)]:
    a = torch.randn(B, H, T, d, device='cuda', dtype=torch.float32)
    b = torch.randn(B, H, T, d, device='cuda', dtype=torch.float32)
    for _ in range(5):
        torch.einsum('bhij,bhij->bhi', a, b); (a*b).sum(-1)
    torch.cuda.synchronize()
    tm = bench.Timer(stmt="(a*b).sum(-1)", globals={"a": a, "b": b}).blocked_autorange(min_run_time=0.3)
    te = bench.Timer(stmt="torch.einsum('bhij,bhij->bhi', a, b)",
                     globals={"torch": torch, "a": a, "b": b}).blocked_autorange(min_run_time=0.3)
    print(f"  B={B} H={H} T={T} d={d}: manual={tm.median*1e6:.2f}us einsum={te.median*1e6:.2f}us ratio={te.median/tm.median:.3f}x")
