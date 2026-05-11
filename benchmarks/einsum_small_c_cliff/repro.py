"""Verify #101249: torch.einsum 'Nc,Nc->N' on CUDA, large N small c."""
import torch
import torch.utils.benchmark as bench

assert torch.cuda.is_available()

def manual(a, b): return (a * b).sum(dim=-1)
def via_einsum(a, b): return torch.einsum("Nc,Nc->N", a, b)

print(f"torch {torch.__version__}, device {torch.cuda.get_device_name()}")
print()

for N, c in [(1048576, 2), (1048576, 3), (1048576, 4), (524288, 8), (262144, 16), (131072, 64)]:
    a = torch.randn(N, c, device="cuda")
    b = torch.randn(N, c, device="cuda")
    for _ in range(5): manual(a, b); via_einsum(a, b)
    torch.cuda.synchronize()
    tm = bench.Timer(stmt="manual(a,b)", globals={"manual": manual, "a": a, "b": b}).blocked_autorange(min_run_time=0.4)
    te = bench.Timer(stmt="via_einsum(a,b)", globals={"via_einsum": via_einsum, "a": a, "b": b}).blocked_autorange(min_run_time=0.4)
    print(f"  N={N}, c={c:3d}:  manual={tm.median*1e6:7.2f}us  einsum={te.median*1e6:8.2f}us  ratio einsum/manual = {te.median/tm.median:6.2f}x")

# Also try a "real" transformer-shape: bmm-like
print("\n--- Other patterns (CUDA) ---")
for shape, expr in [
    (((4, 32, 128, 64), (4, 32, 128, 64)), "bhij,bhij->bhi"),  # query·key per-head
    (((1024, 1024, 8), (8,)), "bnk,k->bn"),
]:
    a = torch.randn(*shape[0], device="cuda")
    b = torch.randn(*shape[1], device="cuda")
    for _ in range(5): torch.einsum(expr, a, b)
    torch.cuda.synchronize()
    te = bench.Timer(stmt="torch.einsum(expr, a, b)", globals={"torch": torch, "expr": expr, "a": a, "b": b}).blocked_autorange(min_run_time=0.3)
    print(f"  '{expr}' shapes={shape}: einsum={te.median*1e6:.2f}us")
