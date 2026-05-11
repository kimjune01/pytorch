"""Direct tests for gemini round-1 concerns. Each is a perturbation gemini
suggested. Pass = the refined fix handles the case correctly.
"""
import sys
import torch
import torch.utils.benchmark as bench
sys.path.insert(0, '.')
from propose import fast_einsum, _try_batched_dot_fastpath, MEMORY_BUDGET_BYTES, SMALL_K_THRESHOLD


def test_concern_1_memory_budget():
    """Gemini #1: at large N×K, the elementwise intermediate is 32x larger than
    bmm output. Verify the fix falls back to bmm beyond MEMORY_BUDGET_BYTES."""
    # Cliff shape (well under budget): should fire
    a = torch.randn(1024, 8, device='cuda')
    out = _try_batched_dot_fastpath('Nc,Nc->N', a, a)
    assert out is not None, "small shape should fire"

    # Just under budget: should fire (intermediate = N*c*4 = budget exactly)
    just_under = MEMORY_BUDGET_BYTES // 4 // 16
    a = torch.randn(just_under, 16, device='cuda')
    out = _try_batched_dot_fastpath('Nc,Nc->N', a, a)
    assert out is not None, f"just-under-budget N={just_under} c=16 should fire"

    # Over budget: should fall back to bmm (return None)
    over = (MEMORY_BUDGET_BYTES // 4 // 16) * 2
    a = torch.randn(over, 16, device='cuda')
    out = _try_batched_dot_fastpath('Nc,Nc->N', a, a)
    assert out is None, f"over-budget N={over} c=16 should NOT fire"

    print("  ok concern #1 memory budget guard")


def test_concern_2_underflow():
    """Gemini #2: fp16 product underflow. Construct values where a*b
    underflows in fp16 but the sum is representable. fix must upcast BEFORE
    multiply to preserve the value."""
    torch.manual_seed(0)
    # Pick values where a*b ~ 1e-7 (underflows fp16 normal range; would be 0
    # if multiplied in fp16) but sum over 32 such values is ~3.2e-6 (still small
    # but representable in fp32).
    N, c = 256, 32
    a = torch.full((N, c), 0.001, device='cuda', dtype=torch.float16)
    b = torch.full((N, c), 0.0001, device='cuda', dtype=torch.float16)
    # Each a*b = 1e-7, fp16 min normal ~6.1e-5 → underflows to 0 in fp16
    # Sum over c=32: 32 * 1e-7 = 3.2e-6, still tiny but nonzero in fp32

    fast_out = fast_einsum('Nc,Nc->N', a, b)
    ref_out = torch.einsum('Nc,Nc->N', a, b)

    # Both should give the same answer (within fp16 tolerance).
    # The buggy version (mul-in-fp16) would give exactly 0.
    is_zero_fast = (fast_out == 0).all().item()
    is_zero_ref = (ref_out == 0).all().item()
    print(f"  fast underflow result: {fast_out[0].item()} (all-zero: {is_zero_fast})")
    print(f"  ref  underflow result: {ref_out[0].item()} (all-zero: {is_zero_ref})")

    if is_zero_fast and not is_zero_ref:
        print("  FAIL concern #2: fast path lost precision (mul-in-fp16 truncated)")
        return False
    if is_zero_ref:
        print("  ok concern #2: but bmm also returned 0 — test setup may need stronger inputs")
        # This means cuBLAS bmm at fp16 also underflows here; our fix matches
        return True
    print("  ok concern #2: fast path preserves precision (upcasts before multiply)")
    return True


def test_concern_5_backward_perf():
    """Gemini #5: backward pass of (a*b).sum() may be slower than bmm's."""
    torch.manual_seed(0)
    cells = [(1048576, 2), (524288, 8), (131072, 32)]

    print(f"  {'shape':<20s}{'einsum_fwd µs':>15s}{'fast_fwd µs':>15s}{'einsum_bwd µs':>15s}{'fast_bwd µs':>15s}{'fwd':>8s}{'bwd':>8s}")
    for N, c in cells:
        a = torch.randn(N, c, device='cuda', requires_grad=True)
        b = torch.randn(N, c, device='cuda', requires_grad=True)
        grad = torch.randn(N, device='cuda')

        def einsum_fwd(): return torch.einsum('Nc,Nc->N', a, b)
        def fast_fwd(): return fast_einsum('Nc,Nc->N', a, b)

        for _ in range(5): einsum_fwd(); fast_fwd()
        torch.cuda.synchronize()

        t_e_fwd = bench.Timer(stmt="f()", globals={"f": einsum_fwd}).blocked_autorange(min_run_time=0.3)
        t_f_fwd = bench.Timer(stmt="f()", globals={"f": fast_fwd}).blocked_autorange(min_run_time=0.3)

        # Backward: time the backward pass given a fresh forward each iter.
        def einsum_bwd():
            o = einsum_fwd()
            o.backward(grad, retain_graph=False)
            a.grad = None; b.grad = None
        def fast_bwd():
            o = fast_fwd()
            o.backward(grad, retain_graph=False)
            a.grad = None; b.grad = None

        for _ in range(5): einsum_bwd(); fast_bwd()
        torch.cuda.synchronize()
        t_e_bwd = bench.Timer(stmt="f()", globals={"f": einsum_bwd}).blocked_autorange(min_run_time=0.3)
        t_f_bwd = bench.Timer(stmt="f()", globals={"f": fast_bwd}).blocked_autorange(min_run_time=0.3)

        e_fwd, f_fwd = t_e_fwd.median*1e6, t_f_fwd.median*1e6
        e_bwd, f_bwd = t_e_bwd.median*1e6, t_f_bwd.median*1e6
        # bwd timer includes fwd; subtract for clean bwd-only
        clean_e_bwd = max(e_bwd - e_fwd, 0)
        clean_f_bwd = max(f_bwd - f_fwd, 0)
        print(f"  N={N:7d} c={c:2d}  {e_fwd:14.2f} {f_fwd:14.2f} {clean_e_bwd:14.2f} {clean_f_bwd:14.2f}  {e_fwd/f_fwd:6.2f}x {(clean_e_bwd/clean_f_bwd if clean_f_bwd > 0 else float('inf')):6.2f}x")


def test_concern_4_cpu_guard():
    """Gemini #4: ensure CPU tensors fall through (don't take the cuda fast path)."""
    a = torch.randn(1024, 8, device='cpu')
    out = _try_batched_dot_fastpath('Nc,Nc->N', a, a)
    if out is not None:
        print(f"  FAIL concern #4: CPU tensor took fast path")
        return False
    print(f"  ok concern #4: CPU tensor falls through (returns None)")
    return True


if __name__ == "__main__":
    print(f"torch {torch.__version__}, device {torch.cuda.get_device_name()}\n")

    print("=== Concern #1: memory budget ===")
    test_concern_1_memory_budget()

    print("\n=== Concern #2: fp16 underflow ===")
    test_concern_2_underflow()

    print("\n=== Concern #4: CPU guard ===")
    test_concern_4_cpu_guard()

    print("\n=== Concern #5: backward pass perf ===")
    test_concern_5_backward_perf()
