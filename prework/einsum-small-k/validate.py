"""Numerical equivalence: fast_einsum matches torch.einsum within fp tolerance
across every shape in shapes.py and every dtype.

For float16/bfloat16 the reduction order matters (fast path uses elementwise+sum,
bmm may use TC fp32 accumulators). Use looser tolerances on low-precision dtypes.
"""
import sys
import torch
sys.path.insert(0, '.')

from propose import fast_einsum, _try_batched_dot_fastpath
from shapes import CLIFF_SHAPES, NONCLIFF_SHAPES, DTYPES

TOLERANCES = {
    torch.float32:  (1e-4, 1e-5),  # rtol, atol
    torch.float16:  (5e-2, 5e-2),  # half: needs loose atol; per-element values can be ~O(1)
    torch.bfloat16: (5e-2, 1e-1),  # bf16 has 7-bit mantissa — even loose atol
}


def check_one(eq, a_shape, b_shape, dtype, desc):
    torch.manual_seed(0)
    a = torch.randn(a_shape, device='cuda', dtype=dtype)
    b = torch.randn(b_shape, device='cuda', dtype=dtype)
    fast = fast_einsum(eq, a, b)
    ref = torch.einsum(eq, a, b)
    rtol, atol = TOLERANCES[dtype]
    ok = torch.allclose(fast, ref, rtol=rtol, atol=atol)
    if not ok:
        diff = (fast.float() - ref.float()).abs()
        return False, f"  FAIL {desc} eq={eq} dtype={dtype}: max_abs_diff={diff.max().item():.3e}, rel={diff.max().item()/(ref.float().abs().max().item() + 1e-30):.3e}"
    return True, f"  ok   {desc} eq={eq} dtype={dtype}"


def check_pattern_detection():
    """Confirm fast path FIRES for cliff shapes (returns non-None) and does NOT
    fire for non-cliff shapes (returns None, so torch.einsum is used unchanged).
    """
    print("--- Fast-path detection ---")
    fail = 0
    for eq, (sa, sb), desc in CLIFF_SHAPES:
        a = torch.randn(sa, device='cuda', dtype=torch.float32)
        b = torch.randn(sb, device='cuda', dtype=torch.float32)
        out = _try_batched_dot_fastpath(eq, a, b)
        ok = out is not None
        print(f"  {'FIRES' if ok else 'MISS '}  {desc} eq={eq} a={sa} b={sb}")
        if not ok: fail += 1

    for eq, (sa, sb), desc in NONCLIFF_SHAPES:
        a = torch.randn(sa, device='cuda', dtype=torch.float32)
        b = torch.randn(sb, device='cuda', dtype=torch.float32)
        out = _try_batched_dot_fastpath(eq, a, b)
        ok = out is None
        print(f"  {'GUARD' if ok else 'OVERAPPLY!'}  {desc} eq={eq} a={sa} b={sb}")
        if not ok: fail += 1
    return fail == 0


def check_numerics():
    """Confirm fast_einsum == torch.einsum within tolerance on every shape and dtype."""
    print("\n--- Numerical equivalence (cliff shapes) ---")
    pass_count = 0
    fail_count = 0
    for eq, (sa, sb), desc in CLIFF_SHAPES:
        for dtype in DTYPES:
            ok, msg = check_one(eq, sa, sb, dtype, desc)
            print(msg)
            if ok: pass_count += 1
            else:  fail_count += 1

    print("\n--- Numerical equivalence (non-cliff shapes — should fall through to torch.einsum) ---")
    for eq, (sa, sb), desc in NONCLIFF_SHAPES:
        for dtype in DTYPES:
            ok, msg = check_one(eq, sa, sb, dtype, desc)
            print(msg)
            if ok: pass_count += 1
            else:  fail_count += 1

    print(f"\nNumerics: {pass_count} pass / {fail_count} fail")
    return fail_count == 0


if __name__ == "__main__":
    detect_ok = check_pattern_detection()
    numerics_ok = check_numerics()
    sys.exit(0 if (detect_ok and numerics_ok) else 1)
