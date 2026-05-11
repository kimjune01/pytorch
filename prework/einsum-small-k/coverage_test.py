"""Comprehensive einsum-pattern coverage test.

Ports test_linalg.py's test_einsum patterns and verifies fast_einsum produces
identical (within tolerance) results to torch.einsum on EVERY pattern, including:
- patterns where the fast path SHOULD NOT fire (most of them)
- the dot pattern 'i,i->' where it SHOULD fire
- patterns with multiple operands (>2) — the wrapper passes through

Phase 5.5 regression check at the Python level. The actual C++ ship will need
this test ported into test/test_linalg.py.
"""
import sys
import torch
sys.path.insert(0, '.')

from propose import fast_einsum, _try_batched_dot_fastpath


def make(*shape, dtype=torch.float32):
    return torch.randn(*shape, device='cuda', dtype=dtype) * 0.5


def check(eq, *operands, dtype_name="fp32", desc=""):
    fast = fast_einsum(eq, *operands)
    ref = torch.einsum(eq, *operands)
    ok = torch.allclose(fast, ref, rtol=5e-3, atol=1e-3) if operands[0].dtype == torch.float32 \
         else torch.allclose(fast, ref, rtol=5e-2, atol=1e-1)
    fired = (len(operands) == 2 and _try_batched_dot_fastpath(eq, operands[0], operands[1]) is not None)
    tag = "FIRES" if fired else "skip "
    if not ok:
        diff = (fast.float() - ref.float()).abs()
        return False, f"  FAIL [{tag}] {dtype_name} '{eq}' {desc} max_diff={diff.max().item():.3e}"
    return True, f"  ok   [{tag}] {dtype_name} '{eq}' {desc}"


def run_test_einsum_patterns(dtype):
    """Mirror test_linalg.py test_einsum patterns."""
    name = {torch.float32: "fp32", torch.float16: "fp16", torch.bfloat16: "bf16"}[dtype]
    torch.manual_seed(0)
    x = make(5, dtype=dtype)
    y = make(7, dtype=dtype)
    A = make(3, 5, dtype=dtype)
    B = make(2, 5, dtype=dtype)
    C = make(2, 3, 5, dtype=dtype)
    D = make(2, 5, 7, dtype=dtype)
    E = make(7, 9, dtype=dtype)
    F = make(2, 3, 3, 5, dtype=dtype)
    G = make(5, 4, 6, dtype=dtype)
    H = make(4, 4, dtype=dtype)
    I = make(2, 3, 2, dtype=dtype)

    cases = [
        # Vector ops
        ('i->',          (x,),          "vector sum"),
        ('i,i->',        (x, x),        "dot — fast path SHOULD fire"),
        ('i,i->i',       (x, x),        "elementwise mul"),
        ('i,j->ij',      (x, y),        "outer product"),
        # Matrix ops
        ("ij->ji",       (A,),          "transpose"),
        ("ij->j",        (A,),          "row sum"),
        ("ij->i",        (A,),          "col sum"),
        ("ij,ij->ij",    (A, A),        "matrix elementwise"),
        ("ij,ij->i",     (A, A),        "diag-of-bmm — fast path SHOULD fire"),
        ("ij,j->i",      (A, x),        "mat-vec"),
        ("ij,kj->ik",    (A, B),        "matmul"),
        ("ij,ab->ijab",  (A, E),        "matrix outer"),
        # Tensor ops
        ("Aij,Ajk->Aik", (C, D),        "batched matmul"),
        ("ijk,jk->i",    (C, A),        "tensor mat contraction"),
        ("aij,jk->aik",  (D, E),        "tensor mat"),
        ("ijk,jk->ik",   (C, A),        "tensor with double indices"),
        ("ijk,jk->ij",   (C, A),        "tensor with double indices v2"),
        ("ijk,ik->j",    (C, B),        "non-contiguous"),
        ("ijk,ik->jk",   (C, B),        "non-contiguous double"),
        # Diagonals & traces
        ("ii",           (H,),          "trace"),
        ("ii->i",        (H,),          "diagonal"),
        ('iji->j',       (I,),          "non-contig trace"),
        # Ellipsis
        ("i...->...",    (H,),          "ellipsis"),
        # Small-K cliff cases (force the fast path)
        ('Nc,Nc->N',     (make(1024, 4, dtype=dtype), make(1024, 4, dtype=dtype)), "cliff K=4"),
        ('Nc,Nc->N',     (make(2048, 16, dtype=dtype), make(2048, 16, dtype=dtype)), "cliff K=16"),
        ('Nc,Nc->N',     (make(4096, 32, dtype=dtype), make(4096, 32, dtype=dtype)), "boundary K=32"),
        ('Nc,Nc->N',     (make(2048, 64, dtype=dtype), make(2048, 64, dtype=dtype)), "above-threshold K=64"),
        ('bhij,bhij->bhi', (make(2, 8, 64, 4, dtype=dtype), make(2, 8, 64, 4, dtype=dtype)), "diag-attn small"),
    ]
    pass_n, fail_n = 0, 0
    for eq, ops, desc in cases:
        ok, msg = check(eq, *ops, dtype_name=name, desc=desc)
        print(msg)
        if ok: pass_n += 1
        else: fail_n += 1
    return pass_n, fail_n


def main():
    print(f"torch {torch.__version__}, device {torch.cuda.get_device_name()}\n")
    total_pass, total_fail = 0, 0
    for dtype in [torch.float32, torch.float16, torch.bfloat16]:
        print(f"=== {dtype} ===")
        p, f = run_test_einsum_patterns(dtype)
        total_pass += p
        total_fail += f
        print(f"  subtotal: {p} pass / {f} fail\n")

    print(f"=== TOTAL: {total_pass} pass / {total_fail} fail ===")
    return 0 if total_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
