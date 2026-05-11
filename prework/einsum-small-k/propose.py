"""Proposed fast path for einsum/sumproduct_pair when the post-permute bmm is a
batched dot — i.e. lo_size == 1 AND ro_size == 1 (the contracted dim is the only
non-output dim shared between left and right).

Examples covered:
  "Nc,Nc->N"        — diag of pairwise contraction
  "bhij,bhij->bhi"  — diag-of-attention pattern, per (batch, head, i)
  "bnk,bk->bn"      — batched matrix-vector contracted to bn (when k is small)
  In general: any einsum where every non-summed dim appears in BOTH operands AND
  there are no left-only or right-only output dims.

The fast path replaces `bmm((B, 1, K), (B, K, 1))` with a fused-equivalent:
  - squeeze the size-1 dims
  - elementwise mul (one kernel)
  - sum along the contracted axis (one kernel)
  - reshape back to (B, 1, 1) so the rest of sumproduct_pair's reshape/permute
    chain is unaffected

Two kernels vs cuBLAS bmm's many launches at small K → expected 5-30x at the
cliff shapes from the issue.

For prework purposes this is a Python pattern detector that wraps torch.einsum.
The ship-form is a small branch added to aten/src/ATen/native/Linear.cpp at the
top of `sumproduct_pair`'s bmm call (line 263), gated on lo_size == 1 AND
ro_size == 1 AND sum_size <= K_THRESHOLD.
"""
import torch
from typing import Sequence


# Shape threshold below which the elementwise+sum fast path beats bmm.
# Determined empirically from the work-constant sweep (h3_work_constant.py):
# at constant N*c, einsum/manual ratio falls from 23x at c=2 to 1.0 at c=512.
# The conservative threshold of 32 gates the path off well before the crossover
# while capturing all of the steepest part of the cliff.
SMALL_K_THRESHOLD = 32

# Memory budget for the intermediate elementwise product. Above this, bmm is
# the only safe choice — elementwise materializes (lro_size * sum_size *
# itemsize) bytes, which can OOM at large N×K. Per gemini bug-hunt round 1:
# at N=100M, c=32 fp32, the intermediate is 12.8 GB. 256 MB matches typical
# scratch-buffer headroom on common GPUs.
MEMORY_BUDGET_BYTES = 256 * 1024 * 1024


def _try_batched_dot_fastpath(equation: str, a: torch.Tensor, b: torch.Tensor):
    """Detect the 'every non-summed dim is shared (bmm with M=1, N=1) AND
    summed dim is small' pattern. Returns the fast-path output, or None if the
    pattern doesn't apply.

    The detection is conservative — it only fires when every dim labeled the
    same in both operands is either an output dim (preserved) or a contracted
    dim (summed), with no broadcast or left-only/right-only semantics.
    """
    if "..." in equation:
        return None
    if equation.count(",") != 1 or "->" not in equation:
        return None

    lhs, rhs = equation.split("->")
    a_labels, b_labels = lhs.split(",")
    out_labels = rhs

    a_labels = a_labels.strip()
    b_labels = b_labels.strip()
    out_labels = out_labels.strip()

    # Both operands must have the same set of labels (no broadcast, no left/right-only)
    if set(a_labels) != set(b_labels):
        return None
    if not set(out_labels).issubset(set(a_labels)):
        return None

    # The contracted (summed) dims are the labels in a_labels but not in out_labels
    summed = [c for c in a_labels if c not in out_labels]
    if not summed:
        return None  # no contraction, einsum just reorders + multiplies — skip

    # All summed dims must be the same size in a and b
    a_sizes = dict(zip(a_labels, a.shape))
    b_sizes = dict(zip(b_labels, b.shape))
    for s in summed:
        if a_sizes[s] != b_sizes[s]:
            return None

    # The total summed size must be small (the cliff threshold)
    sum_size = 1
    for s in summed:
        sum_size *= a_sizes[s]
    if sum_size > SMALL_K_THRESHOLD:
        return None

    # CUDA-only — gemini round 1 #4: CPU/MKL has different cliff structure.
    if not a.is_cuda:
        return None

    # Memory budget guard — gemini round 1 #1: the elementwise product
    # materializes lro_size * sum_size * itemsize bytes. At N=100M c=32 fp32
    # that's 12.8 GB and OOMs. Fall back to bmm.
    out_size = 1
    for c in out_labels:
        out_size *= a_sizes[c]
    intermediate_bytes = out_size * sum_size * a.element_size()
    if intermediate_bytes > MEMORY_BUDGET_BYTES:
        return None

    # Permute both operands so output dims come first (in `out_labels` order),
    # then summed dims (in some consistent order). This is the same logical
    # transform sumproduct_pair does, but we keep the result aligned for elementwise.
    perm_order = list(out_labels) + summed
    a_perm = [a_labels.index(c) for c in perm_order]
    b_perm = [b_labels.index(c) for c in perm_order]
    a_t = a.permute(a_perm)
    b_t = b.permute(b_perm)

    # Elementwise mul + sum along the trailing summed axes.
    # For low-precision dtypes (fp16/bf16): upcast operands BEFORE multiply
    # to match cuBLAS Tensor Core behavior (TC keeps the product in 32-bit
    # registers throughout). Casting only the accumulator (gemini round 1 #2)
    # loses precision because the fp16 multiply truncates before the sum sees
    # the value. Subnormal/underflow inputs would silently zero out.
    if a.dtype in (torch.float16, torch.bfloat16):
        prod = a_t.to(torch.float32) * b_t.to(torch.float32)
        for _ in summed:
            prod = prod.sum(dim=-1)
        return prod.to(a.dtype)
    else:
        prod = a_t * b_t
        for _ in summed:
            prod = prod.sum(dim=-1)
        return prod


def fast_einsum(equation: str, *operands: torch.Tensor) -> torch.Tensor:
    """Drop-in replacement for torch.einsum that uses a fast path when the
    pattern matches the small-K diag-of-bmm shape; otherwise falls back to
    torch.einsum unchanged.
    """
    if len(operands) == 2:
        out = _try_batched_dot_fastpath(equation, operands[0], operands[1])
        if out is not None:
            return out
    return torch.einsum(equation, *operands)
