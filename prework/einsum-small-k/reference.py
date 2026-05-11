"""Reference implementations: explicit elementwise+reduction lowerings for the
einsum patterns under test. These are what the cliff codepath SHOULD compile
to (and what fast_einsum produces). Used by validate.py to check numerics.
"""
import torch


def ref_Nc_Nc_to_N(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """torch.einsum('Nc,Nc->N', a, b)"""
    return (a * b).sum(-1)


def ref_bhij_bhij_to_bhi(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """torch.einsum('bhij,bhij->bhi', a, b) — the diag-of-attention pattern."""
    return (a * b).sum(-1)


def ref_bnk_bk_to_bn(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """torch.einsum('bnk,bk->bn', a, b) — batched mat-vec contracted on k."""
    return (a * b.unsqueeze(1)).sum(-1)


def ref_ij_ij_to_i(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """torch.einsum('ij,ij->i', a, b) — same as Nc,Nc->N"""
    return (a * b).sum(-1)
