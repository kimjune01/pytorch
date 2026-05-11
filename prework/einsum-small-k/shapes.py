"""Test matrix: shapes that exercise the cliff AND shapes that should NOT take
the fast path (regression coverage).
"""
import torch


# Cliff-bearing shapes (the fast path should fire and win)
CLIFF_SHAPES = [
    # (equation, (a_shape, b_shape), description)
    ("Nc,Nc->N",          ((1048576, 2),                  (1048576, 2)),                  "issue #101249 cliff at c=2"),
    ("Nc,Nc->N",          ((1048576, 4),                  (1048576, 4)),                  "issue #101249 cliff at c=4"),
    ("Nc,Nc->N",          ((524288, 8),                   (524288, 8)),                   "cliff at c=8"),
    ("Nc,Nc->N",          ((262144, 16),                  (262144, 16)),                  "cliff at c=16"),
    ("Nc,Nc->N",          ((131072, 32),                  (131072, 32)),                  "boundary at threshold"),
    ("ij,ij->i",          ((4096, 8),                     (4096, 8)),                     "small-N variant"),
    ("bhij,bhij->bhi",    ((4, 16, 256, 4),               (4, 16, 256, 4)),               "diag-of-attention pattern, head_dim=4"),
    ("bhij,bhij->bhi",    ((4, 16, 1024, 8),              (4, 16, 1024, 8)),              "diag-of-attention, head_dim=8"),
]

# Above-threshold shapes (the fast path should NOT fire; einsum unchanged)
NONCLIFF_SHAPES = [
    ("Nc,Nc->N",          ((131072, 64),                  (131072, 64)),                  "above threshold — bmm wins"),
    ("Nc,Nc->N",          ((16384, 256),                  (16384, 256)),                  "well above threshold"),
    ("ij,jk->ik",         ((512, 256),                    (256, 1024)),                   "regular matmul — no fast path"),
    ("bij,bjk->bik",      ((32, 128, 64),                 (32, 64, 256)),                 "bmm — no fast path"),
    ("Nc,Md->NM",         ((1024, 8),                     (2048, 8)),                     "outer product — no fast path"),
    ("...ij,...jk->...ik",((4, 16, 32, 64),               (4, 16, 64, 32)),               "ellipsis bmm — no fast path"),
]

DTYPES = [torch.float32, torch.float16, torch.bfloat16]
