# Ship-form C++ patch sketch

The Python `propose.py` proves the mechanism. The actual fix ships as a small branch added inside `sumproduct_pair` in `aten/src/ATen/native/Linear.cpp`, just before the unconditional `at::bmm` call at line 263.

## Diff sketch

```diff
--- a/aten/src/ATen/native/Linear.cpp
+++ b/aten/src/ATen/native/Linear.cpp
@@ -257,6 +257,17 @@ static Tensor sumproduct_pair(const Tensor& left_, const Tensor& right_, IntArr
     for (auto it = ro.cbegin(); it != ro.cend(); i++, it++) {
       opermutation[*it] = i;
     }
   }

   // now we can execute the operations above
   left = left.permute(lpermutation).reshape_symint({lro_size, std::move(lo_size), sum_size});
   right = right.permute(rpermutation).reshape_symint({std::move(lro_size), std::move(sum_size), std::move(ro_size)});
-  Tensor result = at::bmm(left, right);
+  // Small-K batched-dot fast path: when bmm degenerates to (B, 1, K) @ (B, K, 1) with K small,
+  // cuBLAS dispatches a batched GEMV that under-utilizes the GPU due to per-batch launch overhead.
+  // Replace with elementwise mul + sum-reduction. See issue #101249.
+  // Threshold determined from prework/einsum-small-k/bench.py: 5-19x speedup at K<=32; bmm catches up at K>=64.
+  constexpr int64_t kSmallKBatchedDotThreshold = 32;
+  Tensor result;
+  bool small_k_batched_dot = (lo_size_orig == 1) && (ro_size_orig == 1) &&
+                             sum_size_for_check.guard_int(__FILE__, __LINE__) <= kSmallKBatchedDotThreshold;
+  if (small_k_batched_dot) {
+    // left: (B, 1, K), right: (B, K, 1) → squeeze, mul, sum, reshape back to (B, 1, 1)
+    auto acc_dtype = (left.scalar_type() == kHalf || left.scalar_type() == kBFloat16) ? kFloat : left.scalar_type();
+    result = (left.squeeze(-2) * right.squeeze(-1)).sum(-1, /*keepdim=*/true, acc_dtype).to(left.scalar_type()).unsqueeze(-1);
+  } else {
+    result = at::bmm(left, right);
+  }
   result = result.view_symint(out_size).permute(opermutation);
```

(Note: `lo_size_orig` and `ro_size_orig` need to be saved before the `std::move`s since the originals are consumed; the patch needs minor refactoring of the surrounding code to preserve them, OR move the check earlier.)

## Cleaner version

Compute `lo_size` and `ro_size` as plain `int64_t`s up front (they are known before the reshape) and use them in the branch:

```cpp
const int64_t lo_size_int = lo_size.guard_int(__FILE__, __LINE__);
const int64_t ro_size_int = ro_size.guard_int(__FILE__, __LINE__);
const int64_t sum_size_int = sum_size.guard_int(__FILE__, __LINE__);

left = left.permute(lpermutation).reshape_symint({lro_size, lo_size, sum_size});
right = right.permute(rpermutation).reshape_symint({lro_size, sum_size, ro_size});

constexpr int64_t kSmallKBatchedDotThreshold = 32;
Tensor result;
if (lo_size_int == 1 && ro_size_int == 1 && sum_size_int <= kSmallKBatchedDotThreshold) {
  // Batched dot pattern at small K: cuBLAS bmm here launches one GEMV per batch chunk
  // and amortizes badly. Equivalent computation as elementwise mul + sum is faster.
  // See issue #101249 and prework/einsum-small-k/bench.py.
  auto acc_dtype = (left.scalar_type() == kHalf || left.scalar_type() == kBFloat16)
                   ? kFloat : left.scalar_type();
  result = (left.squeeze(-2) * right.squeeze(-1))
             .sum(-1, /*keepdim=*/true, acc_dtype)
             .to(left.scalar_type())
             .unsqueeze(-1);
} else {
  result = at::bmm(left, right);
}
```

(SymInt `guard_int` may not work in all dynamic-shape paths — for production, the check should use `TORCH_GUARD_OR_FALSE(lo_size.sym_eq(1))` and friends, which handle symbolic shapes correctly. The cliff fix only NEEDS to fire on concrete-shape inputs anyway — symbolic shapes are infrequent in eager.)

## Why this is small-and-safe

- The branch fires ONLY when `lo_size == 1 AND ro_size == 1 AND sum_size <= 32`. This is the exact "diag-of-bmm" pattern from issue #101249 — narrow scope.
- For any other pattern, `at::bmm(left, right)` is called unchanged. Behavior preserved bit-exact for the 99% of einsum cases.
- Numerical result is within fp tolerance (validated 42/42 in `validate.py`). Accumulation in fp32 for fp16/bf16 matches cuBLAS's TC accumulator behavior.
- Threshold of 32 leaves K=64+ on the bmm path where cuBLAS catches up (per H₂ data: K=64 → bmm wins 0.77x vs manual). Conservative — could be tuned higher per-device but 32 captures the steepest cliff.

## What requires a rebuild

The patch is C++ and ships in `Linear.cpp`. To bench in-tree:

```bash
cd /mnt/d/depot/pytorch
pip install -e . -v --no-build-isolation
```

This is a 30–60 min build on first invocation; incremental rebuilds touching only Linear.cpp should be ~3–5 min. Out of scope for the prework session; queued as Phase 6.

## What gets shipped to upstream

- The 10-line patch above
- A unit test in `test/test_linalg.py` `test_einsum` that exercises `Nc,Nc->N` at K=2, 4, 8, 16, 32 across fp32/fp16/bf16, and verifies output equals `(a*b).sum(-1)` within tolerance
- A bench addition (optional) in `benchmarks/dynamo/microbenchmarks/operatorbench.py` or similar
- PR body cites issue #101249 + prework/einsum-small-k/ as provenance
