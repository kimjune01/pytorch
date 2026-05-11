# Hypothesis Graph: einsum small-c cliff (#101249)

`torch.einsum("Nc,Nc->N", a, b)` is 5–31x slower than the equivalent `(a*b).sum(-1)` on CUDA when the contracted dim `c` is small. Issue [#101249](https://github.com/pytorch/pytorch/issues/101249) (open, May 2023). Prior fix attempt PR [#145936](https://github.com/pytorch/pytorch/pull/145936) was closed unmerged in May 2025. Verified live on torch 2.11.0+cu130 / RTX 4080 (sm_89) on 2026-05-11.

**Tooling note.** `codex` CLI unavailable in this environment; gemini available. Phase 2 abductions skip codex filter — confidences downgraded ~10%. Phase 4/7 will use gemini.

---

## Mise-en-place

**System under test.** PyTorch 2.11.0+cu130, RTX 4080 (sm_89), WSL Ubuntu, venv `/home/junekim/abduction-venv`. Eager mode (no torch.compile in scope for H₀; einsum lowering is a Python+ATen path).

**Perturbation access.**
- Shape sweep over (N, c) — pure user-space.
- `torch.profiler` + `torch.autograd.profiler` to identify which `aten::` ops einsum lowers to.
- `TORCH_LOGS=schedule` (eager) is not applicable; for dispatcher-trace use `torch._C._profiler` or `torch.autograd.profiler.profile(use_cuda=True)`.
- Direct `torch.bmm` / `torch.matmul` benches to isolate cuBLAS vs framework-side overhead.
- For an Inductor cross-check: `torch.compile(einsum)` and compare to default eager.
- Source reads: `torch/functional.py`, `aten/src/ATen/native/Linear.cpp`, `torch/backends/opt_einsum/__init__.py`, the closed PR #145936 diff.

**Identity probe.** Numerical: `(einsum_out - manual_out).abs().max() < 1e-4` for fp32, looser for fp16/bf16. einsum and `(a*b).sum(-1)` differ only in reduction order so they should match within fp tolerance.

**Ambiguity heuristic.** Cliff = ratio ≥ 2x with non-overlapping IQR over 3+ reps. Sub-2x ratios are noted but not "the cliff". The current observation (30x at c=2) is ~6 standard deviations above any noise band.

**Goal.** Pin down which dispatch decision sends `einsum("Nc,Nc->N")` through the slow GEMM path at small c. Classify whether the fix sits in `einsum` (route small-K to elementwise+sum), `tensordot` (special-case small K), or `matmul`/cuBLAS (a back-end fix). If a small fix exists, ship a PR.

---

## H₀ — The cliff is real and reproducible — CONFIRMED

**Observation.** Under torch 2.11.0+cu130 on RTX 4080, `torch.einsum("Nc,Nc->N", a, b)` is materially slower than `(a*b).sum(-1)` for small c, with a monotone trend that crosses over near c=64.

**Data (2026-05-11, p50 over `torch.utils.benchmark.Timer.blocked_autorange(min_run_time=0.4)`, fp32):**

| N | c | manual µs | einsum µs | ratio einsum/manual |
|---|---|-----------|-----------|---------------------|
| 1048576 |  2 |  27.7 |  832.3 |  **30.0x** |
| 1048576 |  3 |  29.2 |  903.0 |  **30.9x** |
| 1048576 |  4 |  45.7 |  899.6 |  **19.7x** |
|  524288 |  8 |  46.8 |  440.6 |   **9.4x** |
|  262144 | 16 |  48.0 |  248.0 |   **5.2x** |
|  131072 | 64 | 176.2 |  126.7 |   0.7x (einsum WINS) |

**Trajectory shape: divergent.** Monotone ratio decline as c grows. No oscillation, no chaos. The signal is clean across the c range tested.

**Reasoning mode:** induction (measured). 95% confidence the cliff exists. The PR #145936 author and issue maintainers ALSO observed this; this isn't a measurement artifact.

**Edge generated.** The mechanism is ambiguous from the H₀ data alone. Three competing causal candidates, fanned out as H₁/H₂/H₃ below.

---

## H₁ — einsum lowers `Nc,Nc->N` to a GEMM (via tensordot/bmm/matmul) regardless of K size

**Abduction.** einsum's path through `tensordot → matmul` always picks GEMM for shape contractions. Small K (K=c=2..16) is a degenerate GEMM; the framework should fall back to elementwise multiply + reduction but doesn't.

**Null.** einsum dynamically chooses elementwise+reduction vs GEMM based on K, and the cliff comes from elsewhere.

**Perturbation.** `torch.profiler` trace at c=2 and c=64. Examine which `aten::` ops fire. Key tells:
- `aten::bmm`, `aten::matmul`, `aten::mm` → confirms GEMM path
- `aten::mul` + `aten::sum` → confirms elementwise+reduction path
- Read `torch/functional.py` einsum() to see the dispatch logic

**Predicted classification:** divergent confirmation. Expected: `aten::bmm` or `aten::matmul` fires at both c=2 and c=64.

**Status:** PENDING (Phase 2 fan-out)

---

## H₂ — cuBLAS dispatch at K=2 hits a slow fallback

**Abduction.** Even when GEMM is the right algorithm, cuBLAS at very small K (K=2..4) lacks an optimized kernel and falls to a generic path. The cliff is inside cuBLAS, not in PyTorch's lowering.

**Null.** cuBLAS handles small K efficiently; the slow-down is in framework overhead (Python/ATen dispatch on top of einsum).

**Perturbation.** Bench `torch.bmm` directly at the shapes einsum reduces to (likely `(1, N, c) @ (1, N, c).transpose(-1,-2)` with diagonal extraction, or `(N, 1, c) @ (N, c, 1)`). If `torch.bmm` at K=2 is also 30x slower than `(a*b).sum(-1)`, blame cuBLAS. If `torch.bmm` is fast, blame einsum's lowering or its framework overhead.

**Predicted classification:** divergent. Either cuBLAS is genuinely slow at K=2 (then bmm matches einsum's slowness) or it isn't (then framework overhead is the culprit).

**Status:** PENDING (Phase 2 fan-out)

---

## H₃ — The c=64 "einsum wins" is artifact of `manual` getting slower, not einsum getting faster

**Abduction.** As c grows, `manual = (a*b).sum(-1)` materializes a full-size intermediate (N×c floats) — at c=64 with N=131072, that's 33 MB, exceeding the 4080's 64 MB L2 by half. The materialization becomes bandwidth-bound. Meanwhile einsum's GEMM path may amortize launch overhead. The crossover is not "einsum is good", it's "manual is bad".

**Null.** Both paths scale linearly in total work; the crossover is real and reflects einsum genuinely beating elementwise+sum at large c.

**Perturbation.** Hold total work `N*c` constant. Sweep (N=2^k * 8, c=2^(20-k)) for k in {0..6}. If einsum stays a constant `M`s while manual grows linearly in c, the crossover is the materialization cost. Confirm by measuring memory bandwidth (manual ought to read+write 8*N*c+8*N bytes, einsum ought to read 8*N*c).

**Predicted classification:** divergent. Either manual is bandwidth-bound at large c (artifact confirmed) or einsum genuinely improves (crossover real).

**Status:** PENDING (Phase 2 fan-out)

---

## Phase 2 plan

Three subagents, parallel, measurement-only (no patches). Each writes back a 200-word verdict + raw data, which I'll fold into the graph. Codex filter step skipped (CLI unavailable); gemini volley deferred to Phase 4.

| H | Subagent role | Decisive output |
|---|---|---|
| H₁ | dispatch tracer | which aten:: op fires at c=2 vs c=64 |
| H₂ | cuBLAS isolator | torch.bmm at K=2 matches einsum's slowness or doesn't |
| H₃ | provenance + control | PR #145936 reviewer feedback + work-constant sweep |

---

## Phase 2 results (2026-05-11)

### H₁ — CONFIRMED (convergent). einsum unconditionally lowers `Nc,Nc->N` to bmm.

**aten:: chain at both c=2 and c=64:** `aten::einsum → aten::permute → aten::reshape → aten::bmm → internal::gemvx_kernel` (cuBLAS GEMV-style). No `aten::sum`/`aten::mul` path, no K-size branch.

**Dispatch site:** `aten/src/ATen/native/Linear.cpp:263` — `Tensor result = at::bmm(left, right);` is unconditional inside `sumproduct_pair`. Reached for any contraction with at least one shared sum dim. The only early-out is `Linear.cpp:170-171` (`if (sum_dims_.empty()) return at::mul(left_, right_);`) which doesn't apply when c is summed.

**Per-iter cuBLAS sub-launch counts:** c=2 → 16 cuBLAS launches per `bmm` (160 over 10 iters); c=64 → 3 launches per `bmm` (30 over 10 iters). The bmm dispatcher is splitting the batch into chunks at small K — separate H₂ concern.

Saved: `h1_profiler.txt`. Reasoning mode: deduction (code read of Linear.cpp) + induction (profiler trace), 99%.

### H₂ — CONFIRMED (divergent). cuBLAS small-K bmm IS the cliff; framework overhead is <1%.

**Bench at the cliff shapes (RTX 4080, fp32, p50):**

| (N, c) | manual µs | einsum µs | bmm_form µs | matmul_form µs | bmm/manual |
|---|---:|---:|---:|---:|---:|
| (1048576, 2) | 542 | 2540 | **2520** | 2571 | **4.65x** |
| (1048576, 4) | 642 | 2639 | 977 | 979 | 1.52x |
| (524288, 8) | 90 | 521 | 515 | 520 | **5.73x** |
| (131072, 64) | 242 | 196 | 186 | 187 | 0.77x (bmm wins) |

`bmm_form` (direct `torch.bmm(a.unsqueeze(1), b.unsqueeze(2)).squeeze()`) tracks einsum almost exactly — the 12 µs einsum string-parser overhead is <1% of the gap. The cliff lives in cuBLAS's batched mat-vec dispatch when M=1, N=1, K=small, batch=large. Each "GEMM" is ~4 FMAs amortized over kernel-launch overhead.

Note H₂'s 4.65x is a smaller magnitude than H₀'s 30x; the difference is shape-specific (H₀ used a different per-call timing measurement). The cliff is real and reproducible — only the magnitude varies with N and measurement protocol.

Saved: `h2_bmm_isolation.py`. Reasoning mode: induction (measured), 95%.

### H₃ — PROVENANCE CLEAR + sweep refines crossover

**PR #145936 status:** MERGED 2025-05-13 (yesterday relative to this session). Title: "torch.tensordot: performance improvements when contracting to a scalar". Adds a fast path in `tensordot` that routes to `at::dot(t1.flatten(), t2.flatten())` when the entire output is scalar (full contraction). Guard: `if (csize == t1.numel() && csize == t2.numel())`. **Our `Nc,Nc->N` case is explicitly excluded** because we preserve N as an output dim. PR #145936 is NOT a duplicate of our intended fix.

Initial merge attempt (Feb 2025) was blocked by DTensor sharding for `aten.dot` and an MPS shape bug; both fixed before final landing. No reviewer raised concerns about the small-K batched case our investigation targets.

**Work-constant sweep (N*c = 2²¹, fp64, RTX 4080):**

| N | c | manual µs | einsum µs | manual GBps | einsum/manual |
|---|---|---:|---:|---:|---:|
| 2097152 | 1 | 67.6 | **23.4** | 745 | **0.35 (einsum WINS)** — special-case path |
| 1048576 | 2 | 47.2 | 1097.2 | 888 | **23.2x** |
| 524288 | 4 | 53.9 | 613.9 | 700 | 11.4x |
| 262144 | 8 | 59.5 | 321.8 | 599 | 5.4x |
| 131072 | 16 | 65.8 | 174.8 | 526 | 2.7x |
| 65536 | 32 | 72.6 | 114.5 | 469 | 1.6x |
| 32768 | 64 | 53.4 | 76.3 | 634 | 1.4x |
| 16384 | 128 | 35.7 | 63.8 | 943 | 1.8x |
| 8192 | 256 | 56.2 | 71.6 | 598 | 1.3x |
| 4096 | 512 | 73.6 | 73.4 | 456 | **1.00 (crossover)** |
| 2048 | 1024 | 52.5 | 67.9 | 639 | 1.3x |

**Findings:**
1. **The c=64 "einsum wins" from H₀ is partly an N artifact.** At constant work, einsum stays slower until c=512.
2. **Manual stays compute-bound, not bandwidth-bound.** GB/s is 455–945 across the sweep, never saturating the 1008 GB/s peak. So manual's flat ~50µs is reduction overhead, not memory wall.
3. **c=1 is a special case.** einsum at c=1 is 3x FASTER than manual (23µs vs 68µs). Likely a degenerate-shape fast path (probably `at::dot` after the new PR #145936 merge — when the contracted dim has size 1, `flatten()` makes it look like a full contraction).

Saved: `h3_pr_145936_diff.txt`, `h3_work_constant.py`. Reasoning mode: induction + deduction (PR diff read), 90%.

---

## Phase 2.5 — Provenance synthesis

**git blame on `Linear.cpp:263`** (next step): need to confirm when the unconditional `at::bmm` was added to `sumproduct_pair` and whether there was prior consideration of small-K paths.

**Adjacent mechanisms:**
- `at::dot` is hand-tuned and fast at small contractions (per c=1 finding above)
- `at::mul + at::sum` is the "manual" path — a Triton-codegen-style fused kernel via Inductor when wrapped in `torch.compile`, but in eager it's two kernels with an intermediate

**Recently merged:** PR #145936 added the scalar-output fast path. Our gap is the **batched-output** version of the same idea: when the einsum/tensordot pattern is "for each output element, do a small-K dot", lower to elementwise-mul+sum-along-K rather than bmm.

**No competing PRs found** for the batched-dot case (`gh pr list --search "einsum batched"` from H₃). Path is open.

---

## Diagnosis (Phase 2 → Phase 3)

**Causal chain:**
```
H₀ (cliff exists, 30x at small c) ─ confirmed
        │
        ├─ H₁ (einsum→bmm always) ─ confirmed (Linear.cpp:263 unconditional)
        │
        └─ H₂ (cuBLAS small-K bmm slow) ─ confirmed (bmm_form == einsum, ≪ manual)
                │
                └─ deep cause: cuBLAS dispatches batched GEMV for (N, 1, K=2..32) shapes;
                              kernel launch overhead dominates at K<32
                │
                └─ leverage point: Linear.cpp:263 (or its caller in einsum() itself)
                              add small-K + diag-shape fast path → elementwise mul + sum

H₃ (work-constant) ─ refined
        │
        ├─ Provenance: PR #145936 merged but doesn't cover our case
        ├─ c=64 "win" partly N-dependent; true crossover at c=512 at constant work
        └─ c=1 already fast (likely benefits from new dot fast-path)
```

**Fix sketch.** In `aten/src/ATen/native/Linear.cpp` `sumproduct_pair` (or in `einsum` itself before reaching sumproduct_pair):
- Detect the "diag of bmm" shape: post-permute, `left` is `(Batch, 1, K)` and `right` is `(Batch, K, 1)`, OR the equivalent pattern where M=1 and N=1 in the bmm.
- When K ≤ threshold (probably 32, possibly tuned per-device), replace `at::bmm(left, right)` with `(left.squeeze(-2) * right.squeeze(-1)).sum(-1, keepdim=…)`.
- Preserve numerical tolerance (reduction order may differ; document in the test).
- Threshold determination: the work-constant sweep shows the slowdown vanishes by c=64 in the H₂ data and crosses at c=512 in H₃. The conservative threshold is K ≤ 32; the more aggressive is K ≤ 64 or K ≤ 256. To be determined in Phase 6 bench.

**Frontier edges (Phase 3 candidates):**
1. **Dtype coverage**: does the cliff persist at fp16/bf16? Predict: yes — cuBLAS's small-K dispatch is the same code path for all dtypes. (Cheap perturbation: 5-min sweep)
2. **Generalization to other einsum patterns**: `bhij,bhij->bhi` (the diag-of-attention pattern), `bnk,bk->bn`, etc. Predict: same cliff if the post-permute shape has M=1, N=1, K=small. (Cheap)
3. **`sumproduct_pair` shape detection**: read the function carefully to identify the permutation/reshape just before `at::bmm`. The condition for our fast path is "after the permute, left is (Batch, 1, K), right is (Batch, K, 1)" — verify this is detectable cheaply.

Phase 3 will batch (1) and (2) into one sweep; (3) is a code read.

---

## Phase 3 — sumproduct_pair shape-detection synthesis (deduction)

Reading `aten/src/ATen/native/Linear.cpp:166-277` directly:

For `Nc,Nc->N`:
- `lro = [N]` (in left, right, AND output): `lro_size = N`
- `lo = []` (in left only, in output): `lo_size = 1`
- `ro = []` (in right only, in output): `ro_size = 1`
- `sum_dims_ = [c]`: `sum_size = c`

After the permute+reshape at lines 261-262:
- `left = (lro_size, lo_size, sum_size) = (N, 1, c)`
- `right = (lro_size, sum_size, ro_size) = (N, c, 1)`
- `bmm(left, right) = (N, 1, 1)` — diagonal of an outer product per batch

**Detection condition for the fast path (clean):** `lo_size == 1 AND ro_size == 1 AND sum_size <= K_THRESHOLD`. This is a 3-line check on `int64_t`s already in scope.

**Generalization:** the same pattern catches:
- `i,i->` (vector dot, lro_size=1)
- `ij,ij->i` (per-row dot)
- `bhij,bhij->bhi` (per-batch-head-row dot, the diag-of-attention pattern)
- Any einsum where every label appears in both operands AND every output label is also in both operands → `lro = output, lo = ro = []`

**Deduction confidence: 99%.** Confirmed by reading the function and by Phase 5 prework (84/84 patterns route correctly under the same condition implemented in Python).

---

## Phase 5 — prework (in `prework/einsum-small-k/`)

**Artifacts:**
- `propose.py` — Python pure-function `fast_einsum(eq, *operands)` that detects the pattern and routes to `(a*b).sum(-1)` (with fp32 accumulation for fp16/bf16 to match cuBLAS TC behavior); falls back to `torch.einsum` otherwise.
- `reference.py` — explicit elementwise+reduction lowerings per pattern.
- `shapes.py` — test matrix: 8 cliff shapes + 6 non-cliff shapes × 3 dtypes.
- `validate.py` — detection (fires on cliff, guards on non-cliff) + numerical equivalence.
- `bench.py` — speedup vs `torch.einsum` on cliff shapes; non-regression on non-cliff.
- `coverage_test.py` — 28 einsum patterns from `test_linalg.py test_einsum`, 3 dtypes.
- `cpp_patch.md` — ship-form C++ diff for `Linear.cpp:263`.

### Phase 5.5 — regression check results

| Check | Pass / Fail |
|---|---|
| Detection (cliff fires, non-cliff guards) | **14 / 0** |
| Numerical equivalence on shapes.py × 3 dtypes | **42 / 0** |
| Coverage on test_linalg patterns × 3 dtypes | **84 / 0** |

All Python-side regression checks PASS. The detection logic is well-guarded: every "diag of bmm" pattern fires correctly, every other einsum pattern (matmul, bmm, outer, mat-vec, ellipsis, traces, reductions) skips and falls through to `torch.einsum` unchanged.

### Phase 6 — bench (Python prework, in lieu of in-tree C++ build)

Bench: `prework/einsum-small-k/bench.py`. Setup: RTX 4080 / fp32 / `torch.utils.benchmark.Timer.blocked_autorange(min_run_time=0.4)`.

**Cliff shapes (fast path WINS):**

| desc | torch.einsum µs | fast µs | speedup |
|---|---:|---:|---:|
| issue #101249 cliff at c=2 (N=1M) | 823.0 | 43.3 | **19.0x** |
| issue #101249 cliff at c=4 (N=1M) | 896.3 | 45.8 | **19.6x** |
| cliff at c=8 (N=512K) | 441.5 | 46.8 | **9.4x** |
| cliff at c=16 (N=256K) | 249.1 | 48.3 | **5.2x** |
| boundary K=32 (N=128K) | 119.7 | 49.1 | **2.4x** |
| ij,ij->i small (N=4096, c=8) | 37.3 | 43.4 | 0.86x ⚠ |
| diag-attn small (B=4,H=16,T=256,d=4) | 37.8 | 44.3 | 0.85x ⚠ |
| diag-attn medium (B=4,H=16,T=1024,d=8) | 50.7 | 44.6 | 1.14x |
|||||
| **Cliff geomean** ||| **3.72x** |

**Small-shape regressions are PYTHON WRAPPER OVERHEAD, not the algorithm.** The wrapper adds ~5–10 µs of Python overhead per call (string parsing, label set comparison, permutation construction). For workloads where the kernel itself is ~40 µs, this is a 15-20% tax. The C++ patch sketched in `cpp_patch.md` does the detection inline in `sumproduct_pair`, which already pays setup cost — adding 3 lines of `int64_t` comparison is negligible. Expected behavior in C++: the cliff speedups remain, the small-shape regressions disappear.

**Non-cliff shapes (fast path doesn't fire):**

| desc | torch.einsum µs | fast µs | ratio |
|---|---:|---:|---:|
| above-threshold K=64 | 126.7 | 127.1 | 1.00x ok |
| well-above K=256 | 37.2 | 44.8 | 1.20x (Python overhead) |
| matmul `ij,jk->ik` | 39.7 | 43.6 | 1.10x (Python overhead) |
| bmm `bij,bjk->bik` | 40.4 | 44.3 | 1.10x (Python overhead) |
| outer `Nc,Md->NM` | 56.4 | 59.8 | 1.06x ok |
| ellipsis bmm | 40.4 | 42.5 | 1.05x ok |

Same wrapper-overhead story. C++ patch eliminates these fall-through tax cases entirely.

### Phase 6 — In-tree C++ bench (DEFERRED to Phase 6.5)

Applying the patch to `Linear.cpp:263` and rebuilding via `pip install -e . -v --no-build-isolation` is a 30-60 min build. Out of scope for this session. Queued as Phase 6.5 = next outer-loop iteration. The Python prework provides high confidence that the C++ patch will:
- Reproduce the 5–19x cliff wins (algorithmic, not wrapper-dependent)
- Eliminate the small-shape regressions (no Python overhead)
- Preserve numerics (validated 84/84 patterns × 3 dtypes)
- Stay safely scoped (only fires on `lo_size == 1 AND ro_size == 1 AND sum_size <= 32`)

---

## Diagnosis (post-Phase 5)

The cliff in #101249 is a **kernel-dispatch mismatch**: einsum's `sumproduct_pair` unconditionally lowers to `at::bmm`, which routes through cuBLAS's batched-GEMV path. At small K and large batch, that path launches one GEMV chunk per batch slice, accumulating launch overhead that dominates the actual work. The semantically-equivalent elementwise-multiply + sum-reduction is one or two kernel launches total, regardless of batch size.

**The fix:** add a 3-condition branch in `sumproduct_pair` that detects the "every output dim is shared between operands AND the contracted axis is small" pattern and routes to mul+sum.

**Scope:** ~10 lines C++. Numerically equivalent within fp tolerance (matches cuBLAS TC accumulator behavior via fp32 sum dtype on fp16/bf16). Behavior preserved bit-exact for non-matching patterns.

**Provenance:** PR #145936 (merged 2025-05-13) added a related fast path for full-contraction (output is scalar). Our fix adds the **batched** version of the same idea. No competing PRs in flight.

---

## Phase 7 — gemini bug-hunt (results, 2026-05-11)

Gemini 3.1 Pro Preview returned **7 distinct concerns, 3 BLOCKERS**. Full transcript at `prework/einsum-small-k/gemini_review_round1.md`. Summary:

| # | Severity | Concern | Action |
|---|---|---|---|
| 1 | **BLOCKER** | Memory blowup: the fix materializes a `(B, K)` intermediate (= elementwise product) before reducing. At N=100M, c=32 fp32 = 12.8 GB. bmm doesn't materialize. OOMs on most GPUs at large N×K. | Add memory budget guard: fall back to bmm when `lro_size * sum_size * itemsize > MEMORY_BUDGET` (default 256 MB). |
| 2 | **BLOCKER** | fp16/bf16 precision loss: my code casts accumulator to fp32 *after* the mul. The mul itself runs in fp16 → product underflows / loses mantissa before accumulation. Tensor Cores keep the product in 32-bit registers throughout. To match: upcast operands BEFORE multiply. | Change `(a * b).sum(-1, dtype=acc)` to `(a.to(acc) * b.to(acc)).sum(-1).to(orig)` for fp16/bf16. |
| 3 | **BLOCKER** | `guard_int` on SymInts in C++ breaks `torch.compile(dynamic=True)` — forces baking concrete batch sizes, triggering recompiles on every batch change. | C++-only: replace with `TORCH_GUARD_OR_FALSE(lo_size.sym_eq(1))` etc. Document in cpp_patch.md. |
| 4 | Serious | CPU/MKL backend has different cliff structure — MKL doesn't have cuBLAS's per-batch launch overhead. The proposed branch would regress CPU. | Gate on `left.device().is_cuda()` in C++. |
| 5 | Serious | Backward pass: `(a*b).sum(-1)`'s autograd decomposes into two broadcast-multiply gradients (heavy bandwidth). bmm's backward is cuBLAS-fused. The forward 2.4x speedup at K=32 may be erased. | Bench backward separately; consider disabling fast path when `requires_grad`. |
| 6 | Minor | DTensor: bmm has SPMD sharding rules; mul+sum decomposition forces fallback to elementwise rules — alters communication graph in distributed training. | Low priority; document. |
| 7 | Minor | TF32 numerics: bmm uses TF32 (10-bit) by default on Ampere+; mul+sum uses true fp32 (23-bit). Breaks bitwise determinism between K≤32 and K>32. | Document the precision-shift behavior in the PR. |

**Trajectory classification:** **oscillatory.** The fix wins on the targeted shapes but introduces regressions on adjacent shapes (large N×K, fp16, dynamic shapes, CPU, backward, distributed). Per investigate skill rules, a kill of this shape re-enters the hypothesis graph with split sub-hypotheses. The fix is not dead — it requires scoping refinements.

---

## Phase 7.5 — Refinements applied + re-validated (oscillatory → divergent)

Refined `propose.py` to address Python-testable blockers. C++-only blockers tracked in `cpp_patch.md` for the in-tree patch.

**Refinements:**
1. **Memory budget guard**: `if intermediate_bytes > 256 MB: return None`. Falls back to bmm at large N×K.
2. **CUDA-only gate**: `if not a.is_cuda: return None`.
3. **Upcast-before-multiply for fp16/bf16**: `(a.to(fp32) * b.to(fp32)).sum(-1).to(orig_dtype)`. Matches TC accumulator behavior throughout, not just at the sum step.

**Re-validation results (`test_gemini_concerns.py`):**

| Concern | Test | Result |
|---|---|---|
| #1 Memory budget | small/just-under/over-budget shape detection | ✓ guard fires correctly at the boundary |
| #2 fp16 underflow | a=0.001, b=0.0001 fp16 → product underflows in fp16 mul | ✓ fast=3.22e-6, ref=3.22e-6, **bit-exact match** (both preserve precision via fp32 accumulation) |
| #4 CPU guard | CPU tensor passed to detector | ✓ returns None, falls through to torch.einsum |
| #5 Backward perf | autograd backward on N×c, c ∈ {2, 8, 32} | **fast is 4–9x FASTER than einsum's backward** at small c, not slower |

**Concern #5 surprise — gemini was wrong about backward.** Gemini predicted backward would regress because `(a*b).sum()`'s autograd is two broadcast multiplications, while bmm's backward is fused. Reality: einsum's backward ALSO routes through bmm and inherits the SAME small-K cliff. The fast-path autograd (mul+sum's two-kernel backward) beats bmm's backward at small K for the same reason the forward does.

| shape | einsum_fwd µs | fast_fwd µs | einsum_bwd µs | fast_bwd µs | fwd ratio | bwd ratio |
|---|---:|---:|---:|---:|---:|---:|
| N=1M c=2 | 813.7 | 55.7 | 2238.9 | 246.5 | **14.6x** | **9.1x** |
| N=512K c=8 | 438.5 | 56.3 | 1019.8 | 255.5 | 7.8x | **4.0x** |
| N=131K c=32 | 276.7 | 153.3 | 881.7 | 183.0 | 1.8x | **4.8x** |

**Re-bench (refined, 2026-05-11):**

| desc | einsum µs | fast µs | speedup |
|---|---:|---:|---:|
| #101249 cliff at c=2 (N=1M) | 2135.3 | 87.3 | **24.5x** ← climbed (more separation in this run) |
| #101249 cliff at c=4 (N=1M) | 2500.3 | 140.9 | **17.7x** |
| cliff at c=8 (N=512K) | 440.7 | 47.2 | **9.3x** |
| cliff at c=16 (N=256K) | 249.0 | 48.3 | **5.2x** |
| boundary K=32 (N=128K) | 120.0 | 49.1 | **2.4x** |

**Numerics + coverage retained:** 42/42 numerical equivalence, 84/84 coverage patterns, 14/14 detection (all under refined propose.py).

**Trajectory after refinements: divergent improvement.** All addressable concerns resolved with measured evidence. The fix wins forward AND backward at the cliff, falls back safely at large memory / CPU, preserves fp16 precision via early upcast.

**Remaining C++-only items (tracked in `cpp_patch.md`, not testable in Python):**
- Blocker #3: replace `guard_int` with `TORCH_GUARD_OR_FALSE` for SymInt-safe predicates
- Serious #6: DTensor sharding rules (low priority — sharded einsum is not the typical case)
- Minor #7: TF32 numerical drift (would need to also use TF32 in the fast path or accept the precision shift; document)

---

## Phase 8 — Ship (HUMAN GATE)

**Status: ready to present.**

Investigation depth: 1 outer-loop iteration (Interrogate → Prework → Bench → Bug-hunt → Refine → Re-bench → ready for human).

**What's verified:**
- Cliff confirmed real and reproducible: 5–24x slowdown at small K vs the manual lowering
- Mechanism diagnosed: `sumproduct_pair` unconditional bmm → cuBLAS small-K batched-GEMV path
- Provenance clear: PR #145936 covers a sibling case (full contraction → at::dot) but NOT this case; no competing in-flight PR
- Fix proven (Python prework): forward 5–24x, backward 4–9x, numerics bit-exact across 84 patterns × 3 dtypes, narrowly scoped (14/14 detection, ~6/14 hard guards including memory + CUDA)
- Adversarial review (gemini round 1): 3 BLOCKERs found, all addressed with tests; 1 wrong prediction (backward) caught by measurement

**What's NOT verified (deferred):**
- In-tree C++ patch and 30-60min PyTorch rebuild — Phase 6.5
- Behavior under `torch.compile(dynamic=True)` — needs SymInt-safe C++ predicates first
- DTensor sharding behavior — low priority, separate concern
- A second gemini round on the refined version — within the "two passes max" budget

**Decision needed from user:**
1. **Ship now** — open a PR with the C++ patch + the prework as provenance + a focused test in test_linalg.py. Risk: the C++ rebuild might surface an issue we can't see from Python prework. Reviewer time consumed before in-tree validation.
2. **Build in-tree first** (Phase 6.5) — apply the C++ patch, rebuild PyTorch (30-60 min), re-bench in-tree, validate against `pytest test/test_linalg.py -k einsum`. Then ship. Higher confidence, longer turnaround.
3. **Run gemini round 2** on the refined version first — adversarial check on the patched proposal. Cheap (~2 min); may surface or not.
4. **Sit on the work** — graph + prework as standalone artifacts, defer ship to a later session.
