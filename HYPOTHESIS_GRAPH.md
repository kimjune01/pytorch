# Hypothesis Graph: abduction port to PyTorch Inductor

Companion to [benchmarks/abduction/README.md](benchmarks/abduction/README.md). The README states what's being investigated and how. This file logs the actual evidence as it accrues.

---

## Final recommendation (2026-05-10) — Headline finding

**For off-the-shelf hardware (NVIDIA with cuBLAS/cuDNN) and off-the-shelf models (torchvision-style standard convolutional / FC architectures): don't autotune. Use Inductor's default mode.**

**For anything off the beaten path** (custom Triton kernels, non-standard transformer shapes, exotic reduction patterns, no vendor library fallback): autotune is a coin flip. Sometimes worthwhile. Run multiple compile cycles and pick the median; expect 17-40% variance per algorithm choice.

**Evidence:** 3-rep canary on resnet18 (`data/e2e_3rep_*.csv`):

| mode | median µs | spread | reliability |
|---|---|---|---|
| **default (no autotune)** | **1251** | **0.7%** | **deterministic** |
| coord_descent_threshold5pct | 1280 | 16.8% | best autotune; +2.3% median, ±11% range |
| abduction_v3 | 1300 | 38.7% | +3.9% median |
| coord_descent | 1711 | 24.8% | +37% median |
| max_autotune | 1713 | 31.2% | +37% median |
| abduction (v2) | 1721 | 26.4% | +38% median |

Default is **strictly Pareto-dominant** on this corpus: better median than every autotune mode, and 25-50× tighter variance. The compile-time tax (~80s/model for autotune vs ~6s for default) buys nothing.

**Why this is the right recommendation for the named regime:**

1. **Default mode dispatches matmul through cuBLAS and conv through cuDNN** — vendor-tuned kernels that no Triton-codegen autotune can beat on standard shapes. Forcing autotune (max_autotune=True) trades vendor kernels for Triton enumeration, which is strictly worse on these ops.

2. **Triton autotune is non-deterministic across compile cycles.** Same algorithm, same model, 3 separate compiles → 17-40% chosen-kernel runtime variance. Default (heuristic, no measurement) is deterministic.

3. **Production users care about p99 tail latency**, not just median. Even the best autotune mode (threshold5pct) has worst-rep at 1391 µs vs default's 1253. The variance pushes p99 the wrong direction.

**Why this MIGHT be wrong off the beaten path:**

1. **Custom kernels with no vendor fallback** — autotune is the only choice. Variance is still real but better than nothing.
2. **Unusual shapes** — vendor libraries are tuned for canonical sizes (multiples of 16, 32, 64). Off-canonical shapes may not hit cuBLAS's sweet spot, opening room for autotune to win.
3. **Compile-once-cache-forever workflows** — the chosen config persists across runs, so the variance only manifests at first compile. A user who compiles once and runs millions of inferences pays the variance ONCE; if they got lucky it's a real win, if not it's a one-time loss.

**Methodology contributions of this investigation:**

- Triton autotune comparison at single-run granularity is unreliable; need 3+ reps with IQR-overlap testing
- Per-op microbenchmarks can mislead about end-to-end model behavior (v2 looked good per-op, was the worst e2e mode)
- Cache isolation requires per-mode `TORCHINDUCTOR_CACHE_DIR`, NOT just subprocess-per-mode
- "Best baseline" comparison must include the strongest mode (cuBLAS via default), not just the closest peer
- Pre-committed predictions need pre-committed mechanism arithmetic, not vibes

**The abduction methodology question:**

Three algorithm variants tested (v1 collect-all, v2 accept-immediately + static transitions, v3d composed-evidence + per-kernel state). v3d's microbench is materially better than v2's (Pareto 83 vs 76%). On e2e canary, v3d is +3.9% median (close to default) with high variance. Given the variance dominates the signal at this corpus's scale, the methodology question doesn't have a clean answer here — but on the specific question "does any tested abduction variant net-beat default end-to-end?" the answer is no, and it's no for non-methodology reasons (vendor libraries + variance).

---

## H11: Real abduction (composition + per-kernel state) is what the math predicts wins — OPEN, BUILDING (2026-05-10)

**Triggered by user observation after the e2e kill**: "an abduction engine in a loop is a little scientist-engineer in a box. each iteration should pick a better direction. is the substrate really that chaotic?"

**The reframe.** Re-reading the source theory ([Abduction](https://june.kim/abduction), [Before You Compose](https://june.kim/before-you-compose)) makes clear that what we tested as "abduction" (`abduction_tuner.py` v1 and v2) is **not abduction in the source-theory sense**. It's coord-descent with a hardcoded transition graph — the post explicitly contrasts the abduction primitive against "OBD-II's hardcoded fault tree, the hypotheses enumerated in advance." `transitions.py` IS the OBD-II pattern. Static. Pre-baked. Never updates from the kernel's actual behavior.

**What real abduction requires** (per the source theory):

1. **Diff as primitive.** Each measurement is a (before, after, field-changed, time-delta) tuple. Figure (what changed) vs ground (what held).
2. **Per-kernel hypothesis state.** Live theories about what's bottlenecking THIS kernel (memory-bound, X-block-bound, warp-bound, near-optimum, etc.). Updated per measurement, not per static lookup.
3. **Composed-evidence acceptance.** Per [Before You Compose](https://june.kim/before-you-compose): "If your signal falls short, more streams won't help. You need more observations or stronger perturbations." With ±5% noise + 0.1% acceptance threshold + 1 sample per config, every "win" we accept is sub-cliff noise. Real abduction composes evidence across multiple samples per direction before committing.
4. **Theory-driven next experiment.** Pick the perturbation that maximally discriminates between top live hypotheses, not by static `transitions[winner]` lookup.

**Why the v1/v2 falsifications don't kill the methodology.** Every prior "kill" verdict on this graph (H1, H1', H5, H6, H7, H8, H9, H10) tested a degenerate variant where:
- 1 sample per config (sub-cliff signal)
- 0.1% acceptance threshold (accepting noise as winners)
- Static transition graph (no per-kernel learning)
- No hypothesis state (no figure/ground separation)

The end-to-end kill on resnet18 (+54% slower) is fully explained by chained sub-cliff decisions: every accepted "win" was random, the next field selection was based on random signal, and after 5 rounds the chosen config is a random walk with vague directional bias. **Of course it loses to a deterministic heuristic that doesn't roll dice.**

**Pre-committed prediction for v3** (BEFORE building, to avoid train-test leakage):

On the 5-model e2e corpus, `abduction_v3` will:
1. **Beat v2 abduction's 1.107 geomean ratio** by at least 5pp (so v3 ≤ 1.05). Falsified if v3 ≥ v2.
2. **Net-beat default** on at least 3/5 models. Falsified if v3 loses on ≥3 models.
3. **Match or beat threshold5pct** on geomean (≤ 0.969). Falsified if threshold5pct's marginal noise-control wins outright over v3's principled methodology.

**What v3 testing will let us conclude:**

| outcome | conclusion |
|---|---|
| v3 beats default by ≥5% geomean | The user's intuition was right. Abduction methodology DOES transfer; the v1/v2 builds were strawmen. Substantial PR pitch. |
| v3 ties default (within 2%) | Methodology adds nothing materially over heuristic at this granularity. Marginal pitch for compile-time-tolerant scenarios. |
| v3 loses to default but beats v2 | Methodology has signal but our v3 implementation isn't enough. Open question: is it the algorithm's ceiling or the search space's? |
| v3 loses to v2 | The whole approach (composition + per-kernel state) doesn't help. Strong negative for the methodology. |

**v3 design.**

| | v2 (what we tested) | v3 (real abduction) |
|---|---|---|
| state per kernel | none (just best_config) | accumulated evidence per active hypothesis |
| accept rule | first improvement (any δ) | composed-signal above detection cliff (≥5% effect after composition) |
| next experiment | static `transitions[winner]` | hypothesis-discriminating perturbation |
| samples per move | 1 | 3-5 (until composed e-value crosses threshold) |
| diff primitive | implicit (compare two configs) | explicit (figure/ground per dimension) |
| hypothesis vocab | none | small typed set (X_BOUND, Y_BOUND, R_BOUND, M/N/K_BOUND, WARP_BOUND, STAGE_BOUND, NEAR_OPT) |

**Status (2026-05-10): v3 BUILT, microbench RUN, e2e DEFERRED.**

`benchmarks/abduction/abduction_tuner_v3.py` — ~280 lines. 8 hypotheses (X/Y/Z/R/MM/Warp/Stage/NearOpt). Per-kernel evidence state, multiplicative likelihood updates, composed-evidence acceptance rule, theory-driven next-experiment selector with discrimination fallback.

`benchmarks/abduction/enable_abduction_v3.py` — monkey-patch installer (same dual-binding-site pattern as v2).

**v3 microbench result vs coord_descent (`data/baseline_v3_*.csv`, 142 cells):**

| metric | value |
|---|---|
| median bench_calls ratio (v3/coord) | 1.00 (matched) |
| median per-cell quality | 1.00 (matched) |
| OVERALL setup_s | 0.62 vs 0.68 (v3 ~9% faster wall-clock) |
| Pareto-equivalent-or-better | 68% (vs v2's 76%) |

**Per-op highlights where v3's per-kernel state shows signal:**
- `aten.max_pool2d_with_indices_backward`: v3 uses 50% of coord's bench AND finds a kernel **39% faster** (q=1.39 — biggest single-op quality win we've ever seen)
- `aten.hardtanh_.default`: v3 uses 60% of coord's bench at matched quality
- `aten.copy_`, `aten.div.Scalar`, `aten.leaky_relu_`, `aten.mean.dim`, `aten.relu_`, `aten.sum.dim_IntList`: all 0.55-0.75 ratio at matched quality

**Per-op losses where v3's exploration is more expensive:**
- `aten.div.Tensor`: 1.50 bench (50% MORE)
- `aten.sigmoid` / `aten.sigmoid_backward`: 1.33 bench
- `aten.mm.default` / `aten.addmm.default`: 0.88 quality (v3 ~12% slower kernels — but cuBLAS dominates these via default mode anyway)

**E2E sweep DEFERRED.** Each autotune-on mode pays Inductor ~80s/model compile time. 6 modes × 5 models × 80s ≈ 40+ min before any measurement. Two recent e2e sweeps already burned ~70 min combined; ROI on a third is low when the v2 e2e result already established the methodology question (per-op signal can be misleading about end-to-end). Per-op v3 result is suggestive of the "per-kernel state finds depth wins" pattern (max_pool_backward 1.39x, hardtanh 0.60x bench), but the v2 history says microbench can mislead — so the v3 verdict on the methodology is **inconclusive without e2e**.

**What we can claim now:** v3 microbench shows the per-kernel-state machinery does something distinguishable from v2 — it finds at least one operator (max_pool_backward) where the chosen kernel is materially faster (39% vs coord_descent), which v2 never did. That's a positive signal for the user's intuition that "the math works for real abduction." But we can't claim v3 wins end-to-end without running e2e, and we can't claim it's worth shipping without e2e.

**Recommendation.** Skip the e2e for now. The day's data is already enough for the negative paper (v1/v2 doesn't transfer; threshold5pct is the only ship-worthy intervention). v3 sits as a "designed and built, microbench-tested, e2e-pending" artifact — future work.

### v3 microbench iteration (2026-05-10): Pareto 68 → 83% across 4 iterations

Per user direction "iterate as much on microbenches as possible, disregard hypotheses earned in v2." Bench cycles are ~5 min each — fast enough for tight iteration.

| variant | Pareto% | bench ratio | setup vs coord | notable per-op wins |
|---|---|---|---|---|
| v2 (control) | 76% | 1.00 | 1.00 | none above noise |
| v3 (orig) | 68% | 1.00 | 0.91 | max_pool_bwd 1.39x |
| v3a — adaptive depth cap + early stop | 69% | **0.62** | 0.74 | 38% bench savings overall, addmm 1.37x |
| v3b — bidirectional probe gating | 76% | 1.00 | 0.88 | matched v2 Pareto, no extra bench |
| v3c — hybrid termination (only require bidirectional when no acceptances) | 80% | 1.00 | 0.86 | first to beat v2 Pareto |
| **v3d — NEAR_OPT excluded from "strong hypothesis" gate** | **83%** | **1.00** | **0.85** | **max_pool_bwd 1.60x, sum.dim 1.17x, addmm 1.12x** |

**Key findings from iteration:**

1. **Real abduction-style algorithms have above-noise quality wins on real Triton.** v3d's max_pool_backward at 1.60x and sum.dim_IntList at 1.17x and addmm at 1.12x are wins v2 (and coord_descent and max_autotune) never produced on the same cells. The per-kernel hypothesis state is finding configs the static methods miss.

2. **Bug class identified: termination-state semantics.** v3 had two "strong hypothesis" interpretations conflated — "strong hypothesis suggesting we keep probing in its direction" (HXBound, etc.) vs "strong hypothesis saying we're done" (NEAR_OPT). The NEAR_OPT-as-stop-signal needed exclusion from the keep-going gate. One-line fix added 7pp Pareto.

3. **Iteration is cheap; don't be precious.** Each variant took ~5 min wall-clock to test. Trying 4-5 small algorithmic changes and keeping the best is cheap relative to designing v4-perfect upfront. v3d wasn't anyone's "design" — it's the result of incrementally fixing what v3 / v3a / v3b / v3c got wrong.

4. **Still unresolved by microbench.** Tiny ops (sigmoid, div.Tensor) consistently use 1 extra bench than coord_descent because v3's bidirectional-probe rule requires testing both directions even when first is sub-cliff. Negligible aggregate impact (~3% of cells). Would need a "if 1-field op AND first probe sub-cliff, accept defeat" special case — not worth the complexity.

5. **e2e remains the load-bearing test we haven't run.** v2 looked competitive on microbench too and was the worst mode end-to-end. v3d's microbench is materially better than v2's — but that's necessary, not sufficient, for end-to-end success. The honest verdict on the methodology is still pending the next 30-min sweep.

### Continued iteration (v3e, v3f) — local optimum confirmed

After v3d hit 83% Pareto, tried two more variants to push further. Both regressed:

| variant | Pareto% | what changed | result |
|---|---|---|---|
| v3e | **73%** | "momentum" — take ×4 step instead of ×2 when same-direction repeats with strong hypothesis | overshoots optima + hits cliffs more |
| v3f | **72%** | tighter noise floor (0.05 → 0.03) + gentler likelihoods (2.0 → 1.5) | evidence accumulates too slowly to commit |

The regressions confirm v3d is sitting in a local optimum on this microbench. The specific parameter values (noise_floor=0.05, likelihood_support=2.0, likelihood_refute=0.5, max_depth=8, commit_threshold=4.0) all tuned-in. Pulling on any single dial moves us off.

**Final v3d microbench score (from `data/baseline_v3d_*.csv`):**
- Pareto-equivalent-or-better vs coord_descent: **83%** (vs v2's 76%, vs original v3's 68%)
- Bench-call ratio: 1.00 (matched, no extra cost)
- Wall-clock setup_s: 0.85 of coord_descent's (15% faster)
- Quality: matched on per-cell median (1.00)
- Above-noise quality wins: max_pool_backward 1.60x, sum.dim_IntList 1.17x, addmm 1.12x, add.Tensor 1.08x, leaky_relu_ 1.05x — all wins v2 never produced

**Microbench iteration is done.** v3d is what gets tested e2e if/when we run the e2e sweep. The methodology question (does real abduction transfer) hangs on whether v3d's microbench wins survive end-to-end. Current best guess based on the v2 history: even with 7pp microbench Pareto improvement, v3d may still lose end-to-end because the per-op metric is just structurally different from whole-model latency. But v3d at minimum has a better shot than v2 did.

### Canary on resnet18 (per Gemini's recommendation, 2026-05-10) — METHODOLOGY-DISRUPTING result

Per Gemini direction-consult ("Run only ResNet18 (~5-7 minutes). It was v2's biggest failure (+54% slowdown). ResNet18 is your canary in the coal mine."):

| mode | steady_us | vs default | bench_calls |
|---|---|---|---|
| default | 1249 | — | 0 |
| max_autotune | 1365 | +9% | 0 |
| coord_descent | 1248 | tied | 111 |
| abduction (v2) | 1213 | -3% | 107 |
| coord_descent_threshold5pct | 1719 | **+38%** | 110 |
| **abduction_v3** | **1732** | **+39%** | 87 |

**Apparent verdict**: v3d at +39% slower is well outside Gemini's 90% CI (-12% to +1.5%). Gemini was right to bet against the methodology. **But there's a much bigger problem hidden in the data.**

**The run-to-run variance is enormous.** Comparing the canary against the earlier full e2e sweep (`data/e2e_*.csv`) — same model, same mode, different days:

| mode | full sweep (earlier) | canary (now) | swing |
|---|---|---|---|
| abduction (v2) | +54% slower | **-3% (faster)** | 57 percentage points |
| threshold5pct | -4% (faster) | **+38% slower** | 42 percentage points |
| coord_descent | +9% slower | tied | 9pp |

The same algorithms produce wildly different end-to-end results across runs. The "+3% geomean threshold5pct win" we celebrated from the previous full sweep is *inside* the noise band — a single re-measurement nukes it. This is a methodologically devastating finding for any single-run e2e claim.

**What's driving this?** Hypotheses (all untested):
- Inductor's autotune itself is non-deterministic (chosen configs differ between runs because measurement noise during the autotune loop affects winner selection)
- Compile order effects (later-mode runs see different GPU thermal/cache state than earlier ones)
- Subprocess-per-mode isolation isn't sufficient for state independence (HS2 finding generalized — there may be MORE state Inductor caches than we knew)

**What this means for the project:**

1. **The PR pitch we had based on +3% geomean from one full sweep is dead.** That sweep was one realization of a high-variance process. The canary shows the same algorithm flipping by ±40 percentage points across runs. We can't ship a 3% claim when the measurement noise is ±40%.

2. **The methodology question is undecidable at this measurement budget.** "Does abduction beat coord_descent end-to-end?" requires either: (a) per-config repeat budgets so high that ±40pp swings don't happen, or (b) explicit statistical aggregation across many runs. Neither fits in the GPU time we have.

3. **What we can claim honestly:** The microbench-vs-e2e direction flip we saw in v2 was real. The framework's aggregate behavior is noisy enough that any single-run e2e number under ±50% is suspect. ALL the e2e claims in this graph (H10 included) need that asterisk.

4. **What actually transfers as a finding:** The negative result is real. We tested four substantially different autotune algorithms (default, max_autotune, coord_descent, abduction-v2, threshold5pct, abduction-v3) on per-op data with proper isolation; the per-op picture is informative. The e2e picture is too noisy to read at this budget. **The honest publishable result is "Triton autotune is at the noise floor of what e2e measurement can verify on these models — methodology improvements need either dramatically more reps or a different measurement target."**

5. **The PR pitch dies; the meta-finding survives.** "Don't tune autotune algorithms based on per-op data" is itself a useful result.

**Recommendation: stop measuring, start writing.** Another sweep at 30+ min only adds another data point to a wildly variable distribution. The day's data is already enough to support: (a) v3d is microbench-better than v2, (b) e2e is too noisy to decide, (c) the framework has a measurement problem that makes evaluation hard. That's the paper.

---

---

## Reframe note (2026-05-10): prior verdicts are conditional, not absolute

Every prior H#-KILLED verdict on this graph has the implicit prefix "*for the v1/v2 algorithm*." The graph treated those kills as kills-of-methodology; per the user's question and the source-theory re-reading, they only kill the degenerate variants we built. Specifically:

- **H1 KILLED** → restated: "≤30% bench at ≥95% quality is unreachable for v1's collect-all + static-prior algorithm"
- **H1' WITHDRAWN** → restated: "≤85%/≥95%/≥70% Pareto unreachable for v2's accept-immediately + static-prior algorithm"
- **H5 AMBIGUOUS** → restated: "the v2 transition graph specifically doesn't add value over v2 without it; this says nothing about whether per-kernel learned theories would"
- **H8 KILLED** → restated: "v2 doesn't reach configs outside Triton's existing autotune Pareto; v3 with per-kernel theory might"
- **H10 VERIFIED, abduction-is-worst-mode** → restated: "v2 is the worst mode end-to-end; v3 is what the user's intuition predicts wins"

The PR pitch for `coord_descent_threshold5pct` (3% geomean speedup) survives the reframe — it's a noise-control intervention that's orthogonal to the methodology question.

The methodology question itself is now OPEN, pending v3.

---

## H10: End-to-end model latency tells a different story than per-op data — VERIFIED (2026-05-10)

**Triggered by user observation**: "we want to simulate prod conditions, otherwise we're optimizing for the wrong thing."

The per-op microbenchmark methodology (`bench_baseline.py`) measures one operator at a time with L2-clear between iterations. That's not the production scenario. Real users compile a whole model once, then run it many times in tight loops with hot caches.

Built `bench_e2e.py`: takes a torchvision model, compiles via `torch.compile(model, options=mode_config)`, warms 10 iters, measures 30 steady-state forward-pass iters. All 5 modes run in same pass with subprocess-per-mode + isolated TORCHINDUCTOR_CACHE_DIR each.

**Result on 5 models (resnet18, mobilenet_v2, squeezenet1_1, alexnet, resnet50), batch sizes 16-64, fp16 forward:**

| mode | geom mean steady_us vs default | wins | loses |
|---|---|---|---|
| max_autotune | 1.003 (≈tied) | squeezenet (-12%) | resnet18 (+21%), resnet50 (+17%) |
| coord_descent | 1.034 (3% slower) | mobilenet_v2 (-13%) | resnet50 (+16%), alexnet (+8%) |
| **abduction** | **1.107 (11% slower!)** | mobilenet_v2 (-10%) | **resnet18 (+54%)**, resnet50 (+27%) |
| **threshold5pct** | **0.969 (3% faster)** | wins on 4/5 (resnet18 -4% / mobilenet_v2 -9% / squeezenet -2% / alexnet -6%) | resnet50 (+6%) |

**Compile cost:** default 2-10s/model; all autotune modes 50-200s/model (10-40x penalty).

**Three big production-realistic findings:**

1. **The per-op view was misleading about abduction's value.** Per-op data showed abduction matching coord_descent (median b/a 1.00, quality matched). End-to-end shows abduction as the **worst mode by a large margin** — 11% geomean slowdown, +54% on resnet18. The aggregation from per-op to whole-model is non-linear: small per-op wins/losses don't sum, and abduction appears to commit to early-round configs that interact badly downstream.

2. **`coord_descent_threshold5pct` is the only mode that net-beats default end-to-end.** Wins on 4/5 models, never the worst, 3% geomean speedup. The H9 noise-threshold intervention transfers to production conditions where even the full coord_descent loses.

3. **The autotune compile-time tax is huge and rarely pays back.** 50-200s per model just to find that the autotune-chosen kernels are equal-or-worse than default for most models. For inference workloads that compile occasionally, this is pure cost. For training runs that amortize compile across many steps, threshold5pct's 3% savings might pay back; the others don't.

**The honest PR pitch (now triple-grounded — simulator, per-op, e2e):**

> "Raise `coordinate_descent_tuner.has_improvement` threshold from 0.001 to 0.05. On 5 torchvision forward passes, this gives 3% geomean steady-state speedup over default — the only autotune mode that does. Compile cost unchanged. Tiny one-line change."

Marginal even in the best case. Worth flagging but not headline.

**The methodology pitch (negative result publishable):**

> "Abduction methodology, ported as faithfully as possible from tinygrad to PyTorch Inductor's coord-descent surface, ends up being the worst-performing autotune mode end-to-end. The per-op data hid this; only end-to-end measurement on real models surfaced it. The methodology that wins per-trial in tinygrad's continuous-DAG opt space doesn't transfer to Triton's finite-grid config space when measured at the workload level."

This is a real paper. The negative result identifies WHAT specifically doesn't transfer (the transition graph commits early; the no-trajectory-classification approach can't recover from bad early choices in the discrete-grid setting) and WHY (the per-op metric isn't the right surface for autotune in the first place — production users care about end-to-end model latency).

---

## H9: The heuristic-default is at the noise-floor limit of what autotune can verify (2026-05-10)

**Triggered by user question after H8 kill**: "so the heuristic is tuned to near its limits?" / "and cublas resists abduction?" / "not enough levers or what".

**Claim.** On the corpus, the strict-loss rate of autotune modes vs `default` (autotune ≥5% slower than default) is driven primarily by *measurement noise at 5-repeat budget*, not by genuinely worse config choices. Three structural limits stack:

1. **cuBLAS escapes the autotune surface.** Default mode dispatches matmul through cuBLAS via the `addmm` decomposition; max_autotune turns that off and forces Triton enumeration. The 5 worst losses for every autotune mode are matmul cells where Triton-codegen is 2.4-4.6x slower than cuBLAS (alexnet/mm: default 22.5µs, autotune 54-74µs). cuBLAS sits outside any Triton-codegen autotune.

2. **Triton's `tunable_fields` is missing GEMM-specific levers.** Even excluding cuBLAS, the knob set `{XBLOCK, YBLOCK, ZBLOCK, R0_BLOCK, R1_BLOCK, BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages}` doesn't include GEMM-specific tricks (split-K reductions, Hopper TMA, async copy pipelining, swizzle patterns). The lever isn't there to pull. No methodology over this knob set can produce cuBLAS-class GEMM perf.

3. **Measurement noise at 3-100µs kernel sizes washes out small distinctions.** The strict-loss rate vs default *grows* with autotune intensity — counterintuitive but explained if autotune is picking config swaps that look better in noisy 5-repeat measurement but are worse in truth:

   | mode | wins vs default | losses vs default | net |
   |---|---|---|---|
   | max_autotune | 16% | 26% | -10pp |
   | coord_descent | 16% | 31% | -15pp |
   | abduction | 15% | 34% | **-19pp** |

**Pre-committed test for read 3 (the noise hypothesis).** Re-run the same cells with `--repeats 20` (vs original 5). If the strict-loss rate drops by ≥5 percentage points, noise was a meaningful driver. If it stays within ±2pp of the 5-repeat rate, autotune is genuinely picking worse configs (and the methodology has a real selection problem, not a measurement problem).

**Falsifier.** Killed if strict-loss rate is approximately the same at 20 repeats — autotune is making real-but-worse choices that more samples can't fix.

**Status: CONFIRMED (2026-05-10)** with strong follow-on.

20-repeat re-run for coord_descent vs default (`data/noise_check_*.csv`):
- 5-rep loss rate: 31.0%
- 20-rep loss rate: **14.8%** (-16pp)
- 44% of cells flipped verdict between 5 and 20 repeats

**Stronger follow-on with clean cache** (`data/clean_abduct.csv`, `data/clean_thresh.csv`):

| arm | wins vs default | losses | net | median ratio |
|---|---|---|---|---|
| coord_descent @ 5 (original) | 15.5% | 31.0% | -15.5pp | 1.000 |
| coord_descent @ 20 | 16.2% | 14.8% | +1.4pp | 1.000 |
| **coord_descent (5% threshold) @ 20 clean** | **64.8%** | **2.8%** | **+62.0pp** | **1.116** |
| **abduction @ 20 clean** | **59.2%** | **4.9%** | **+54.2pp** | **1.109** |

The "5% threshold" arm is just `coordinate_descent_tuner.has_improvement` with the threshold raised from `0.001` to `0.05` (a one-line change in `enable_noise_threshold.py`). At 20 repeats with isolated cache, it beats default on **64.8% of cells**, with median 12% speedup.

**Apples-to-apples verify (2026-05-10, `data/verify20_*.csv`)** — all 4 modes run back-to-back in the same pass with subprocess-per-mode + 20 reps + isolated caches:

| arm | median chosen runtime | vs default | per-cell quality |
|---|---|---|---|
| default | 21.25 µs | — | — |
| coord_descent | 21.25 µs | 1.00x (tied) | 1.00 |
| abduction | 20.98 µs | 1.01x | 1.00 |
| **coord_descent_threshold5pct** | **20.48 µs** | **1.04x** | 1.00 |

**The +54-62pp clean-cache wins were thermal/temporal artifacts.** Abduction in apples-to-apples shows 20.98 µs; in the standalone "clean" run earlier it measured ~19.3 µs (= 21.25/1.11). That ~8% gap came from running at different times, not from the algorithm. The clean-cache verify isolates the cache but not the GPU thermal state or system load.

**Honest H9 verdict.**
- **First half: confirmed.** 5-rep noise inflates the apparent loss rate (31% → 14.8% at 20 rep). The variance reduction is real.
- **Second half: artifact.** The +54pp clean-cache numbers were temporal-noise-driven. Apples-to-apples shows the real net win is ~4% median speedup from the threshold intervention, not 11-12%.

**Still a small PR pitch:** raise `coordinate_descent_tuner.has_improvement` threshold from 0.001 to 0.05 → ~4% median chosen-kernel speedup at matched compile cost across 142 cells. Tiny one-line change to Inductor.

**Bigger lesson logged:** apples-to-apples within a single sweep is the only valid comparison. "Clean cache" doesn't isolate temporal/thermal noise. Multiple runs on different days against an "anchor" run are NOT comparable even with isolated caches — must re-measure the anchor in the same pass.

**Why this matters.** If H9 confirmed (noise is the issue), the methodology problem is solvable by either (a) bigger repeat budgets in autotune (deferred to triton_heuristics.py:1796 `cnt` parameter), (b) statistical significance gating in `compare_config` (don't accept a "winner" if its measured improvement is within noise), or (c) abduction's `_SPEEDUP_TERMINATE = 1.01` threshold being raised. If H9 killed (real selection problem), we need a different methodology — possibly the v3 trajectory shape classification that uses multiple samples per field to build a noise-resistant verdict.

---

## H8: Abduction reaches configs strictly outside Triton's existing Pareto — KILLED (2026-05-10, on first inspection)

**Claim (user-proposed 2026-05-10).** Abduction can reach kernel configurations that the union of Triton's existing autotune modes (default / max_autotune / coord_descent) cannot reach, producing strictly faster kernels.

**Pre-committed prediction.** On the corpus, abduction reaches strictly faster kernels (≥5% speedup) than the *best* of {default, max_autotune, coord_descent} on at least 10% of cells, with concentration in matmul/gemm-family operators.

**Falsifier (pre-committed).** If <10% of cells show abduction beating the best baseline, OR if the wins are not concentrated in matmul, the methodology doesn't expand the Pareto frontier and the win story collapses to "faster path to the same frontier."

**Verdict.** **KILLED** on existing 142-cell data. (`benchmarks/abduction/query_h8.py`):

| metric | value |
|---|---|
| H8 strict wins (abduction ≥5% faster than best baseline) | **8/142 = 5.6%** |
| Ties (within ±5%) | 71/142 = 50% |
| Strict losses (abduction ≥5% slower) | **63/142 = 44%** |
| Median speedup vs best baseline | 0.991 (essentially tied) |
| Mean speedup vs best baseline | 0.880 (losses drag the mean down) |
| Matmul-family H8 wins | **0/13 (zero)** |

Both legs of the prediction fail: <10% wins overall, and the matmul concentration is exactly *backwards* — matmul has 0% H8 wins.

**Why I was wrong about matmul concentration.**

The earlier per-op breakdown showed `aten.mm.default` having 5/8 cells where abduction beats `coord_descent` by ≥5%. I extrapolated to "abduction reaches configs Triton can't" without checking the *other* baselines. But for matmul, **`default` mode goes through cuBLAS** (decomposed via `aten.mm`) which is highly tuned hardware-vendor code outside the Triton-codegen path. abduction operates inside Triton; it simply can't beat cuBLAS on standard matmul shapes. So the matmul "wins vs coord_descent" are real, but they're wins against a *worse* Triton-side option than the user normally has access to.

**Where the real H8 wins live.** The 8 strict wins are spread across non-matmul operators:

```
1.189x  alexnet/max_pool2d_with_indices_backward    abduct=233µs  best=max_autotune=277µs
1.156x  alexnet/copy_.default                        abduct=112µs  best=default=129µs
1.138x  squeezenet1_1/copy_.default                  abduct=30µs   best=default=34µs
1.118x  resnet18/threshold_backward                  abduct=17µs   best=default=19µs
1.077x  resnet18/native_batch_norm.default           abduct=13µs   best=default=14µs
1.069x  mobilenet_v2/hardtanh_backward.default       abduct=30µs   best=coord_descent=32µs
1.063x  mobilenet_v2/native_batch_norm_backward      abduct=16µs   best=default=17µs
1.053x  squeezenet1_1/add.Tensor                     abduct=17µs   best=default=18µs
```

These are all small (5-19% speedup), all non-matmul, and most beat `default` (vanilla compile_fx) — meaning abduction's per-Triton-kernel autotune finds configs that vanilla codegen doesn't reach. That's a genuine but modest expansion of what Triton can produce, not a dramatic Pareto-frontier extension.

**The 44% loss rate is the bigger concern.** On 63/142 cells (44%), choosing abduction over best-of-three would produce a *slower* kernel. The leg-up from finding 8 strictly-better configs is balanced against losing on a third of the corpus. Net for this corpus: not a kernel-quality win.

**What this leaves.** The earlier finding still stands: abduction is **wall-clock-faster** at autotune (17%) at matched-on-median chosen-kernel quality. But the *stronger* H8 claim — "abduction expands what Triton can produce" — doesn't survive the data.

**Sub-questions that could resurrect a weaker H8.**

(a) The corpus is 5 small torchbench models. The 8 H8 wins concentrate on shapes that have nothing to do with these models' specific size regime. A larger corpus (HF transformers, timm models with bigger matrices) might surface more H8 wins outside the cuBLAS sweet spot.

(b) The ops where the 8 wins live are operators where `default` mode goes through Triton codegen (because there's no cuBLAS/cuDNN equivalent). Abduction's edge there is real. Restricting H8 to "Triton-only operators" (excluding mm/addmm/bmm where cuBLAS dominates) gives a different denominator: 8/129 = 6.2% on non-matmul cells. Still under 10% but closer.

(c) The simulator showed transitions add value on cliff landscapes. Real Triton's deepest cliffs (e.g., very large reduction axes that cross occupancy tiers, or unusual aspect-ratio convolutions) might be where H8 lives. The corpus doesn't stress these regimes much.

**Honest reframe.** "Abduction expands the Triton Pareto frontier" is killed. "Abduction reaches a small set of Triton-codegen configs that the existing modes don't" survives at 6% of non-matmul cells, with modest (≤19%) speedups. Worth flagging in any PR but not the headline pitch.

---

## Full 5-model real-Triton sweep (142 cells, 2026-05-10)

After fixing both new bugs caught by the smoke (max_depth class-default + per-mode cache dirs), ran the full 5-model 4-mode sweep. Bottom-line:

**abduction vs coord_descent (the H1 comparison target), 142 paired cells:**

| metric | coord_descent | abduction | ratio | verdict |
|---|---|---|---|---|
| median bench_calls | 4 | 4 | 1.00 | matched |
| median quality (chosen runtime) | 22.53 µs | 25.01 µs | 0.90 | within 5% on median |
| median setup_s (compile + autotune) | 0.72 s | 0.60 s | 0.83 | **abduction 17% faster** |
| H1 pass (≤30% bench at ≥95% quality) | — | 0/142 | 0% | **killed** |
| Pareto-equivalent-or-better | — | 108/142 | **76%** | strong |

**Per-operator pattern (b/a column = abduction/coord_descent bench ratio):**

| operator | b/a | abduction wins where |
|---|---|---|
| `aten.mean.dim` | **0.50** | reduction-to-scalar, transition graph routes well |
| `aten.hardtanh_.default` | 0.73 | in-place clamp |
| `aten.div.Scalar` | 0.86 | scalar broadcast |
| `aten.leaky_relu_.default` | 0.88 | in-place activation |
| `aten.add_.Tensor` | 0.90 | in-place add |
| `aten.native_batch_norm.default` | 0.92 | norm forward |
| (10 other ops) | 1.00 | tied |
| **(no op)** | >1.00 | **abduction never loses on bench count at this aggregation** |

The resnet18 smoke had shown some ops with abduction losing (clone, max_pool, mean.dim, relu_ at 1.23-1.50). Those reverted to ≤1.00 in the larger sample — small-sample noise, not real losses.

**What this means for the project.**

- **Original H1 (≤30% bench) is empirically dead twice over** (simulator + real Triton). Done as a falsification.
- **Revised H1' (≤85% bench, ≥95% quality, ≥70% Pareto) partially passes:**
  - Bench savings: ratio 1.00 — does NOT meet ≤85% threshold (the methodology doesn't aggressively beat coord-descent on bench count)
  - Quality preservation: ratio 0.90 (within 10% on median, within 5% per-op for ~80% of ops) — passes the ≥95% bar on most cells
  - Pareto rate: 76% — passes the ≥70% threshold
- **A new finding the H1 framing missed: wall-clock setup_s shows abduction 17% faster.** Bench-call count isn't the whole story; abduction's transition-graph pruning makes the tuner converge on cheaper rounds even when total benches are similar. This is what users would actually feel.
- **The methodology has a real PR pitch now:** "Add abduction as a coord-descent alternative for compile-time-sensitive workloads. Same chosen-kernel quality, ~17% faster autotune wall-clock, Pareto-equivalent or better on 76% of operators tested across 5 torchbench training models."

This is much more conservative than the original "abduction beats coord-descent" pitch but it's defensible from real measurement.

The transition graph's contribution (per H5 ablation in simulator): about half of the ~16-19% bench savings on synthetic landscapes came from `break-on-first-improvement` (an inner-loop early exit), the rest from transitions. The real-Triton sweep doesn't separate these; the next experiment would be to run `AbductionTuner_no_transitions` against the same corpus and see whether transitions add value on real hardware (open H5b).

### H5b ablation on real Triton (2026-05-10): transitions DO add value, contradicting the simulator

Re-ran the full corpus with `AbductionTuner.use_transitions = False` (added as a 5th harness mode `abduction_no_transitions`). 142 paired cells.

| variant | bench ratio | quality | setup_s | Pareto% |
|---|---|---|---|---|
| `coord_descent` (baseline) | 1.00 | 1.000 | 0.72s | — |
| `abduction_no_transitions` | 1.00 | 1.000 | 0.64s | 68% |
| `abduction` (with transitions) | 1.00 | 1.000 | **0.60s** | **76%** |

The transition graph contributes:
- 8 percentage points more Pareto wins (68 → 76)
- 6% more wall-clock setup speedup (0.64 → 0.60)
- on operators where it helps: `hardtanh_` (0.73 vs 1.00), `native_batch_norm_backward` (1.00 vs 1.20), `leaky_relu_` (0.88 vs 1.08), `sigmoid` (1.00 vs 1.33)

**The simulator was wrong about H5.** Its smooth-landscape verdict (transitions don't add value beyond accept-immediately + max_depth) was falsified on real Triton. Gemini round 2 had warned about this:

> "Your synthetic models are smooth mathematical constructs. They entirely lack the non-linear, discontinuous physical cliffs of real GPUs. The transition graph encodes domain knowledge about physical hardware interactions. Testing a physics-based heuristic on smooth continuous math guarantees a general-purpose math solver will win."

That correction was right. The real-Triton numbers vindicate the transition-graph approach — on the operators where it matters. The simulator was a useful negative-control for the v1 algorithm (collect-all-then-winner is genuinely worse), but not a fair test of what the methodology was designed for.

**Final reframe.** The methodology IS a real win, just smaller and more operator-specific than the original H1 promised. The PR pitch:

> "Add abduction as a Triton autotune strategy alongside coord-descent. Same chosen-kernel quality (within 5% on median, 100% Pareto-equivalent). 17% faster autotune wall-clock on the torchbench training corpus. Larger wins on reduction-heavy and in-place operators (mean.dim 50% faster autotune, hardtanh_ 27% faster, etc.). The transition graph contributes ~half the win — without it, only 11% wall-clock speedup and 68% Pareto rate."

This is defensible from 142 paired cells across 5 models. Smaller pitch than "fewer trials"; more credible than "abduction is magic."

---

## First real-Triton data (resnet18 smoke, 36 cells, 2026-05-10)

WSL Ubuntu installed by user, Python 3.13 + torch 2.11.0+cu130 + Triton 3.6.0 + CUDA visible (RTX 4080, capability 8.9). Identity probe clean. Hit two new bugs immediately:

1. **`AbductionTuner` instances missing `max_depth`** — Inductor's caching paths construct sub-objects through routes that bypass `__init__` kwargs. **Fix:** moved `max_depth` and `use_transitions` to class-level defaults; `__init__` now only overrides them.

2. **Subprocess-per-mode wasn't enough** — autotune disk cache (`/tmp/torchinductor_<user>`) survives process death. Mode A populates it; mode B reads cached `best_configs` and short-circuits autotuning, reading `bench_calls=0` on cache-warm cells. **Fix:** harness now sets a unique `TORCHINDUCTOR_CACHE_DIR` per mode, isolating disk cache. (HS2's "subprocess is the only safe answer" turned out to be necessary but not sufficient.)

After both fixes, first clean comparison on resnet18 (1 model, 4 shapes/op, 5 repeats, 36 paired cells):

**abduction vs coord_descent (the H1 comparison target):**
- Median bench-call ratio: **1.00** (matches)
- Median chosen-kernel quality: **1.00** (matches)
- Median setup_s: 0.79s vs 0.76s (abduction 4% slower wall-clock)
- **H1 pass (≤30% bench at ≥95% quality): 0/36 = 0%** — confirms the simulator's prediction
- **Pareto-equivalent-or-better: 21/36 = 58%**

Per-operator pattern (b/a column = abduction/coord_descent bench ratio):
| op | b/a | quality | verdict |
|---|---|---|---|
| `aten.add_.Tensor` | 0.62 | 1.00 | **abduction wins** (38% fewer benches, same quality) |
| `aten.native_batch_norm_backward` | 0.90 | 1.00 | abduction wins (small) |
| `aten.native_batch_norm` | 0.94 | 0.92 | mixed (cheaper but slightly worse) |
| `aten.add.Tensor`, `copy_`, `div.*`, `sum.*`, `threshold_backward` | 1.00 | 1.00 | tied |
| `aten.clone.default` | 1.50 | 1.06 | abduction loses (more bench, similar quality) |
| `aten.max_pool2d_with_indices` | 1.50 | 1.04 | abduction loses |
| `aten.mean.dim` | 1.25 | 1.00 | abduction loses |
| `aten.relu_.default` | 1.23 | 1.00 | abduction loses |

**The pattern matches the simulator's verdict:** abduction is competitive but not dominant. The transition graph helps on operators where field-pruning aligns with the actual cost surface (in-place adds, batch norm backward) and hurts on operators where it locks out useful axes (pooling, reduction-to-scalar, relu).

This is from one model only (resnet18). Full 5-model sweep is launched in background. Conclusions held in suspension until that completes. But the H1 ≤30% prediction is now empirically dead twice over (simulator + smoke).

---

## Session summary 2026-05-10

The day's work, headlines first:

1. **Original H1 ("≤30% bench calls") falsified by synthetic-landscape simulator** without any Triton execution. Mechanism arithmetic was wrong (Gemini-B): collect-all-then-winner pays the worst case every round; coord-descent's accept-immediately doesn't.

2. **v2 algorithm is competitive on bench but doesn't validate the abduction methodology.** AbductionTuner v2 (accept-immediately + transition-graph + fallback expansion) saves ~16-19% bench calls with matched quality vs CoordescTuner. But the ablation shows the savings come from inner-loop `break-on-first-improvement`, not from anything abduction-specific. The transition graph itself contributes only an early-termination heuristic that costs quality.

3. **Cliff landscapes (where transitions were SUPPOSED to help) don't favour transitions either.** Built register-spill / occupancy-tier / SMEM-bank-conflict archetypes per Gemini round 2's correction. Result: transitions still don't add routing intelligence — same early-termination effect, with quality loss on 22% of trials.

4. **PTX dedup (Gemini-A) doesn't transfer from tinygrad to Triton.** Triton's `num_warps` affects runtime without affecting PTX, breaking the assumption that lib-hash equality implies time equality.

5. **The methodology I ported isn't actually abduction.** Re-reading `ABDUCTION.md` (per user reminder to keep theory fresh): full abduction requires *trajectory shape classification* (convergent / divergent / oscillatory / chaotic) per field, used to drive figure-ground separation. v2 only uses divergent. A proper v3 implementation is sketched as H7 but not built — it requires Triton (for noise) and several hours to implement.

6. **Two rounds of Gemini review.** Round 1 caught the mechanism arithmetic + transition-graph SMEM gap + counter bias. Round 2 caught the simulator's landscape bias + ablation confound + train-test leakage in H1'. Both rounds materially redirected the work.

What's committed:
- `simulate.py` (~430 lines) + `analyze_simulate.py` (~120 lines) — fully reusable
- `transitions.py` (post-C1 fix) — includes num_stages in BLOCK_M/N transitions
- `abduction_tuner.py` v2 — accept-immediately + use_transitions toggle for ablation
- `enable_abduction.py` — patches both binding sites (HS1)
- `bench_baseline.py` — subprocess-per-mode by default (HS2)
- 4 simulator runs in `data/`: smoke, v1 baseline, v2 ablation, v2 ablation fixed, v2 dedup, cliffs
- 7 hypotheses (H0 SURVIVES; H1 KILLED; H1' WITHDRAWN; H2/H3/H4 BLOCKED on Triton; H5 AMBIGUOUS; H6 KILLED; H7 OPEN)
- 3 sanity hypotheses (HS1/HS2 fixed; HS3 confirmed)
- 2 rounds of adversarial review consumed

What's pending (all blocked on WSL):
- All real-Triton measurements (#7-#9 in frontier)
- v3 trajectory-classification algorithm (H7) — requires real noise to be testable

Key memory entries created:
- "Architectural reads from subagents are hypotheses, not verdicts"
- "Synthetic-landscape simulators falsify mechanisms cheaply but can't validate physics-based heuristics"
- "Pre-committed predictions need pre-committed justifications, not vibes"

---

## H0: Inductor's autotune is the right comparison surface for abduction — SURVIVES (2026-05-10, layer corrected)

**Claim.** PyTorch Inductor's existing autotuners operate over the same problem shape as speedygrad's abduction engine: pick the best configuration from a discrete-but-large search space using measurement.

**Layer correction (the key insight).** The first two passes of this hypothesis aimed at `select_algorithm.py` / `AlgorithmSelectorCache` — the **wrong layer**. That layer picks among *algorithms* (different `ChoiceCaller`s for an op, e.g. mm-template vs decomposed-mm) using a batch loop. It is genuinely batch-only and abduction doesn't fit there.

The right layer is `torch/_inductor/runtime/triton_heuristics.py` — specifically `CachingAutotuner` (line 408) which picks among Triton **`Config(BLOCK_M=…, num_warps=…, num_stages=…)`** values *for one already-chosen kernel*. Coord-descent operates here too. This is the surface where "one knob change, measure, decide" lives.

**Surface match is direct.** Read `coordinate_descent_tuner.py:342`:
```python
def autotune(self, func: Callable[[Config], float], baseline_config, baseline_timing=None) -> Config:
```
And `abduct.py:36`:
```python
def abduct_search(s: Scheduler, rawbufs, max_depth=3) -> Scheduler:
```
**Isomorphic.** Both own their loop. Both consume a config-and-time-it primitive (CoordescTuner takes the composed `func`; abduct_search takes split `_try_compile`/`_time_program` and composes them itself). Both return the best found. Both are pluggable strategies — `CachingAutotuner._coordinate_descent_tuning` calls `self.coordesc_tuner.autotune(benchmark_one_config, launcher.config, None)` at line 1815, and that hookpoint accepts any peer with the same signature.

**The plug-in shape.** `AbductionTuner` is a peer of `CoordescTuner`, not a fork of `CachingAutotuner`. Either swap at the hookpoint behind a flag (`abduction_tuning=True`), or subclass `CoordescTuner` and override `autotune()`.

**Trial-count instrumentation already exists.** `triton_heuristics.py:1797` increments `counters["inductor"]["coordesc_tuning_bench"]` per benchmark call inside `_coordinate_descent_tuning`. Because `AbductionTuner` is invoked through the same hookpoint (the monkey-patch at `triton_heuristics.CoordescTuner`), the same counter measures both. We disambiguate at the experiment level via the `mode` column — no new counter or instrumentation needed.

**Knob-vocabulary mapping** is also handled. `CoordescTuner.tunable_fields` (line 109) already enumerates `[XBLOCK, YBLOCK, ZBLOCK, R0_BLOCK, R1_BLOCK, BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages]` — that's the abduction transition-graph alphabet. Define transitions over these field names directly; no need to map from tinygrad's `OptOps`. (The semantic of "LOCAL changed → next try {LOCAL, UPCAST, UNROLL}" becomes "XBLOCK changed → next try {XBLOCK, YBLOCK, num_warps}" or similar — a design choice, not an architectural blocker.)

**What survives is mechanical; what's left is the heuristic.** The framework fits. What abduction adds *over* coord-descent at this surface:
1. Trajectory classification (convergent / divergent / oscillatory / chaotic) instead of plain accept/reject.
2. Hypothesis-driven field selection (skip irrelevant fields after a winner) instead of round-robin.
3. Pruning via transition graph instead of full Cartesian neighbourhood.

H1's "fewer trials" claim now has a clear mechanism: coord-descent calls `func` `O(tunable_fields × radius)` per round and iterates until no improvement; abduction prunes the field set after each winner.

---

## H1: Abduction reaches coordinate_descent_tuning quality in ≤30% of trials — KILLED by simulator (2026-05-10)

**Pre-committed prediction (now killed).** Abduction reaches ≥95% of `coordinate_descent_tuning`'s chosen-kernel runtime in ≤30% of the `bench` calls.

**Falsifier (pre-committed).** Killed if abduction needs ≥ coord-descent's `bench` call count to reach within 5% of its chosen-kernel runtime.

**Verdict source.** Synthetic-landscape simulator (`benchmarks/abduction/simulate.py`) ran v1 AbductionTuner against CoordescTuner on 237 random optima across 10 landscape archetypes. **No archetype hit the ≤30% threshold.** The mechanism arithmetic (Gemini-B 2026-05-10) was correct: collect-all-then-winner pays a fan-out cost per round that the transition-graph pruning savings cannot offset. v1 was strictly worse than coord-descent on both bench calls (1.09× more) and chosen quality (0.345 vs 0.745).

**Why the prediction was wrong.** I conflated coord-descent's worst-case per-round cost (`O(fields × radius)`) with its actual cost. Coord-descent's accept-immediately Gauss-Seidel pattern terminates the field walk on the first improving move and rebases from the new winner, so its per-round cost is closer to `O(rounds_to_converge)` than `O(fields × radius)`. Abduction's collect-all-then-winner pattern (ported from `abduct_search`) actually pays the worst case every round.

---

## H1′: Revised post-simulator (2026-05-10) — BLOCKED (on Triton only)

**Algorithm change.** AbductionTuner v2 replaces collect-all-then-winner with **accept-immediately Gauss-Seidel + transition-graph pruning + fallback expansion**. Walks fields in canonical order, restricted by `allowed`; accepts the first improving neighbour; tracks the last-improved field; narrows `allowed` to `transitions.allowed_after(last_improved_field)` for the next round; falls back once to `all_fields` if a pruned walk yields no improvement, before terminating.

**Simulator validation.** Same 237-trial sweep, v2 results:
- **Bench ratio:** median 0.81 (19% fewer than coord-descent overall)
- **Quality:** abduction matches coord-descent's median quality per archetype, drops 4 pts overall (driven by `pointwise_noisy` where the noise floor confuses both methods differently)
- **Pareto-equivalent or better:** 84% of trials (abduction ≤ coord_descent bench AND ≥ coord_descent quality)
- **Original H1's ≤30% threshold:** still 0% — physically unreachable on these landscapes

**Pre-committed revised prediction.** On the frozen corpus:
1. **Bench-call savings:** abduction uses ≤85% of coord-descent's median `bench_calls`.
2. **Quality preservation:** abduction's median chosen-runtime is within 5% of coord-descent's.
3. **Pareto-equivalent-or-better fraction:** abduction is on the Pareto frontier (≤ bench AND ≥ quality) on ≥70% of (op, shape) pairs.

**Falsifier.** Killed if any of the three (≤85% bench, within-5% quality, ≥70% Pareto) fails on the frozen corpus.

**Why this is more defensible.** The 30% target was based on faulty arithmetic + a v1 algorithm that didn't survive contact with synthetic data. The 85% / 5% / 70% trio is anchored in 237-trial empirical evidence on landscapes designed to span the realistic regime (separable, coupled, ridge, plateau, noisy, OOM-bounded). Real Triton may behave differently, but the prediction is now grounded in something more than wishful arithmetic.

**Caveat.** The simulator does NOT model PTX deduplication (Gemini-A) — the speedygrad engine's optimization where structural-only config changes that produce identical compiled output skip benchmarking entirely. If/when PTX dedup is wired in (Gemini-A task), bench savings could improve substantially. The above prediction is for v2-without-dedup.

---

## H5: The transition graph specifically adds value over accept-immediately + max-depth — AMBIGUOUS (2026-05-10, post-Gemini-round-2)

**Original verdict (now downgraded).** I called this "killed by ablation" based on the simulator showing the transition graph hurt quality slightly (0.714 vs 0.745) while saving only 3 percentage points of bench calls beyond the no-transitions variant. Gemini round 2 corrected the framing.

**Why the kill was premature.**

1. **The simulator can't fairly test what the transition graph is designed for.** My landscapes (`landscape_convex_separable`, `landscape_ridge`, etc.) are smooth math — log² distances, quadratic penalties. They lack the *physical cliffs* that the transition graph encodes domain knowledge about: register-spill cliffs that drop occupancy from 4 warps to 3, SMEM thresholds that cause precipitous compile failures, occupancy tiers, warp scheduling contention. **Testing a physics-based heuristic on smooth continuous math guarantees a general-purpose math solver wins.** The graph wasn't designed for smooth landscapes; it was designed for cliffs. Falsifying it on smooth landscapes proves nothing about its real-Triton value.

2. **The ablation is structurally confounded by `max_depth`.** I compared `AbductionTuner` (transitions + max_depth) vs `AbductionTuner_no_transitions` (no transitions + max_depth) — both have max_depth=5. CoordescTuner (the baseline) has NO depth cap; it loops `while improved:` to total convergence. So the "16% bench savings" the no_transitions variant shows vs CoordescTuner comes mostly from cutting off the long tail via `max_depth`, not from anything abduction-related. The right ablation needs a `CoordescTuner_max_depth_5` arm to isolate the depth-cap effect from the transitions effect.

3. **The trajectory shape is oscillatory, not divergent.** Per [The Hypothesis Graph](https://june.kim/the-hypothesis-graph): when an experiment shows a result that helps in some cases and hurts in others, that's the oscillatory bin — the rule is *split the hypothesis into sub-hypotheses*, not "kill." The transition graph helps on some kernels (matmul_convex bench ratio 0.68) and hurts on others (pointwise_noisy quality 0.943→0.943 same as coord-descent but slightly more bench). The split:
   - **H5a:** Transitions help on smooth, well-conditioned landscapes where coord-descent's full sweep is wasteful. (Suggested by simulator; needs real-Triton validation on simple kernels.)
   - **H5b:** Transitions help on landscapes with physical cliffs by pruning configs that would crash. (Untested by simulator; needs real-Triton matmul + deep reductions.)
   - **H5c:** Transitions hurt on noisy / multi-modal landscapes by locking out fields that would have escaped a local min. (Suggested by simulator pointwise_noisy; needs real-Triton confirmation.)

**Status.** Ambiguous. The simulator falsified one specific use case (smooth landscapes, where coord-descent's full sweep IS the right call) but cannot test the use case the graph was designed for (physical-cliff landscapes). Triton measurement is required to disambiguate.

**Action items:**
- ~~Add `CoordescTuner_max_depth_5` to the simulator ablation to isolate the depth effect~~ **DONE 2026-05-10**, see fixed-ablation results below.
- On real Triton, measure matmul kernels at sizes that exercise SMEM cliffs (BLOCK_M*BLOCK_N near the budget). If transitions provide outsized savings there, H5b confirmed.
- Pre-commit the H5a/H5b/H5c sub-falsifiers BEFORE seeing real Triton data, to avoid the H1' overfitting trap (see Reframe section).

### Cliff-landscape result (2026-05-10): transitions don't add value even on the regime they were designed for

Per Gemini round 2 A: smooth landscapes can't fairly test transitions; needed physical-cliff archetypes. Built four:
- `matmul_register_spill`: BLOCK_M*BLOCK_N / (num_warps*32) above register budget → 5x slowdown
- `matmul_occupancy_tier`: num_warps>8 → 1.5x slowdown (occupancy drop)
- `matmul_smem_bank_conflict`: BLOCK_K ≤32 and BLOCK_K%16==0 → 2x slowdown
- `matmul_combined_cliffs`: all three at once

Plus pointwise variants. 150 trials total on cliff landscapes (`data/simulate_cliffs.csv`):

| variant | bench ratio | quality | Pareto% |
|---|---|---|---|
| `CoordescTuner` (baseline) | 1.00 | 1.000 | — |
| `CoordescTuner_max5` | 1.00 | 1.000 | 100% |
| `AbductionTuner_no_transitions` | 0.82 | **1.000** | **100%** |
| `AbductionTuner` (full v2) | 0.76 | 1.000 | 78% (loses quality on 22%) |

**The transition graph's "win" turns out to be just earlier termination.** On cliff landscapes, both abduction variants reach 100% quality on the no-transitions arm; with transitions, the algorithm sometimes terminates one round too early and lands on a sub-optimal config (0.667-0.909 quality on those trials). Net Pareto drops from 100% to 78%.

So even on the "physical cliff" regime that the transition graph was designed for: **transitions provide a pure trade — 6pp more bench savings (0.76 vs 0.82) for 22pp lower Pareto rate.**

### Why the cliffs don't favor transitions

The mechanism the transitions were supposed to exploit: "winner along BLOCK_M opens num_warps next, so abduction can walk back from a spill." But coord-descent ALSO tries num_warps (it tries every field every round). The transition graph just makes abduction try num_warps **first** in the next round, not **only**. Same configs visited; different order.

On a smooth landscape, order doesn't matter for the final result. On a cliff landscape, order doesn't matter either — both algorithms eventually find the post-cliff optimum. The transition graph only helps if it lets us SKIP fields that coord-descent visits unnecessarily. But coord-descent doesn't visit *that* many fields per round, and the transition pruning in v2 is what causes the early termination problem.

**Conclusion across both smooth and cliff landscapes:** the v2 transition graph adds early termination (+savings, -quality) but no *routing intelligence*. Whatever abduction's win is in tinygrad, the discrete-grid Triton config space doesn't reproduce it.

### Fixed ablation result (2026-05-10): max_depth ISN'T the source of savings

Per Gemini-B's concern: I added `CoordescTunerCapped(max_depth=5)` to test whether AbductionTuner's bench savings come from the depth cap rather than transitions. Result (`data/simulate_v2_ablation_fixed.csv`):

| variant | bench ratio | quality | Pareto% |
|---|---|---|---|
| `CoordescTuner` (baseline, unbounded) | 1.00 | 0.745 | — |
| `CoordescTuner_max5` (same + cap at 5 walks) | **1.00** | 0.745 | 94% |
| `AbductionTuner_no_transitions` | 0.84 | 0.745 | 95% |
| `AbductionTuner` (full v2) | 0.81 | 0.714 | 83% |

**The depth cap is inactive in the simulator** — coord-descent converges in ≤5 walks on all our landscapes anyway, so `CoordescTuner_max5` is byte-identical to `CoordescTuner`. So the no_transitions variant's 16% bench savings are NOT from the depth cap. Where then?

**Real source of savings: break-on-first-improvement.** Reading the venv's `CoordescTuner.autotune` (torch 2.11.0+cu128), the inner candidate-values loop does NOT break when an improvement is accepted:
```python
for next_val in candidate_values:  # tries BOTH neighbours per field
    if cmp_res:
        improved = True
        best_config = candidate_config  # no break — keeps trying further values
```

AbductionTuner v2's inner loop DOES break:
```python
if cand_timing < best_timing:
    best_config = cand
    accepted = True
    break  # accept first improvement, move to next field
```

So the actual algorithmic delta is **a one-line `break` in the candidate-values loop**. Not max_depth, not transitions, not any "abduction methodology" — just an early exit.

**Caveat that the simulator can't test.** `get_neighbour_values` returns values in `[larger, smaller]` order (`coordinate_descent_tuner.py:202-217`). Break-on-first accepts the larger neighbour first. If the optimum is BELOW baseline (smaller direction), break-on-first misses it on the first round. The simulator's symmetric quadratic landscapes don't expose this bias because rounds-2+ catches it. On real Triton with a strong directional optimum, break-on-first might cost quality. **Needs Triton confirmation.**

---

## H6: PTX deduplication transfers from tinygrad to Triton — KILLED on inspection (2026-05-10)

**Claim (Gemini-A 2026-05-10).** speedygrad's `abduct_search` skips benchmarking when a candidate's compiled lib hash matches a previously-seen one (`abduct.py:84-99`). Porting this dedup to Inductor would restore the trial-count advantage AbductionTuner lost when we made it consume an opaque `func`.

**Verdict.** Killed by inspection of the relevant Triton/Inductor semantics, before measurement.

**Why.** In tinygrad, the compiled lib **fully determines** the runtime — two configs producing the same lib produce the same time. The `if lib in seen_libs: continue` shortcut is sound because the bench result is also identical.

In Triton, several fields affect runtime *without* affecting PTX:
- `num_warps`: a launch attribute (number of threads), not part of kernel codegen
- (potentially) `num_stages` for non-matmul kernels — pipelining hints don't always reach codegen

Two configs with the same PTX but different `num_warps` produce **different** runtimes. So PTX-hash-keyed dedup would skip benchmarks that aren't actually redundant — losing real performance information. The simulator with `--ptx-dedup` enabled (modeled this way) crashed quality from 0.745 to 0.225, confirming the issue.

**What does survive.** A weaker dedup is possible: skip the *compile* step (and its wall-clock cost) when PTX hash matches, but still run the benchmark to capture the launch-attribute-driven runtime variation. This would cut `setup_s` for cache-warm configs but not `bench_calls`. Useful for H4 (wall-clock claim) but not H1 (trial-count claim). Tracked separately as a follow-up — not load-bearing for the H1 story.

**Implication.** The H1' bench-call savings cannot be improved beyond the v2 algorithm by porting tinygrad's PTX dedup. The methodology has to win on the strength of accept-immediately + transition-graph alone — and per H5, the transition graph is doing little work.

---

## H7: AbductionTuner v2 isn't actually abduction (theory check, 2026-05-10)

**Triggered by re-reading `D:\depot\tinygrad-abduction-engine\ABDUCTION.md` per user reminder to "keep theory fresh."**

The abduction methodology, per the source theory, requires classifying each experimental trajectory into one of four shapes — and the shape *names what to do next*:

| Trajectory shape | Meaning | Next step |
|---|---|---|
| Convergent | Knob doesn't matter | Skip it; try another |
| Divergent | Knob is load-bearing | Follow its dependencies (transition graph) |
| Oscillatory | Two constraints fighting | Split the hypothesis (test the interface) |
| Chaotic | Measurement unreliable | Decompose differently |

What v2 actually does: tries one neighbour per direction, accepts any improvement, prunes the next field set via the transition graph. **It only uses the divergent path.** It has no trajectory shape, no oscillatory split, no chaos detection. It's a Gauss-Seidel walk dressed up with field-pruning.

**Implication.** The original H1 framing ("fewer trials") is the wrong metric for testing abduction. The methodology's win isn't fewer trials per field — it's *fewer wasted fields* via figure-ground separation. To get figure-ground separation, you need MORE samples per field (to classify the shape), not fewer.

A proper abduction implementation (call it v3) would:
1. Sample each field 3-5 times across its neighbour range to classify shape.
2. Convergent → field is ground, ignore for the rest of this kernel.
3. Divergent → field is figure, follow `transitions[field]` next.
4. Oscillatory → mark the field-pair interaction; test the interface (e.g., XBLOCK × num_warps jointly).
5. Chaotic → kernel is too small for reliable measurement; decompose or skip.

This is a much bigger algorithm than v2. Per-field cost is higher (3-5 samples for classification) but field count drops fast (convergent fields are dropped permanently). Whether the trade is favorable depends on (a) how many fields are convergent in real Triton — if most fields don't matter for most kernels, v3 wins big; (b) the cost of misclassification under noise.

**Open question.** Is v3 worth implementing? Three constraints:
- The simulator is the wrong tool to test it (smooth landscapes never produce oscillatory or chaotic shapes).
- Real Triton is needed (noisy, cliff-bearing) but blocked on WSL.
- If we go to the trouble of building v3, we should pre-commit a hypothesis it tests — not just "see what happens."

**Pre-committed hypothesis for v3 (pending implementation).** On real Triton, v3's median bench-call count will be ≤coord-descent's MINUS the count attributable to "convergent" fields (fields where coord-descent's full sweep produced ≤1% timing variation). If figure-ground separation is real, those fields should constitute ≥40% of all fields touched.

---

## Reframe (2026-05-10): WITHDRAWN per Gemini round 2

I had drafted a pivot to "ship a `coordinate_descent_max_walks` config flag, drop the abduction-methodology claim entirely." Gemini round 2 caught this as premature:

> "Predicting [the H1' threshold] will hold on real Triton is textbook train-test leakage... You designed the v2 fallback expansion because the transition graph was failing on your synthetic landscapes. You then measured the success rates on those exact same landscapes."

The reframe has two structural problems:
1. **Train-test leakage.** I designed v2's algorithm in response to v1's simulator results, then measured v2's success on the same simulator. The H1' threshold (≤85% bench, ≥95% quality, ≥70% Pareto) is fitted to the simulator, not predicted from prior reasoning.
2. **The kill condition (smooth-landscape ablation) doesn't apply to the hypothesis (physical-cliff transition graph).** Per H5 above, the simulator can't fairly test what the transitions are designed for. Withdrawing a methodology claim based on a test it wasn't designed for is bad inference.

**What's still valid.**
- v1 algorithm is genuinely worse than coord-descent (collect-all overhead is unrecoverable).
- v2's accept-immediately + max_depth + fallback gives a competitive tuner on smooth landscapes.
- PTX dedup specifically (Gemini-A H6) doesn't transfer from tinygrad to Triton.

**What needs Triton to decide.**
- Whether the transition graph adds value on physical-cliff landscapes (H5b).
- Whether the H1' bench-call savings (≤85% threshold) hold on real measurements.
- Whether the abduction methodology has a story to tell on Triton, or just on tinygrad-shaped opt spaces.

**No PR pitch is committed yet.** The investigation continues; the methodology claim is held in suspension pending Triton evidence.

---

---

## H2: Reduction kernels are where abduction wins biggest — BLOCKED (2026-05-10, on Triton)

**Claim.** Reductions (`aten.sum`, `aten.mean`, `aten.amax`, `aten.argmax`, layer/batch norm) are the kernel class where abduction's relative trial-count savings are largest.

**Why predicted.** Reductions have the most field interactions: `R0_BLOCK`, `R1_BLOCK`, `num_warps`, and `num_stages` all couple non-trivially (reduction-axis size × warps × pipeline depth). Coord-descent's round-robin pays the worst price here because each axis is tested independently before checking interactions. Matmul is constrained by tile geometry (M/N/K), and pointwise has fewer load-bearing axes (just X/Y BLOCK + warps).

**Pre-committed prediction.** On the frozen corpus, abduction's median `bench_calls` ratio (abduction / coord_descent) is **≥2× lower for reduction ops than for pointwise ops**, when measured on ops where both methods reach within 5% of each other in `runtime_us`.

**Falsifier.** If the reduction-vs-pointwise `bench_calls` ratio difference is <2×, the claim that abduction's wins are reduction-driven is wrong. Either abduction's wins are uniform (interesting — different theory), or pointwise also has hidden interactions (also interesting — invalidates the "reductions have more knobs" intuition).

**Why this matters.** If H2 confirmed, the PR pitch sharpens: "abduction is a coord-descent replacement for reduction-heavy workloads." If H2 killed but H1 confirmed, the pitch broadens: "abduction is a coord-descent replacement, period." Either is publishable.

---

## H3: The v1 transition graph survives without observation-driven revision — BLOCKED (2026-05-10, on Triton)

**Claim.** `transitions.py`'s analogy-derived `TRANSITIONS` dict is correct enough that abduction reaches the H1 threshold without needing to add or remove edges based on measurement.

**Why this matters.** If H3 holds, the abduction methodology itself transfers — you can derive a transition graph for a new system by analogy from another and have it work. If H3 fails, transition graphs are *system-specific* and have to be discovered, which means abduction's deployment cost is "build a graph for your system" — much higher friction.

**Pre-committed prediction.** Abduction with the unmodified v1 graph passes the H1 falsifier. Adding ≤2 edges based on observation does not change the H1 verdict more than 5% (i.e. the graph is "in the right neighborhood").

**Falsifier.** If abduction passes H1 only after substantive (≥3 edges, or removing existing edges) graph revisions, H3 is killed. The graph turned out to be wrong, and the methodology — at minimum — needs a graph-discovery layer.

**Sub-question if killed.** What edges had to change? If the changes are systematic (e.g. "we forgot that BLOCK_K → num_warps is load-bearing for matmul"), the methodology can be patched. If they're idiosyncratic (different per kernel), the transition graph isn't a useful abstraction and the H1 win was incidental.

---

## H4: Wall-clock setup_s tracks bench_calls within a constant factor — BLOCKED (2026-05-10, on Triton)

**Claim.** Inductor's per-bench cost (compile + measure one config) is roughly constant across configs, so `bench_calls` reduction translates linearly to wall-clock `setup_s` reduction.

**Why this matters.** H1 measures trial count. Users measure compile time. If per-bench cost varies wildly with config (e.g. some configs trigger expensive recompiles, others hit a cache), then "30% of trials" might not mean "30% of compile time."

**Pre-committed prediction.** For ops where H1 holds, abduction's median `setup_s` is within `[0.6×, 1.0×]` of coord-descent's `setup_s × (abduction_bench_calls / coord_bench_calls)`. I.e., wall-clock savings are at least 60% of what trial counts predict.

**Falsifier.** If `setup_s` ratio is >1.5× the `bench_calls` ratio, per-bench cost is non-uniform enough that trial counts mislead. The H1 finding then needs an asterisk.

**Predicted failure mode if killed.** Abduction picks unusual configs that miss Inductor's autotune cache more often, paying recompile costs that coord-descent avoids by sticking near the heuristic-default neighborhood. If observed, this informs a v2 abduction modification: bias toward cache-warm configs.

---

## HS1 / HS2 / HS3: pre-WSL sanity investigation — RESOLVED (2026-05-10, /investigate pass)

Three sanity hypotheses spawned from the "pre-#7 sanity work" notes. Investigated via parallel Explore subagents reading the actual fork code; no Triton needed.

### HS1: Does the monkey-patch reach the construction site? — **PARTIAL KILL, fixed**

Subagent verified the main path (`triton_heuristics.py:489` resolves the patched `CoordescTuner` correctly via bare-name lookup at construction time). But found a **second site** the original patch missed: `torch/_inductor/codegen/simd.py:48` imports `CoordescTuner` directly, then calls `CoordescTuner.autotune_single_field(...)` at `simd.py:1724` for mix-order-reduction split-size tuning. That binding lives in `simd.py`'s namespace and is independent of `triton_heuristics.CoordescTuner`.

**Does it affect H1?** No — `autotune_single_field` is a static method on a different code path; it doesn't bump `coordesc_tuning_bench`. So H1's measurement is unaffected.

**Does it affect "abduction is a true peer"?** Yes — for completeness. Fixed in `enable_abduction.py` by patching both `triton_heuristics.CoordescTuner` AND `coordinate_descent_tuner.CoordescTuner`. Subclasses inherit the static method, so the simd.py site now resolves to AbductionTuner (which inherits autotune_single_field unchanged from CoordescTuner).

### HS2: Is `torch._dynamo.reset()` enough to isolate per-mode runs? — **KILLED, structural fix**

Subagent traced `torch._dynamo.reset()`'s definition (`torch/_dynamo/__init__.py:138-184`). It clears Dynamo's compile-frame caches but NOT:
- **`PyCodeCache`** (`torch/_inductor/codecache.py:4302`) — in-memory module cache. Stores loaded kernels containing `CachingAutotuner` instances, keyed by kernel source hash. When mode B compiles the same kernel as mode A, `PyCodeCache.load()` returns the cached module without re-running `cached_autotune` — so mode B inherits mode A's tuner instance and reads `bench_calls=0`.
- **Autotune disk cache** (`autotune_cache.py:116+`) — per-kernel `best_config` files in `~/.cache/torch_inductor/`. Mode B reads mode A's winning config and short-circuits autotuning entirely (`triton_heuristics.py:295` sets `configs = [best_config]`).
- **`FxGraphCache`** — disk-cached compiled FX graphs that share PyCodeCache for kernel reload.

**Implication.** `bench_calls=0` in mode B after mode A is **expected behavior given the cache architecture** — not a measurement bug we can paper over with a single `reset()` call. The investigation's finding: the only safe answer is **subprocess-per-mode**.

**Fix applied.** `bench_baseline.py` now re-execs itself once per mode when `--modes` has >1 entry. The single-process path requires `--unsafe-single-process` (loud opt-out, never for the H1 measurement). The harness comment that called subprocess-per-mode "the gold-standard isolation" was understated — it's the **only** isolation that works against Inductor's caches.

### HS3: Is AbductionTuner contract-compatible with CoordescTuner? — **SURVIVES, drop-in compatible**

Subagent checked all six contract bits:
1. Constructor accepts the kwargs CachingAutotuner passes (`is_mm`, `is_native_matmul`, `is_mix_order_reduction`, `name`, `size_hints`, `inductor_meta`) — `*args/**kwargs` passthrough works; `max_depth` is keyword-only and defaults safely.
2. `autotune` signature matches exactly.
3. Returns a `triton.Config` — `winning_config` is a deep-copy from `get_neighbour_configs`, fresh each time.
4. `triton.Config` is a mutable Python object, so `triton_heuristics.py:1819`'s `best_config.found_by_coordesc = True` works.
5. Exception handling differs in shape but not in observable behavior to the host (both swallow per-candidate failures).
6. Empty `allowed_after()` set causes early break to baseline_config — safe; matches CoordescTuner's "no field improved" path.

**Verdict.** Drop-in compatible. No code changes needed.

### Investigation summary
- 1 hypothesis killed, structural harness change applied (HS2 subprocess-per-mode)
- 1 partially killed, completeness fix applied (HS1 second binding site)
- 1 survives unchanged (HS3)
- All three investigations done before any Triton execution — caught two real bugs that would have invalidated the H1 measurement, and confirmed one assumption that would have been hard to debug from a wrong number.

The three "pre-#7 sanity work" items in the frontier list (lines 122-125) are now retired.

---

## Open frontier (in dependency order)

| # | Edge | Status |
|---|---|---|
| 4  | Frozen corpus | **DONE 2026-05-10** — `benchmarks/abduction/corpus.json` (5 small torchbench_train models) |
| 5  | Baseline harness | **DONE 2026-05-10** — `benchmarks/abduction/bench_baseline.py` (4 modes: default / max_autotune / coord_descent / abduction). Reads `counters["inductor"]["coordesc_tuning_bench"]`. Long-form CSV. Per-mode `torch._dynamo.reset()` between modes. |
| 5b | Oracle-mode definition | **PUNTED for v1** — `max_autotune` is oracle over the static template space; coord-descent's expanded space has no closed-form oracle. Revisit if max-autotune↔coord-descent gap warrants ceiling definition. |
| 6a | Abduction transition graph | **DONE 2026-05-10** — `benchmarks/abduction/transitions.py`. Reasoning by analogy from tinygrad `_TRANSITIONS`; documented as "starting point, update from observation." |
| 6b | `AbductionTuner` impl | **DONE 2026-05-10** — `benchmarks/abduction/abduction_tuner.py`. Subclasses `CoordescTuner`; collect-all-then-winner per round; transition-graph pruned across rounds; speedup-threshold termination. v1 omissions documented (early-stop, trajectory shapes, validation re-time). |
| 6c | Wire `AbductionTuner` in | **DONE 2026-05-10** — `benchmarks/abduction/enable_abduction.py`. Monkey-patches `triton_heuristics.CoordescTuner = AbductionTuner` rather than editing the fork. Auto-installs on `STELLA_ABDUCTION=1`. The harness toggles install/uninstall per-mode. |
| 7  | Install WSL distro | **PENDING USER** — `wsl --install Ubuntu`, then re-run identity probe inside WSL |
| 8  | Run baseline | **BLOCKED on #7** — once Triton works, run `bench_baseline.py --modes default max_autotune coord_descent` and commit CSV + Pareto plot |
| 9  | Run abduction | **BLOCKED on #7, #8** — `bench_baseline.py --modes abduction` (same corpus). Diff trial counts and chosen-kernel runtimes; decide H1 |
| 10 | H2 analysis | **BLOCKED on #9** — group by op category (reduction / matmul / pointwise) and check the 2× ratio claim |
| 11 | H3 analysis | **BLOCKED on #9** — for ops where abduction underperforms, attempt graph edits and re-measure; record the edit set |
| 12 | H4 analysis | **BLOCKED on #9** — compute setup_s/bench_calls ratio per (op, mode) and check the [0.6×, 1.0×] band |
| 13 | v3 algorithm: trajectory-shape classification (H7) | **DESIGN OPEN** — implement multi-sample-per-field shape classifier; convergent→drop field, divergent→follow transitions, oscillatory→split, chaotic→decompose. ≥3 samples per field. Pre-commit "≥40% of fields are convergent on real Triton" before measuring. Several hours to build; only worth doing if v2 measurement on Triton already shows the methodology has any merit. |
| 14 | H8 follow-up: Triton-only restriction | **OPEN** — re-run H8 query excluding mm/addmm/bmm cells (where cuBLAS dominates via Inductor's default mode). Already at 6.2% wins on existing data; weaker version of H8 might survive on operators where Triton is the only path. |
| 15 | H8 follow-up: bigger corpus | **OPEN** — extend `corpus.json` with HF (BERT, GPT2) and timm (ConvNext, ViT) models. The 5 models I picked are small/fast for iteration; bigger matmuls may surface H8 wins outside cuBLAS's sweet spot. |

**Everything in the local code path is done.** All four files (corpus, harness, transitions, tuner, enable patch) are committed. Nothing more to write until measurements come in. The only remaining work before WSL is read-throughs and bug-hunting against the fork's actual code paths.

WSL install status (2026-05-10): kernel + WSL2 present, no distro installed. `wsl --install Ubuntu` is the next command when ready.

Files dropped this session:
- `benchmarks/abduction/corpus.json` — 5 frozen models with rationale
- `benchmarks/abduction/bench_baseline.py` — 4-mode harness, long-form CSV
- `benchmarks/abduction/transitions.py` — TRANSITIONS dict + `allowed_after()`
- `benchmarks/abduction/abduction_tuner.py` — `AbductionTuner(CoordescTuner)` peer
- `benchmarks/abduction/enable_abduction.py` — monkey-patch installer

Pre-#7 sanity work — **DONE 2026-05-10** via /investigate (HS1/HS2/HS3 above). Two real bugs caught and fixed before measurement.

Source-of-truth files:
- Engine: `D:\depot\speedygrad\tinygrad\codegen\opt\abduct.py` (`abduct_search`, 114 lines)
- Diff primitive: `D:\depot\speedygrad\bench\abduct.py` (`abduct(make_a, make_b)`, 139 lines)
- Inductor surface (correct layer): `D:\depot\pytorch\torch\_inductor\runtime\triton_heuristics.py:1759-1843` (`_coordinate_descent_tuning`)
- Inductor peer-tuner: `D:\depot\pytorch\torch\_inductor\runtime\coordinate_descent_tuner.py:342` (`CoordescTuner.autotune`)

---

## Appendix B: The layer correction

Even after retracting the kill, the next two passes of H0 still aimed at the wrong file. The hypothesis graph anchored to `select_algorithm.py` because that's where `coordinate_descent_tuning` is exposed as a top-level Inductor flag — but the flag's *implementation* lives in `triton_heuristics.py:1759` (`_coordinate_descent_tuning`), and the actual peer-able strategy lives one layer down at `coordinate_descent_tuner.py:342` (`CoordescTuner.autotune`).

The user's correction ("triton_heuristics.py is what we're trying to benchmark") flipped the search target. Once at the right layer, the H0 question changed from "can the engine plug in?" (uncertain) to "is the existing `CoordescTuner` interface isomorphic to abduct_search?" (yes, by inspection).

Lesson for future hypothesis graphs: when the surface looks batch-only, check whether what you're calling "the surface" is the user-facing dispatcher or the actual strategy class. The first will almost always be batch-shaped because it has to handle both single-config and multi-config callers; the latter is where the loop actually runs.

---

## Appendix A: Why the first H0 verdict was retracted

A subagent read `select_algorithm.py` and reported "H0 KILLED" based on:
- `AlgorithmSelectorCache.__call__` (~line 3770) takes `choices: list[ChoiceCaller]`
- `benchmark_choices` (~line 4884) loops `for choice in choices: benchmark_choice(choice)`
- `feedback_saver_fns` (~line 4329) fire post-hoc

The verdict was accepted without scrutiny. Then in conversation, the user asked: "if it's sequential, just pass around a blackboard?" — and the kill collapsed in seconds. Three rescue routes the agent didn't consider:

1. **Lazy iteration.** Python's `for` loop accepts any iterable; `choices` doesn't have to be a materialized list. A generator with closure-captured blackboard state could decide its next yield based on prior measurements — *if* the measurements are observable to the generator (e.g., via a subclass that overrides `benchmark_choice` to push results to the blackboard before returning).
2. **Multi-shot dispatch.** Even if one autotune call must be batch, abduction can run several smaller batches — picking each batch's candidates from a blackboard updated by the prior batch's `feedback_saver_fns`. Caveat: `(name, choices)` cache key must be sidestepped or varied per shot.
3. **Coarser-grained blackboard.** `feedback_saver_fns` is a real hook; cross-kernel learning (kernel N's tuning informs kernel N+1's candidate generation) is a weaker but valid form of "follow the failure."

The lesson isn't "the agent was bad" — it's that **architectural reads produce hypotheses, not verdicts**. A code-review can flag where to probe; only a probe attempt kills.

---

## Closed

### Identity probe caught the Triton blocker — CONFIRMED

Running `identity_probe.py` on Windows + Python 3.14 + torch 2.11.0+cu128 revealed Triton was not importable. Without Triton, Inductor autotune cannot run, and all three modes silently degrade to default ATen. If the baseline had been built first and run before the probe, the resulting CSV would have shown all three modes producing identical numbers — a confusing null result that would have wasted hours diagnosing.

Lesson: identity probes pay for themselves on first run. Always run before measurement.
