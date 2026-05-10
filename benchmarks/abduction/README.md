# Abduction port — PyTorch Inductor

Experiment repo for porting [speedygrad](https://github.com/kimjune01/speedygrad)'s abduction engine into PyTorch Inductor's autotuner.

## Goal

Show that abduction reaches `coordinate_descent_tuning` quality in fewer trials, on a frozen corpus extracted from PyTorch's own `benchmarks/dynamo/microbenchmarks/operator_inp_logs/`. If true, this is a real product win for `torch.compile`'s autotune mode (compile time is the #1 user complaint) and a credible pitch to land at PyTorch.

If false, the kill is itself evidence about what abduction *requires* — whether its wins depend on tinygrad's UOp shape or transfer to finite-grid config spaces.

## Falsification

Pre-committed before measurement.

> Hypothesis killed if abduction needs ≥ coordinate_descent_tuning's trial count to reach within 5% of its chosen-kernel runtime, on the frozen corpus.

## Mise-en-place

| Item | Value |
|---|---|
| **System under test** | PyTorch Inductor's autotuner. Source: `D:\depot\pytorch` (fork at HEAD `05b4150f3b7`). Baseline runs use the public wheel (`torch==2.11.0+cu128`) installed in `D:\depot\speedygrad\.venv` to keep "what we measured" reproducible. The fork is for *implementing* abduction as a fourth selector, not for the baseline. |
| **Perturbation access** | Yes — env vars (`TORCHINDUCTOR_MAX_AUTOTUNE`, `TORCHINDUCTOR_COORDESC_TUNING`), Python config (`torch._inductor.config.*`), and (later) monkey-patching `AlgorithmSelectorCache`. |
| **Identity probe** | `identity_probe.py` — prints resolved torch version, CUDA backend, autotuner class, `select_algorithm.AlgorithmSelectorCache` identity, and which mode each env var actually selects. **Run before every measurement.** Labels are not identity. |
| **Methodology** | Abduction (hypothesis graph: this file's neighbor `HYPOTHESIS_GRAPH.md`). Prereg checklist for the baseline measurement: predicted numbers and falsifier above are pre-committed before any data is collected. |
| **Ambiguity heuristic** | When two configs look equally good: prefer fewer trials. Ties broken by lower runtime variance (interquartile range). |
| **Dependency graph** | Linear: identity probe → corpus freeze → baseline harness → baseline data → abduction port → abduction data → comparison. No parallelism — each step's artifact gates the next. |
| **Goal** | Land abduction in PyTorch Inductor as an alternative to `coordinate_descent_tuning`. Stop condition: either a merged PR or a CONFIRMED kill in `HYPOTHESIS_GRAPH.md`. |

## Files

- `identity_probe.py` — runtime identity verification
- `corpus.json` — frozen list of (op, shape, dtype) tuples (TBD)
- `bench_baseline.py` — runs corpus under default / max_autotune / coord_descent / oracle (TBD)
- `data/` — raw CSV outputs (TBD)
- `HYPOTHESIS_GRAPH.md` — investigation log

## Status

Identity probe **caught a blocker** before any wasted measurement: Triton ships no Windows wheels (any version), and Python 3.14 is unsupported even on Linux Triton wheels (cp313 max). Inductor autotune cannot run in the current Windows + Python 3.14 environment — all three modes degrade to default ATen, making the comparison meaningless.

**Decision (2026-05-09):** move the experiment to WSL2. PyTorch CI is Linux anyway; any future PR would need to land there. The pytorch fork stays at `D:\depot\pytorch` (accessible from WSL via `/mnt/d/depot/pytorch`), but the venv and execution will be Linux-side.

### Mise-en-place revisions for WSL move

- **System under test (revised):** PyTorch Inductor autotuner under Linux + CUDA + Triton. Source path unchanged. Runtime path becomes WSL Ubuntu.
- **Identity probe (revised):** must be re-run inside WSL once Triton is installed. Re-confirm: torch version, CUDA capability visible from WSL (NVIDIA + WSL2 GPU passthrough), Triton version, AlgorithmSelectorCache identity, mode resolution.
- **Experiment repo:** unchanged — `D:\depot\pytorch\benchmarks\abduction\` on branch `abduction-engine`.

### Lesson recorded

Identity probe paid for itself on first run — the labels in the venv (torch installed, CUDA working) hid that the autotune backend couldn't be selected. **Always run identity probe before measurement.** Labels are not identity.

### Next steps (after WSL install)

1. `wsl --install` (admin, reboot)
2. Inside WSL: install uv, create venv with Python 3.13, install torch + triton
3. Re-run `identity_probe.py` from inside WSL — must show triton present and AlgorithmSelectorCache wiring intact
4. Pick & freeze corpus (task #3)
5. Build baseline harness (task #4)
6. Run baseline & commit data (task #5)
