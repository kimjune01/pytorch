# Stale `torch/headeronly/version.h` causes shim_common.cpp build failure

## Summary

`torch/headeronly/version.h` is generated from `version.h.in` by `tools/setup_helpers/gen_version_header.py` using `version.txt` as the source of truth. When `version.h` already exists in the working tree (from a previous build, a different branch, or a partial build), CMake / setup.py do NOT regenerate it. The stale header carries an incorrect version number, which propagates into `TORCH_FEATURE_VERSION`, which trips a `static_assert` in `torch/csrc/stable/stableivalue_conversions.h:69` when compiling `torch/csrc/shim_common.cpp` for any feature gated on `TORCH_VERSION_2_10_0+` (e.g. `std::string` support).

## Reproducer

Setup: NVIDIA pytorch dev container `nvcr.io/nvidia/pytorch:25.10-py3` (Ubuntu 24.04, GCC 13, CUDA 13.0), source on ext4 (not slow 9p mount).

Steps:
```bash
git clone https://github.com/pytorch/pytorch.git
cd pytorch
git checkout v2.12.0-rc9   # or any tag/branch where version.txt > 2.9
git submodule update --init --recursive

# Simulate a stale version.h (this happens naturally when you switch branches,
# do an incremental build, or copy from a prior build attempt):
cat > torch/headeronly/version.h <<'EOF'
#pragma once
#define TORCH_VERSION_MAJOR 2
#define TORCH_VERSION_MINOR 9
#define TORCH_VERSION_PATCH 0
#define TORCH_VERSION_ABI_TAG 0
#define TORCH_VERSION "2.9.0"
#define TORCH_ABI_VERSION ( \
  ((0ULL + TORCH_VERSION_MAJOR) << 56) | \
  ((0ULL + TORCH_VERSION_MINOR) << 48) | \
  ((0ULL + TORCH_VERSION_PATCH) << 40) | \
  ((0ULL + TORCH_VERSION_ABI_TAG) << 0))
EOF

# Build:
USE_CUDA=1 USE_DISTRIBUTED=0 USE_KINETO=0 BUILD_TEST=0 USE_FBGEMM=0 \
  pip install -e . --no-build-isolation -v
```

## Failure

```
torch/csrc/stable/stableivalue_conversions.h:69:15: error: static assertion failed:
    std::string requires TORCH_FEATURE_VERSION >= TORCH_VERSION_2_10_0
   69 |     !std::is_same_v<T, std::string>,
      |      ~~~~~^~~~~~~~~~~~~~~~~~~~~~~~~

torch/csrc/shim_common.cpp:155:42: required from here
```

Fully ~50 errors cascade from this in `shim_common.cpp` — `StableListHandle`, `StringHandle`, `ParallelFunc` etc. all "not declared in this scope" because the relevant overloads are gated on `TORCH_FEATURE_VERSION >= TORCH_VERSION_2_10_0`.

## Mechanism

1. `version.h.in` declares `TORCH_VERSION_MAJOR/MINOR/PATCH` as `@TORCH_VERSION_*@` template variables.
2. `tools/setup_helpers/gen_version_header.py` reads `version.txt` and substitutes the values.
3. `torch/csrc/stable/version.h:24-26` defines `TORCH_FEATURE_VERSION` as either `TORCH_TARGET_VERSION` (if defined by extension authors) or `TORCH_ABI_VERSION` (computed from `version.h`).
4. `torch/csrc/stable/stableivalue_conversions.h:69` static-asserts `TORCH_FEATURE_VERSION >= TORCH_VERSION_2_10_0` for `std::string`.
5. `torch/csrc/shim_common.cpp:155` instantiates the template with `std::string` (in `from_ivalue` for `c10::TypeKind::StringType`).

If `version.h` is stale at MINOR=9 but `version.txt` says 2.12, `TORCH_FEATURE_VERSION` resolves to `0x0209_0000_0000_0000` < `TORCH_VERSION_2_10_0 = 0x020A_0000_0000_0000` → static_assert fires.

The `gen_version_header.py` script DOES exist and DOES produce correct output when run manually:
```bash
python3 tools/setup_helpers/gen_version_header.py \
    --template-path torch/headeronly/version.h.in \
    --version-path version.txt \
    --output-path torch/headeronly/version.h
# produces correct MINOR=12 for v2.12.0-rc9
```

## Workaround (user-side)

```bash
rm torch/headeronly/version.h
python3 tools/setup_helpers/gen_version_header.py \
    --template-path torch/headeronly/version.h.in \
    --version-path version.txt \
    --output-path torch/headeronly/version.h
# Then build proceeds cleanly.
```

## Suggested fix

Either:
1. **Always regenerate `version.h` at build start** — add `version.h` as a target in `setup.py` / cmake that always re-runs (depends on `version.txt` mtime), so a stale on-disk copy doesn't block.
2. **Generate `version.h` to the build dir, not the source dir** — `torch/headeronly/version.h` should not be in the source tree at all; cmake can place it under `build/include/` and add `-Ibuild/include` to the compile flags. Same pattern other CMake projects use for generated headers.
3. **Make stable-shim source code self-contained** — the static_assert is meant to gate EXTENSION code that opts into specific stable ABI versions, not pytorch's own internal compile of shim_common.cpp. Either define `TORCH_TARGET_VERSION` to current version inside `shim_common.cpp`, or weaken the assert when compiling pytorch itself.

Option 1 is the smallest fix; Option 2 is the cleanest long-term; Option 3 fixes the root semantic mismatch (the shim implementation file shouldn't trip the shim consumer's compile-time guard).

## Environment

- Ubuntu 24.04 (in `nvcr.io/nvidia/pytorch:25.10-py3`)
- GCC 13.3
- CUDA 13.0.88
- Building v2.12.0-rc9 + 3 user commits (commits don't touch any version-related files)
