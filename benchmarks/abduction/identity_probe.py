"""Identity probe — verify what's actually loaded before any measurement.

Prints torch version, CUDA backend, Inductor autotuner class, and which selector
each of the three modes (default / max_autotune / coord_descent) actually picks
at runtime. Labels are not identity. Run before every measurement.
"""
import sys, os, importlib

def section(title):
  print(f"\n{'='*60}\n{title}\n{'='*60}")

section("Python + torch")
print(f"python: {sys.version.split()[0]}")
import torch
print(f"torch: {torch.__version__}")
print(f"torch.__file__: {torch.__file__}")
print(f"cuda available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
  print(f"cuda device: {torch.cuda.get_device_name(0)}")
  print(f"cuda capability: {torch.cuda.get_device_capability(0)}")
  print(f"cuda runtime: {torch.version.cuda}")

section("Inductor + Triton")
import torch._inductor
import torch._inductor.config as ic
print(f"torch._inductor: {torch._inductor.__file__}")
try:
  import triton
  print(f"triton: {triton.__version__}  ({triton.__file__})")
except ImportError as e:
  print(f"triton: NOT INSTALLED ({e})")

section("Autotuner class identity")
from torch._inductor import select_algorithm as sa
print(f"select_algorithm.__file__: {sa.__file__}")
print(f"AlgorithmSelectorCache: {sa.AlgorithmSelectorCache}")
print(f"  module: {sa.AlgorithmSelectorCache.__module__}")
print(f"  qualname: {sa.AlgorithmSelectorCache.__qualname__}")

section("Inductor config defaults (relevant subset)")
for key in ["max_autotune", "max_autotune_gemm", "coordinate_descent_tuning",
            "coordinate_descent_check_all_directions", "search_autotune_cache",
            "benchmark_kernel", "force_disable_caches"]:
  if hasattr(ic, key):
    print(f"  {key} = {getattr(ic, key)!r}")
  else:
    print(f"  {key}: NOT PRESENT in this torch version")

section("Mode resolution (what each env var actually toggles)")
for env_name, cfg_name in [
  ("TORCHINDUCTOR_MAX_AUTOTUNE", "max_autotune"),
  ("TORCHINDUCTOR_MAX_AUTOTUNE_GEMM", "max_autotune_gemm"),
  ("TORCHINDUCTOR_COORDESC_TUNING", "coordinate_descent_tuning"),
  ("TORCHINDUCTOR_BENCHMARK_KERNEL", "benchmark_kernel"),
]:
  env_val = os.environ.get(env_name, "<unset>")
  cfg_val = getattr(ic, cfg_name, "<missing-attr>")
  print(f"  {env_name}={env_val}  ->  config.{cfg_name}={cfg_val!r}")

section("Cache locations")
print(f"  TORCHINDUCTOR_CACHE_DIR = {os.environ.get('TORCHINDUCTOR_CACHE_DIR', '<unset>')}")
try:
  from torch._inductor.codecache import cache_dir
  print(f"  resolved cache_dir(): {cache_dir()}")
except Exception as e:
  print(f"  cache_dir() failed: {e}")

print()
