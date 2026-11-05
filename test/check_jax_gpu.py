"""Show whether this Python environment can use a GPU with JAX."""

import os
import subprocess
import sys

import jax
import jaxlib


print("Python:", sys.executable)
print("JAX:", jax.__version__)
print("jaxlib:", jaxlib.__version__)
print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>"))
print("JAX_PLATFORMS:", os.environ.get("JAX_PLATFORMS", "<unset>"))

try:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
         "--format=csv,noheader"],
        capture_output=True,
        text=True,
        check=False,
    )
    print("nvidia-smi:", result.stdout.strip() if result.returncode == 0
          else result.stderr.strip())
except FileNotFoundError:
    print("nvidia-smi: unavailable")

try:
    print("JAX default backend:", jax.default_backend())
    print("JAX default devices:", jax.devices())
    print("JAX GPU devices:", jax.devices("gpu"))
except RuntimeError as error:
    print("JAX GPU backend unavailable:", error)
