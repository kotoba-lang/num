#!/usr/bin/env bash
set -euo pipefail
command -v nvidia-smi >/dev/null || { echo "CUDA verification requires nvidia-smi" >&2; exit 2; }
command -v nvcc >/dev/null || { echo "CUDA verification requires nvcc" >&2; exit 2; }
nvidia-smi --query-gpu=name,compute_cap,driver_version,memory.total --format=csv,noheader
cmake -S native/cuda -B target/cuda -DCMAKE_BUILD_TYPE=Release
cmake --build target/cuda --parallel
./target/cuda/num_cuda_smoke
LD_LIBRARY_PATH="$PWD/target/cuda${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" clojure -M:cuda-verify
