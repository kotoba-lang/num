# Native CUDA bridge

This optional shared library implements `num.cuda/ICudaDriver`'s native side
with CUDA Runtime, cuBLAS and cuSPARSE. It is not loaded by the portable core.

Requirements: Linux x86-64, NVIDIA driver, CUDA Toolkit 12, CMake 3.24+.

```sh
cmake -S native/cuda -B target/cuda -DCMAKE_BUILD_TYPE=Release
cmake --build target/cuda -j
./target/cuda/num_cuda_smoke
LD_LIBRARY_PATH="$PWD/target/cuda" clojure -M:cuda-verify
```

The smoke executable exercises device provenance, allocation/copies, dot,
norm, reduction, elementwise addition, row-major GEMM and CSR SpMV. Production
qualification then runs all 14 `num.contract` operations through the JNA JVM
host binding and the same native library.

During JVM backend creation the JNA adapter passes the seven compiler-sealed
CUDA sources to NVRTC. `libnum_cuda` loads the resulting PTX with the CUDA
Driver API and returns opaque kernel handles; ewise/reduction calls launch those
handles. Kernel modules are unloaded before the CUDA context is destroyed.

No CPU fallback exists inside this library. Every exported function returns a
stable status and stores native diagnostics in the context. Dense storage is
row-major at the public ABI; the bridge applies the cuBLAS operand convention.
