#pragma once
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

typedef struct num_cuda_context num_cuda_context;
typedef struct num_cuda_kernel num_cuda_kernel;
enum { NUM_CUDA_OK = 0, NUM_CUDA_INVALID = 1, NUM_CUDA_RUNTIME = 2,
       NUM_CUDA_CUBLAS = 3, NUM_CUDA_CUSPARSE = 4 };

int num_cuda_create(int device, num_cuda_context **out);
int num_cuda_destroy(num_cuda_context *ctx);
const char *num_cuda_last_error(num_cuda_context *ctx);
int num_cuda_device_name(num_cuda_context *ctx, char *out, uint64_t capacity);
int num_cuda_versions(num_cuda_context *ctx, int *runtime, int *driver, int *cublas, int *cusparse);
int num_cuda_malloc_f32(num_cuda_context *ctx, uint64_t n, void **out);
int num_cuda_free(num_cuda_context *ctx, void *ptr);
int num_cuda_h2d_f32(num_cuda_context *ctx, void *dst, const float *src, uint64_t n);
int num_cuda_d2h_f32(num_cuda_context *ctx, float *dst, const void *src, uint64_t n);
int num_cuda_axpy(num_cuda_context *ctx, float alpha, const float *x, float *y, int n);
int num_cuda_scal(num_cuda_context *ctx, float alpha, float *x, int n);
int num_cuda_dot(num_cuda_context *ctx, const float *x, const float *y, int n, float *out);
int num_cuda_nrm2(num_cuda_context *ctx, const float *x, int n, float *out);
int num_cuda_ewise(num_cuda_context *ctx, int op, const float *x, const float *y, float *z, int n);
int num_cuda_reduce(num_cuda_context *ctx, int op, const float *x, int n, float *out);
int num_cuda_compile_kernel(num_cuda_context *ctx, const char *source, const char *kernel_name,
                            num_cuda_kernel **out);
int num_cuda_kernel_destroy(num_cuda_context *ctx, num_cuda_kernel *kernel);
int num_cuda_launch_ewise(num_cuda_context *ctx, num_cuda_kernel *kernel,
                          const float *x, const float *y, float *z, uint32_t n,
                          uint32_t workgroup_size);
int num_cuda_launch_reduce(num_cuda_context *ctx, num_cuda_kernel *kernel,
                           const float *x, float *parts, uint32_t n,
                           uint32_t workgroup_size);
int num_cuda_gemv_row_major(num_cuda_context *ctx, float alpha, const float *A, int m, int n,
                            const float *x, float beta, float *y);
int num_cuda_gemm_row_major(num_cuda_context *ctx, float alpha, const float *A, int m, int k,
                            const float *B, int n, float beta, float *C);
int num_cuda_spmv_csr(num_cuda_context *ctx, int rows, int cols, int nnz,
                      const int *row_ptr, const int *col_idx, const float *values,
                      const float *x, float *y);
#ifdef __cplusplus
}
#endif
