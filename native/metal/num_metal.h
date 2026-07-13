#pragma once
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
typedef struct num_metal_context num_metal_context;
typedef struct num_metal_kernel num_metal_kernel;
enum { NUM_METAL_OK=0, NUM_METAL_INVALID=1, NUM_METAL_RUNTIME=2, NUM_METAL_COMPILE=3 };
int num_metal_create(num_metal_context **out); int num_metal_destroy(num_metal_context *c);
const char *num_metal_last_error(num_metal_context *c);
int num_metal_device_name(num_metal_context*c,char*out,uint64_t capacity);
int num_metal_malloc_f32(num_metal_context*c,uint64_t n,void**out);int num_metal_free(num_metal_context*c,void*p);
int num_metal_h2d_f32(num_metal_context*c,void*p,const float*x,uint64_t n);int num_metal_d2h_f32(num_metal_context*c,float*x,void*p,uint64_t n);
int num_metal_axpy(num_metal_context*c,float a,void*x,void*y,uint32_t n);int num_metal_scal(num_metal_context*c,float a,void*x,uint32_t n);
int num_metal_dot(num_metal_context*c,void*x,void*y,uint32_t n,float*out);int num_metal_nrm2(num_metal_context*c,void*x,uint32_t n,float*out);
int num_metal_ewise(num_metal_context*c,int op,void*x,void*y,void*z,uint32_t n);int num_metal_reduce(num_metal_context*c,int op,void*x,uint32_t n,float*out);
int num_metal_gemv(num_metal_context*c,float a,void*A,uint32_t m,uint32_t n,void*x,float b,void*y);
int num_metal_gemm(num_metal_context*c,float a,void*A,uint32_t m,uint32_t k,void*B,uint32_t n,float b,void*C);
int num_metal_spmv(num_metal_context*c,uint32_t rows,uint32_t cols,uint32_t nnz,const int*rp,const int*ci,const float*v,void*x,void*y);
int num_metal_compile(num_metal_context*c,const char*source,const char*name,num_metal_kernel**out);int num_metal_kernel_destroy(num_metal_context*c,num_metal_kernel*k);
int num_metal_launch_ewise(num_metal_context*c,num_metal_kernel*k,void*x,void*y,void*z,uint32_t n,uint32_t wg);
int num_metal_launch_reduce(num_metal_context*c,num_metal_kernel*k,int op,void*x,uint32_t n,uint32_t wg,float*out);
#ifdef __cplusplus
}
#endif
