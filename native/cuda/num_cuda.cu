#include "num_cuda.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusparse.h>
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include <thrust/extrema.h>
#include <cstdio>
#include <cstring>
#include <string>

struct num_cuda_context {
  int device;
  cublasHandle_t blas;
  cusparseHandle_t sparse;
  std::string error;
};

static int fail(num_cuda_context *c, int code, const char *where, int status) {
  if (c) c->error = std::string(where) + " status=" + std::to_string(status);
  return code;
}
#define CUDA(c, x) do { cudaError_t s=(x); if(s!=cudaSuccess) return fail(c,NUM_CUDA_RUNTIME,#x,(int)s); } while(0)
#define BLAS(c, x) do { cublasStatus_t s=(x); if(s!=CUBLAS_STATUS_SUCCESS) return fail(c,NUM_CUDA_CUBLAS,#x,(int)s); } while(0)
#define SPARSE(c, x) do { cusparseStatus_t s=(x); if(s!=CUSPARSE_STATUS_SUCCESS) return fail(c,NUM_CUDA_CUSPARSE,#x,(int)s); } while(0)

extern "C" int num_cuda_create(int device, num_cuda_context **out) {
  if (!out) return NUM_CUDA_INVALID;
  num_cuda_context *c = new num_cuda_context{device, nullptr, nullptr, ""};
  cudaError_t cs = cudaSetDevice(device);
  if (cs != cudaSuccess) { delete c; return NUM_CUDA_RUNTIME; }
  if (cublasCreate(&c->blas) != CUBLAS_STATUS_SUCCESS) { delete c; return NUM_CUDA_CUBLAS; }
  if (cusparseCreate(&c->sparse) != CUSPARSE_STATUS_SUCCESS) { cublasDestroy(c->blas); delete c; return NUM_CUDA_CUSPARSE; }
  *out = c; return NUM_CUDA_OK;
}
extern "C" int num_cuda_destroy(num_cuda_context *c) {
  if (!c) return NUM_CUDA_INVALID;
  cusparseDestroy(c->sparse); cublasDestroy(c->blas); delete c; return NUM_CUDA_OK;
}
extern "C" const char *num_cuda_last_error(num_cuda_context *c) { return c ? c->error.c_str() : "null context"; }
extern "C" int num_cuda_device_name(num_cuda_context *c, char *out, uint64_t capacity) {
  if (!c || !out || !capacity) return NUM_CUDA_INVALID;
  cudaDeviceProp p{}; CUDA(c, cudaGetDeviceProperties(&p,c->device));
  std::snprintf(out,(size_t)capacity,"%s sm_%d%d",p.name,p.major,p.minor); return NUM_CUDA_OK;
}
extern "C" int num_cuda_versions(num_cuda_context *c,int *runtime,int *driver,int *blas,int *sparse) {
  if (!c||!runtime||!driver||!blas||!sparse) return NUM_CUDA_INVALID;
  CUDA(c,cudaRuntimeGetVersion(runtime)); CUDA(c,cudaDriverGetVersion(driver));
  BLAS(c,cublasGetVersion(c->blas,blas)); SPARSE(c,cusparseGetVersion(c->sparse,sparse)); return NUM_CUDA_OK;
}
extern "C" int num_cuda_malloc_f32(num_cuda_context *c,uint64_t n,void **out) {
  if(!c||!out||!n) return NUM_CUDA_INVALID; CUDA(c,cudaMalloc(out,n*sizeof(float))); return NUM_CUDA_OK;
}
extern "C" int num_cuda_free(num_cuda_context *c,void *p) { if(!c||!p)return NUM_CUDA_INVALID; CUDA(c,cudaFree(p)); return NUM_CUDA_OK; }
extern "C" int num_cuda_h2d_f32(num_cuda_context *c,void *d,const float *h,uint64_t n) { if(!c||!d||!h)return NUM_CUDA_INVALID; CUDA(c,cudaMemcpy(d,h,n*4,cudaMemcpyHostToDevice)); return NUM_CUDA_OK; }
extern "C" int num_cuda_d2h_f32(num_cuda_context *c,float *h,const void *d,uint64_t n) { if(!c||!d||!h)return NUM_CUDA_INVALID; CUDA(c,cudaMemcpy(h,d,n*4,cudaMemcpyDeviceToHost)); return NUM_CUDA_OK; }
extern "C" int num_cuda_axpy(num_cuda_context *c,float a,const float*x,float*y,int n) { BLAS(c,cublasSaxpy(c->blas,n,&a,x,1,y,1)); return NUM_CUDA_OK; }
extern "C" int num_cuda_scal(num_cuda_context *c,float a,float*x,int n) { BLAS(c,cublasSscal(c->blas,n,&a,x,1)); return NUM_CUDA_OK; }
extern "C" int num_cuda_dot(num_cuda_context *c,const float*x,const float*y,int n,float*out) { BLAS(c,cublasSdot(c->blas,n,x,1,y,1,out)); return NUM_CUDA_OK; }
extern "C" int num_cuda_nrm2(num_cuda_context *c,const float*x,int n,float*out) { BLAS(c,cublasSnrm2(c->blas,n,x,1,out)); return NUM_CUDA_OK; }

__global__ static void ewise_kernel(int op,const float*x,const float*y,float*z,int n) {
  int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return;
  z[i]=op==0?x[i]+y[i]:op==1?x[i]-y[i]:op==2?x[i]*y[i]:x[i]/y[i];
}
extern "C" int num_cuda_ewise(num_cuda_context*c,int op,const float*x,const float*y,float*z,int n) {
  if(!c||op<0||op>3)return NUM_CUDA_INVALID; ewise_kernel<<<(n+255)/256,256>>>(op,x,y,z,n);
  CUDA(c,cudaGetLastError()); return NUM_CUDA_OK;
}
extern "C" int num_cuda_reduce(num_cuda_context*c,int op,const float*x,int n,float*out) {
  if(!c||!x||!out||n<=0||op<0||op>2)return NUM_CUDA_INVALID;
  try { thrust::device_ptr<const float> p(x);
    *out=op==0?thrust::reduce(p,p+n,0.0f,thrust::plus<float>()):
         op==1?*thrust::max_element(p,p+n):*thrust::min_element(p,p+n);
  } catch(...) { return fail(c,NUM_CUDA_RUNTIME,"thrust reduction",-1); }
  return NUM_CUDA_OK;
}
extern "C" int num_cuda_gemv_row_major(num_cuda_context*c,float a,const float*A,int m,int n,const float*x,float b,float*y) {
  // Row-major A(m,n) is column-major A^T(n,m); transpose it back for y=A*x.
  BLAS(c,cublasSgemv(c->blas,CUBLAS_OP_T,n,m,&a,A,n,x,1,&b,y,1)); return NUM_CUDA_OK;
}
extern "C" int num_cuda_gemm_row_major(num_cuda_context*c,float a,const float*A,int m,int k,const float*B,int n,float b,float*C) {
  // C_row=A_row*B_row => C_col^T=B_col^T*A_col^T; swap operands/dimensions.
  BLAS(c,cublasSgemm(c->blas,CUBLAS_OP_N,CUBLAS_OP_N,n,m,k,&a,B,n,A,k,&b,C,n)); return NUM_CUDA_OK;
}
extern "C" int num_cuda_spmv_csr(num_cuda_context*c,int rows,int cols,int nnz,const int*hrp,const int*hci,const float*hv,const float*x,float*y) {
  int *rp=nullptr,*ci=nullptr; float *v=nullptr; void *workspace=nullptr; size_t bytes=0;
  cusparseSpMatDescr_t mat=nullptr; cusparseDnVecDescr_t vx=nullptr,vy=nullptr; float alpha=1,beta=0;
  CUDA(c,cudaMalloc((void**)&rp,(rows+1)*sizeof(int))); CUDA(c,cudaMalloc((void**)&ci,nnz*sizeof(int))); CUDA(c,cudaMalloc((void**)&v,nnz*sizeof(float)));
  CUDA(c,cudaMemcpy(rp,hrp,(rows+1)*sizeof(int),cudaMemcpyHostToDevice)); CUDA(c,cudaMemcpy(ci,hci,nnz*sizeof(int),cudaMemcpyHostToDevice)); CUDA(c,cudaMemcpy(v,hv,nnz*sizeof(float),cudaMemcpyHostToDevice));
  SPARSE(c,cusparseCreateCsr(&mat,rows,cols,nnz,rp,ci,v,CUSPARSE_INDEX_32I,CUSPARSE_INDEX_32I,CUSPARSE_INDEX_BASE_ZERO,CUDA_R_32F));
  SPARSE(c,cusparseCreateDnVec(&vx,cols,(void*)x,CUDA_R_32F)); SPARSE(c,cusparseCreateDnVec(&vy,rows,y,CUDA_R_32F));
  SPARSE(c,cusparseSpMV_bufferSize(c->sparse,CUSPARSE_OPERATION_NON_TRANSPOSE,&alpha,mat,vx,&beta,vy,CUDA_R_32F,CUSPARSE_SPMV_ALG_DEFAULT,&bytes));
  CUDA(c,cudaMalloc(&workspace,bytes)); SPARSE(c,cusparseSpMV(c->sparse,CUSPARSE_OPERATION_NON_TRANSPOSE,&alpha,mat,vx,&beta,vy,CUDA_R_32F,CUSPARSE_SPMV_ALG_DEFAULT,workspace));
  cudaFree(workspace); cusparseDestroyDnVec(vy); cusparseDestroyDnVec(vx); cusparseDestroySpMat(mat); cudaFree(v); cudaFree(ci); cudaFree(rp); return NUM_CUDA_OK;
}
