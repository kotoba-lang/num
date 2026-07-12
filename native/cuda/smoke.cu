#include "num_cuda.h"
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <vector>

static void check(num_cuda_context *c,int code,const char *op) {
  if(code!=NUM_CUDA_OK){ std::fprintf(stderr,"%s: %s\n",op,num_cuda_last_error(c)); std::exit(code); }
}
int main(){
  num_cuda_context *c=nullptr; check(c,num_cuda_create(0,&c),"create");
  char name[256]; int runtime,driver,blas,sparse;
  check(c,num_cuda_device_name(c,name,sizeof name),"device");
  check(c,num_cuda_versions(c,&runtime,&driver,&blas,&sparse),"versions");
  std::printf("GPU=%s runtime=%d driver=%d cublas=%d cusparse=%d\n",name,runtime,driver,blas,sparse);
  std::vector<float> hx{1,2,3,4},hy{10,20,30,40},out(4); void *x,*y,*z;
  check(c,num_cuda_malloc_f32(c,4,&x),"malloc x"); check(c,num_cuda_malloc_f32(c,4,&y),"malloc y"); check(c,num_cuda_malloc_f32(c,4,&z),"malloc z");
  check(c,num_cuda_h2d_f32(c,x,hx.data(),4),"h2d x"); check(c,num_cuda_h2d_f32(c,y,hy.data(),4),"h2d y");
  float dot=0,norm=0,sum=0; check(c,num_cuda_dot(c,(float*)x,(float*)y,4,&dot),"dot");
  check(c,num_cuda_nrm2(c,(float*)x,4,&norm),"nrm2"); check(c,num_cuda_reduce(c,0,(float*)x,4,&sum),"sum");
  check(c,num_cuda_ewise(c,0,(float*)x,(float*)y,(float*)z,4),"ewise"); check(c,num_cuda_d2h_f32(c,out.data(),z,4),"d2h");
  bool ok=std::fabs(dot-300)<1e-4&&std::fabs(norm-std::sqrt(30.0f))<1e-4&&std::fabs(sum-10)<1e-4&&out==std::vector<float>({11,22,33,44});
  float Ahost[]={1,2,3,4},Bhost[]={5,6,7,8},Chost[]={0,0,0,0}; void *A,*B,*C;
  check(c,num_cuda_malloc_f32(c,4,&A),"malloc A");check(c,num_cuda_malloc_f32(c,4,&B),"malloc B");check(c,num_cuda_malloc_f32(c,4,&C),"malloc C");
  check(c,num_cuda_h2d_f32(c,A,Ahost,4),"h2d A");check(c,num_cuda_h2d_f32(c,B,Bhost,4),"h2d B");check(c,num_cuda_h2d_f32(c,C,Chost,4),"h2d C");
  check(c,num_cuda_gemm_row_major(c,1,(float*)A,2,2,(float*)B,2,0,(float*)C),"gemm");check(c,num_cuda_d2h_f32(c,out.data(),C,4),"gemm d2h");
  ok=ok&&out==std::vector<float>({19,22,43,50});
  int rp[]={0,2,3},ci[]={0,2,1};float vv[]={1,2,3},x3h[]={1,1,1};void *x3,*sy;
  check(c,num_cuda_malloc_f32(c,3,&x3),"malloc x3");check(c,num_cuda_malloc_f32(c,2,&sy),"malloc sy");check(c,num_cuda_h2d_f32(c,x3,x3h,3),"h2d x3");
  check(c,num_cuda_spmv_csr(c,2,3,3,rp,ci,vv,(float*)x3,(float*)sy),"spmv");out.resize(2);check(c,num_cuda_d2h_f32(c,out.data(),sy,2),"spmv d2h");ok=ok&&out==std::vector<float>({3,3});
  for(void*p:{x,y,z,A,B,C,x3,sy})check(c,num_cuda_free(c,p),"free");check(c,num_cuda_destroy(c),"destroy");
  std::puts(ok?"CUDA native smoke: PASS":"CUDA native smoke: FAIL");return ok?0:1;
}
