#include <stdio.h>
#include <stdlib.h>

const int threadsPerBlock = 4;
// Square matrix multiplication with dimention that has power of 8

__global__ void kernel(float *a, float *b, float *c, int i, int n){

  // // const int shared_size = n;
  __shared__ float cache[threadsPerBlock];
  
  int k = threadIdx.x;
  int j = blockIdx.x;
  //
  int a_idx = i*n+k;
  int b_idx = j+k*n;
  //
  if(a_idx < n*n){
    cache[k] = a[a_idx] * b[b_idx];
  }
  __syncthreads();
  //
  // // blockDim has to be a power of 2
  int iter = blockDim.x/2;
  while( iter!= 0){
    cache[k] += cache[k + iter];
    __syncthreads();
    iter/=2;
  }

  // c[j+n*i] = a[i];
  c[j+n*i] = cache[0];
}

void print_mat(float *a, int n){
  for(int j = 0; j < n; j++){
    for(int i = 0; i < n; i++){
      printf("%.3f\t", a[i+n*j]);
    }
    printf("\n");
  }
}

void print_vec(float *a, int n){
  for(int i = 0; i < n; i++){
    printf("%.3f\t", a[i]);
  }
  printf("\n");
}

int main(int argc, char** argv){
  int n = threadsPerBlock;
  int matDim = n*n;
  // dim3 grid(n, n);

  float *a_host, *b_host, *c_host;
  a_host = (float*) malloc(matDim*sizeof(float));
  b_host = (float*) malloc(matDim*sizeof(float));
  c_host = (float*) malloc(matDim*sizeof(float));

  float *a_dev, *b_dev, *c_dev;
  cudaMalloc((float**) &a_dev, matDim*sizeof(float));
  cudaMalloc((float**) &b_dev, matDim*sizeof(float));
  cudaMalloc((float**) &c_dev, matDim*sizeof(float));

  for(int i = 0; i<matDim; i++){
    a_host[i] = i+1;
    b_host[i] = i;
  }


  cudaMemcpy(a_dev, a_host, matDim*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(b_dev, b_host, matDim*sizeof(float), cudaMemcpyHostToDevice);

  // i row
  // j col
  for(int i = 0; i < n; i++){
      kernel<<<n, n>>>(a_dev, b_dev, c_dev, i, n);
  }

  cudaMemcpy(c_host, c_dev, matDim*sizeof(float), cudaMemcpyDeviceToHost);

  printf("----\n");
  print_mat(c_host, n);

  //Free the mem allocation
  free(a_host);
  free(b_host);
  free(c_host);
  cudaFree(a_dev);
  cudaFree(b_dev);
  cudaFree(c_dev);
}
