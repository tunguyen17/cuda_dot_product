#include <stdio.h>
#include <stdlib.h>

const int shared_size = 4;
// Square matrix multiplication with dimention that has power of 8

__global__ void kernel(float *a, float *b, float *c, int i, int j, int n){

  // const int shared_size = n;
  __shared__ float cache[shared_size];

  int idx = threadIdx.x;

  if(idx < n){
    cache[idx] = a[idx] * b[idx];
  }
  __syncthreads();

  // blockDim has to be a power of 2
  int iter = blockDim.x/2;
  while( iter!= 0){
    cache[idx] += cache[idx + iter];
    __syncthreads();
    iter/=2;
  }

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
  int n = shared_size;
  int matDim = n*n;
  // dim3 grid(n, n);

  float *a_host, *b_host, *c_host;
  a_host = (float*) malloc(matDim*sizeof(float));
  b_host = (float*) malloc(matDim*sizeof(float));
  c_host = (float*) malloc(matDim*sizeof(float));

  // Two temp vector
  float *u_host, *v_host;
  u_host = (float*) malloc(n*sizeof(float));
  v_host = (float*) malloc(n*sizeof(float));

  float *u_dev, *v_dev, *c_dev;
  cudaMalloc((float**) &u_dev, n*sizeof(float));
  cudaMalloc((float**) &v_dev, n*sizeof(float));
  cudaMalloc((float**) &c_dev, matDim*sizeof(float));

  for(int i = 0; i<matDim; i++){
    a_host[i] = i+1;
    b_host[i] = 1;
  }

  // i row
  // j col
  for(int i = 0; i < n; i++){
    for(int j = 0; j < n; j++){
        // v_host[k] = a_host[j + (i+k)*n];
      // printf("%d - %d ---*\n", i, j);
      for(int k = 0; k < n; k++){
        u_host[k] = a_host[i*n + k];
        v_host[k] = b_host[j + k*n];
      }
      // print_vec(u_host, n);
      // print_vec(v_host, n);
      // printf("---*\n");
      cudaMemcpy(u_dev, u_host, n*sizeof(float), cudaMemcpyHostToDevice);
      cudaMemcpy(v_dev, v_host, n*sizeof(float), cudaMemcpyHostToDevice);
      kernel<<<1, n>>>(u_dev, v_dev, c_dev, i, j, n);
    }
  }

  cudaMemcpy(c_host, c_dev, matDim*sizeof(float), cudaMemcpyDeviceToHost);

  // printf("Hello");

  // for(int i = 0; i<n; i++){
    // printf("%.3f -- %.3f -- %.3f\n", a_host[i], b_host[i], c_host[i]);
  // }

  printf("----\n");
  print_mat(c_host, n);

  //Free the mem allocation
  free(a_host);
  free(b_host);
  free(c_host);
  cudaFree(u_dev);
  cudaFree(v_dev);
  cudaFree(c_dev);
}
