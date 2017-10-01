#include <stdio.h>
#include <stdlib.h>

/*
  This program will calculate the dot product of two square matrices with dimention as a power of 2
*/

// Declare the size for the shared memory
// It's important to note that hte size of the shared memory must equal to the size of the dimension of the matrix
// For example if the matrix is n by n, the shared size must be n
// Need to fix this so that I don't have to worry about declare threadsPerBlock at the begining of the program
const int threadsPerBlock = 10;

__global__ void kernel(float *a, float *b, float *c, int i, int n){
  // Declare the shared memory cache | this is shared memory is used to calculate the entries
  __shared__ float cache[threadsPerBlock];

  // declare the variables that hold the position
  int k = threadIdx.x;
  int j = blockIdx.x;

  // calculate the corresponding index for a and b
  int a_idx = i*n+k;
  int b_idx = j+k*n;

  // calculate the entries multiplication
  // the if statement is used to prevent reading and writing indexices that are not in the scope
  // might need to take care of k too
  if(a_idx < n*n){
    cache[k] = a[a_idx] * b[b_idx];
  }
  // sync threads so that all the thread will finish the multiplication cacluation before moving into addition
  __syncthreads();

  // Caclulate the sum of all the multiplication
  // Note that the blockDim has to be a power of 2
  int iter = blockDim.x/2;
  while( iter!= 0){
    cache[k] += cache[k + iter];
    __syncthreads();
    iter/=2;
  }

  // storing the calculation of row i column j to c
  c[j+n*i] = cache[0];
}


// Method to print the matrix
void print_mat(float *a, int n){
  for(int j = 0; j < n; j++){
    for(int i = 0; i < n; i++){
      printf("%.3f\t", a[i+n*j]);
    }
    printf("\n");
  }
}

int main(int argc, char** argv){

  // Declare the matrix dimention
  int n = threadsPerBlock;
  // Caclulate the array length required size of for the storing of the matrix
  int arr_length = n*n;

  // Declaring and allocate memory for the host variables
  float *a_host, *b_host, *c_host;
  a_host = (float*) malloc(arr_length*sizeof(float));
  b_host = (float*) malloc(arr_length*sizeof(float));
  c_host = (float*) malloc(arr_length*sizeof(float));

  // Declaring and allocate memory for the device variables
  float *a_dev, *b_dev, *c_dev;
  cudaMalloc((float**) &a_dev, arr_length*sizeof(float));
  cudaMalloc((float**) &b_dev, arr_length*sizeof(float));
  cudaMalloc((float**) &c_dev, arr_length*sizeof(float));

  // Generate values for the matricesi
  for(int i = 0; i<arr_length; i++){
    a_host[i] = 1;
    b_host[i] = 1;
  }

  // Copy memory form host to device
  cudaMemcpy(a_dev, a_host, arr_length*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(b_dev, b_host, arr_length*sizeof(float), cudaMemcpyHostToDevice);

  /* Note:
    - i row
    - j column
  */
  for(int i = 0; i < n; i++){
      // each kernel will calculate the resulting entries for row i
      /*
        Each block will caclulate each entry j for row i
          - the tread is calculating the multiplication part
          - the shared memory is used to store and sum up the products
      */
      kernel<<<n, n>>>(a_dev, b_dev, c_dev, i, n);
  }

  // copy memory from device to host
  cudaMemcpy(c_host, c_dev, arr_length*sizeof(float), cudaMemcpyDeviceToHost);

  // print out the result
  printf("----\n");
  print_mat(c_host, n);

  //Free the memory allocation
  free(a_host);
  free(b_host);
  free(c_host);
  cudaFree(a_dev);
  cudaFree(b_dev);
  cudaFree(c_dev);
}
