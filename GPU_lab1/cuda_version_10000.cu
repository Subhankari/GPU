#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_profiler_api.h>
#include <time.h>
#include <math.h>

#define SIZE 1000
#define BLKS 4
#define THREADSPBLKS 256
#define TILE_WIDTH 16

__global__
void heatCalcKernel(float * g_d,float * h_d, int width, int itr)
{
  int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
  int col = blockIdx.x * TILE_WIDTH + threadIdx.x;
  int i = (row * width) + col;
  int left = i - 1;
  int right = i + 1;
  int top = ((row - 1) * width) + col;
  int bottom = ((row + 1) * width + col);
 
	if(((i % width) == 0) || ((i % width) == (width - 1)) || (i < width) || (i >= (width * (width - 1)))){
		h_d[i] = g_d[i];
	}else{
		h_d[i] = 0.25 * (g_d[top] + g_d[left] + g_d[bottom] + g_d[right]);
	}
	__syncthreads();
	
	g_d[i] = h_d[i];
	
	__syncthreads();
	
}

void heatCalc()
{
  clock_t tic;
  clock_t toc;
  tic = clock();
  long width = 10000; //32
  int itr = 500;
  long len = width * width;
  float *inhost = (float*)malloc(len*sizeof(float));
  float *outhost =(float*)malloc(len*sizeof(float));

  long j;
  float * g_d;
  float * h_d;
  int counter = 0;

  /*----------------------------------------------------------------*/
  cudaError_t error;
  cudaDeviceProp dev;
  error = cudaGetDeviceProperties(&dev, 0);
     if(error != cudaSuccess)
     {
        printf("Error: %s\n", cudaGetErrorString(error));
        exit(-1);
     }
     printf("\nDevice %d:\n", 0);
     printf("name: %s\n",dev.name);
  
  cudaSetDevice(0);
  /*--------------------------------------------------------------*/
  
  for( j = 0; j < len; j++){
	if((j >= 10) && (j <= 30)){
        inhost[j] = 150;
    }else if((j < width) || ((j % width) == 0) || ((j % width) == (width - 1)) || (j >= (width * (width - 1)))){
        inhost[j] = 80;
    }else{
        inhost[j] = 0;
    }
 }
 
  for( j = 0; j < len; j++){
    outhost[j] = 0;
 }
 
  printf("---------\n");
  
  
  cudaMalloc((void**)&g_d, len*sizeof(float));
  
  //intialize the matrix
 cudaMemcpy(g_d,inhost,len*sizeof(float),cudaMemcpyHostToDevice);
  
  cudaMalloc((void**)&h_d, len*sizeof(float));

  int grid = ceil(width / TILE_WIDTH);
  dim3 dimGrid(grid,grid);
  dim3 dimBlock(TILE_WIDTH,TILE_WIDTH);

  // kernel invocation
 
  for(counter = 0; counter < itr; counter++){
  heatCalcKernel<<<dimGrid,dimBlock>>>(g_d,h_d,width,itr);
  cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) 
    printf("Error: %s\n", cudaGetErrorString(err));
	}

  //transfer C_d from device to host
  cudaMemcpy(outhost, h_d, (len*sizeof(float)), cudaMemcpyDeviceToHost);
   cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) 
    printf("Error: %s\n", cudaGetErrorString(err));
  
  cudaFree(g_d);
  cudaFree(h_d);
 
	for( j = 0; j < len; j++){
		inhost[j] = outhost[j];
	}
	
	toc = clock();
	double time_taken_parallel = (double)(toc -tic)/CLOCKS_PER_SEC; // in seconds
	printf("time taken: %f\n", time_taken_parallel);

}

int main()
{
   heatCalc();
   
    return 0;
}
