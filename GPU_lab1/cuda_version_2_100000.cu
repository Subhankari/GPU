#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_profiler_api.h>
#include <time.h>
#include <math.h>

#define SIZE 1000
#define BLKS 4
#define THREADSPBLKS 256
#define TILE_WIDTH 8

__global__
void heatCalcKernel(float * g_d,float * h_d, int width, int itr,int count)
{
	int blockId = blockIdx.x + blockIdx.y * gridDim.x;
	int i = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;
  int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
  int col = blockIdx.x * TILE_WIDTH + threadIdx.x;
  int id = (row * width) + col;
   row = (i / width);
   col = i % width;
  int left = i - 1;
  int right = i + 1;
  int top = ((row - 1) * width) + col;
  int bottom = (((row + 1) * width)+ col);

	if(((i % width) == 0) || ((i % width) == (width - 1)) || ((i * count) < width) || ((i * count) >= (width * (width - 1)))){
		h_d[id] = g_d[id];
	}else{
		h_d[id] = 0.25 * (g_d[top] + g_d[left] + g_d[bottom] + g_d[right]);
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
  int ori_width = 100000;
  int width = 25000;
  int itr = 50;
  int len = width * width;
  int ori_len = ori_width * ori_width;
  float *inhost = (float*)malloc(ori_len*sizeof(float));
  if(inhost == NULL){
	  printf("Out of memory\n");  
	   exit(-1);  
	}
  float *outhost =(float*)malloc(ori_len*sizeof(float));
    if(outhost == NULL){
	  printf("Out of memory\n");  
	   exit(-1);  
	}

  int j;
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
		if(j == 10){
		printf("%f\n",inhost[j]);
	}
    }else if((j < width) || ((j % width) == 0) || ((j % width) == (width - 1)) || (j >= (width * (width - 1)))){
        inhost[j] = 80;
		if(j == 1){
		printf("%f\n",inhost[j]);
		}
    }else{
        inhost[j] = 0;
    }
    //inhost[j] = j;
 }
 
  for( j = 0; j < len; j++){
    outhost[j] = 0;
 }
  
  printf("---------\n");
  int l = 0;
 for(counter = 0; counter < itr; counter++){
 
  int count = ceil(ori_len / len);
  for(int l = 0; l < count; count++){
	float *inhost1 = (float*)malloc(len*sizeof(float));
	float *outhost1 = (float*)malloc(len*sizeof(float));
	int index = 0;
	if(l == 0){
		for(j = 0; j < len; j++){
			inhost1[index] = inhost[j];
		}
	}else{
		for(j = ((l * len) - width); j < (((l + 1) * len) - width); j++){
			inhost1[index] = inhost[j];
		}
	}
	
	
  cudaMalloc((void**)&g_d, (len*sizeof(float)));
  
  //intialize the matrix
 cudaMemcpy(g_d,inhost1,len*sizeof(float),cudaMemcpyHostToDevice);
  
  cudaMalloc((void**)&h_d, len*sizeof(float));
  
  int grid = ceil(width / TILE_WIDTH);
  dim3 dimGrid(grid,grid);
  dim3 dimBlock(TILE_WIDTH,TILE_WIDTH);

  // kernel invocation
 
  
  heatCalcKernel<<<dimGrid,dimBlock>>>(g_d,h_d,width,itr,(count + 1));
  cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) 
    printf("Error: %s\n", cudaGetErrorString(err));

  //transfer C_d from device to host
  cudaMemcpy(outhost1, h_d, (len*sizeof(float)), cudaMemcpyDeviceToHost);
   err = cudaGetLastError();
  if (err != cudaSuccess) 
    printf("Error: %s\n", cudaGetErrorString(err));
  
  cudaFree(g_d);
  cudaFree(h_d);
  
  if(count == 0){
	for(j = 0; j < len; j++){
		outhost[j] = outhost1[j]; 
	}
  }else{
		int add = (count * len) - width;
		for(j = 0; j < len; j++){
			outhost[j + add] = outhost1[j];
		}
  }
  
  free(inhost1);
  free(outhost1);
  
  }
  for(j = 0; j < (ori_len - width); j++){
	inhost[j] = outhost[j];
	outhost[j] = 0;
	}
  }
	toc = clock();
	double time_taken_parallel = (double)(toc -tic)/CLOCKS_PER_SEC; // in seconds
	printf("time taken: %f\n", time_taken_parallel);

	free(inhost);
	free(outhost);

}

int main()
{
   heatCalc();
   
    return 0;
}
