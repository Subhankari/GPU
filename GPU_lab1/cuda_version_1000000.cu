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
void heatCalcKernel(float * g_d,float * h_d, int width, int itr, int new_len, int new_width)
{
  int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
  int col = blockIdx.x * TILE_WIDTH + threadIdx.x;
  int id = (row * new_width) + col;
   row = (id / width);
   col = id % width;
  int left = id - 1;
  int right = id + 1;
  int top = ((row - 1) * width) + col;
  int bottom = ((row + 1) * width + col);
  int tmp = 2;
  
  if( id < (width * (itr + 1))){
		if(((id % width) == 0) || ((id % width) == (width - 1)) || (id < width) || (id >= (width * (width - 1)))){
			h_d[id] = g_d[id];
		}else{
			h_d[id] = 0.25 * (g_d[top] + g_d[left] + g_d[bottom] + g_d[right]);
		}
	}else if ( id < (new_len - (width * (itr + 1)))){
		if(((id % width) == 0) || ((id % (2 * itr)) == 0) || ((id % (tmp * itr)) == ((tmp * itr) - 1))){
			h_d[id] = g_d[id];
		}else{
			h_d[id] = 0.25 * (g_d[id -1] + g_d[id + 1] + g_d[id - itr - tmp] + g_d[id + itr + tmp]);
		
		}
	}else{
		if((((id - (new_len - (width * (itr + tmp)))) %  width) == 0) || (((id - (new_len - (width * (itr + 1)))) % width) == (width - 1)) || ((id - (new_len - (width * (itr + 1)))) >= (width * (itr - 1)))){
			h_d[id] = g_d[id];
		
		} else{
			h_d[id] = 0.25 * (g_d[id -1] + g_d[id + 1] + g_d[id - width] + g_d[id + width]);
		}
	
	}

	__syncthreads();
	
	g_d[id] = h_d[id];
	
	__syncthreads();
	
}

void heatCalc()
{
  clock_t tic;
  clock_t toc;
  tic = clock();
  int width = 1000000;
  int len = width * width;
  int itr = 50;
  int remove = (width - itr - 1) * (width - itr - 1);
  int new_len = len - remove;
  float *inhost = (float*)malloc(len*sizeof(float));
  if(inhost == NULL){
	  printf("Out of memory\n");  
	   exit(-1);  
	}
 float *inhost1 = (float*)malloc(new_len*sizeof(float));
 if(inhost1 == NULL){
	  printf("Out of memory\n");  
	   exit(-1);  
	}
 float *outhost = (float*)malloc(new_len*sizeof(float));
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
  int count = 0;
  for( j = 0; j < len; j++){
	if((j >= 10) && (j <= 30)){
	
        inhost[j] = 150;
		inhost1[count] = inhost[j];
    }else if((j < width) || ((j % width) == 0) || ((j % width) == (width - 1)) || (j >= (width * (width - 1)))){
        inhost[j] = 80;
		inhost1[count] = inhost[j];
    }else{
        inhost[j] = 0;
		if (((j % width) < (itr + 1)) || ((j % width) > (width -(itr + 1)))){
			inhost1[count] = inhost[j];
		}
    }
	count++;
 }
 
 free(inhost);
   
  printf("---------\n");
 
  
  cudaMalloc((void**)&g_d, new_len*sizeof(float));
  
  //intialize the matrix
 cudaMemcpy(g_d,inhost1,new_len*sizeof(float),cudaMemcpyHostToDevice);
  
  cudaMalloc((void**)&h_d, new_len*sizeof(float));
  int new_width = ceil(sqrt(new_len));
  int grid = ceil(new_width / TILE_WIDTH);
  dim3 dimGrid(grid,grid);
  dim3 dimBlock(TILE_WIDTH,TILE_WIDTH);

  // kernel invocation
 
  for(counter = 0; counter < itr; counter++){
  heatCalcKernel<<<dimGrid,dimBlock>>>(g_d,h_d,width,itr,new_len,new_width);
  cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) 
    printf("Error: %s\n", cudaGetErrorString(err));
	
	}

  //transfer C_d from device to host
  cudaMemcpy(outhost, h_d, (new_len*sizeof(float)), cudaMemcpyDeviceToHost);
   cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) 
    printf("Error: %s\n", cudaGetErrorString(err));
  
  cudaFree(g_d);
  cudaFree(h_d);
  
  for( j = 0; j < new_len; j++){
		inhost1[j] = outhost[j];
	}
	
	toc = clock();
	double time_taken_parallel = (double)(toc -tic)/CLOCKS_PER_SEC; // in seconds
	printf("time taken: %f\n", time_taken_parallel);
	free(inhost1);
	free(outhost);

}

int main()
{
   heatCalc();
   
    return 0;
}
