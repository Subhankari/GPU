#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_profiler_api.h>
#include <math.h>
#include <curand_kernel.h>
#include <time.h>
#include <string.h>

int sudoku[81];
int state[81];
int len = 81;


__constant__ int mstate_d[81];

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

__device__ int compute_score_d(int puzzle[9][9]){
	
	int unique_count = 0;
	for(int i = 0; i < 9; i++){
		int count_row[10] = {0,0,0,0,0,0,0,0,0,0};
		int count_col[10] = {0,0,0,0,0,0,0,0,0,0};
		for(int j = 0; j < 9; j++){
			count_row[puzzle[i][j]] += 1;
			count_col[puzzle[j][i]] += 1;
		}
		for(int j = 0; j < 10; j++){
			if(count_row[j] > 0){
				unique_count++;
			}
			if(count_col[j] > 0){
				unique_count++;
			}
		}	
	}
	return (162 - unique_count);
}

int compute_score_h(int puzzle[81]){
	
	int unique_count = 0;
	for(int i = 0; i < 9; i++){
		int count_row[10] = {0,0,0,0,0,0,0,0,0,0};
		int count_col[10] = {0,0,0,0,0,0,0,0,0,0};
		for(int j = 0; j < 9; j++){
			count_row[puzzle[(i * 9) + j]] += 1;
			count_col[puzzle[(j * 9) + i]] += 1;
		}
		for(int j = 0; j < 10; j++){
			if(count_row[j] > 0){
				unique_count++;
			}
			if(count_col[j] > 0){
				unique_count++;
			}
		}	
	}
	return (162 - unique_count);
}

__global__ void initCurand(curandState *rstate_b1,curandState *rstate_b2,curandState *rstate_3, unsigned long seed1,unsigned long seed2,unsigned long seed3){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    curand_init(seed1, idx , 0, &rstate_b1[idx]);
    curand_init(seed2, idx , 0, &rstate_b2[idx]);
    curand_init(seed3, idx , 0, &rstate_3[idx]);

}


__global__ void init_rsudoku(curandState *rstate_b1, curandState *rstate_b2, int * sudoku_db1, int * sudoku_db2){
 
	__shared__ int shared_puzzle[9][9];
	
	int thread_x = threadIdx.x;
	int thread_y = threadIdx.y;
	int thread_block_id = threadIdx.x*blockDim.x + threadIdx.y;
	int block_num = blockIdx.x * blockDim.x + blockIdx.y;
	int blockId = blockIdx.x + blockIdx.y * gridDim.x;
	int threadId = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;
	
	shared_puzzle[thread_x][thread_y] = sudoku_db1[(thread_x * 9) + thread_y];
	
	if(thread_block_id == 0){
		int block_x;
		int block_y;
		int x1, y1, x2, y2;
		int temp;
		int new_score = 1000;
		int current_score=compute_score_d(shared_puzzle);
			if(block_num == 0){
		
				block_x = 3*(int)(3.0*curand_uniform(&rstate_b1[block_num]));
				block_y = 3*(int)(3.0*curand_uniform(&rstate_b1[block_num]));
				do
				{
					x1=(int)3.0*curand_uniform(&rstate_b1[block_num]);
					y1=(int)3.0*curand_uniform(&rstate_b1[block_num]);

				}while(mstate_d[((block_x+x1) * 9 )+(block_y+y1)]==1);


				do{
					x2=(int)3.0*curand_uniform(&rstate_b1[block_num]);
					y2=(int)3.0*curand_uniform(&rstate_b1[block_num]);

				}while(mstate_d[((block_x+x2) * 9)+ (block_y+y2)]==1);

		
				temp=shared_puzzle[block_x+x1][block_y+y1];
				shared_puzzle[block_x+x1][block_y+y1]=shared_puzzle[block_x+x2][block_y+y2];
				shared_puzzle[block_x+x2][block_y+y2]=temp;

				new_score=compute_score_d(shared_puzzle);
				if(new_score >= current_score){
					temp=shared_puzzle[block_x+x1][block_y+y1];
					shared_puzzle[block_x+x1][block_y+y1]=shared_puzzle[block_x+x2][block_y+y2];
					shared_puzzle[block_x+x2][block_y+y2]=temp;
					new_score = current_score;
				}
			}else{
				if(block_num == 1){
					block_x = 3*(int)(3.0*curand_uniform(&rstate_b2[block_num]));
					block_y = 3*(int)(3.0*curand_uniform(&rstate_b2[block_num]));
					do
					{
						x1=(int)3.0*curand_uniform(&rstate_b2[block_num]);
						y1=(int)3.0*curand_uniform(&rstate_b2[block_num]);

					}while(mstate_d[((block_x+x1) * 9 )+(block_y+y1)]==1);


					do{
						x2=(int)3.0*curand_uniform(&rstate_b2[block_num]);
						y2=(int)3.0*curand_uniform(&rstate_b2[block_num]);

					}while(mstate_d[((block_x+x2) * 9)+ (block_y+y2)]==1);

			
					temp=shared_puzzle[block_x+x1][block_y+y1];
					shared_puzzle[block_x+x1][block_y+y1]=shared_puzzle[block_x+x2][block_y+y2];
					shared_puzzle[block_x+x2][block_y+y2]=temp;

					new_score=compute_score_d(shared_puzzle);
					if(new_score >= current_score){
						temp=shared_puzzle[block_x+x1][block_y+y1];
						shared_puzzle[block_x+x1][block_y+y1]=shared_puzzle[block_x+x2][block_y+y2];
						shared_puzzle[block_x+x2][block_y+y2]=temp;
						new_score = current_score;
					}
				}
				
			}	
		}

		if(block_num == 0){
			sudoku_db1[(thread_x * 9) + thread_y] = shared_puzzle[thread_x][thread_y];
		}else{
		 if (block_num == 1){
				sudoku_db2[(thread_x * 9) + thread_y] = shared_puzzle[thread_x][thread_y];
			}
		}		
	
}


__global__ void compute_score(int *sudoku_db1, int* sudoku_db2,int * d_score){
	
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	int count[10] = {0,0,0,0,0,0,0,0,0,0};
	int row = (idx / 9) % 2;
	int array = idx / 18;
	for(int i = 0; i < 9; i++){
		if((row == 0) && (array == 0)){
			count[sudoku_db1[(idx * 9) + i]] += 1;
		}else if ((row == 1) && (array == 0)){
			count[sudoku_db1[(idx - 9) + (i * 9)]] += 1;
		}else if ((row == 0) && (array == 1)){
			count[sudoku_db2[((idx - 18) * 9)  + i]] += 1;
		}else if ((row == 1) && (array == 1)){
			count[sudoku_db2[(idx - 27) + (i * 9)]] += 1;
		}
	}
	
	int num_unique = 0;
	for(int i = 0; i < 10; i++){
		if(count[i] > 0){
			num_unique++;
		}
	}
	d_score[idx] = num_unique;

}

__global__ void crossover(int* rsrc, int *rdest, int * csrc, int * cdest, int r, int c){
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	

	rdest[(r * 27) + idx] = rsrc[(r * 27) + idx];
	cdest[(3 * c) + ((idx / 3) * 9) + (idx % 3)] = csrc[(3 * c) + ((idx / 3) * 9) + (idx % 3)];

}

__global__ void mos(int* sudoku_src, curandState * r3state,int current_score,int * score_block,int * block0,int * block1, int * block2, int * block3, int * block4, int * block5){

	__shared__ int shared_puzzle[9][9];
	
	int thread_x=threadIdx.x;
	int thread_y=threadIdx.y;
	int thread_block_id = threadIdx.x*blockDim.x + threadIdx.y;
	int block_num= blockIdx.x*blockDim.x + blockIdx.y;
	int block_x;
	int block_y;
	int x1, y1, x2, y2;
	int temp;
	int new_score = 1000;
	float divisor = 0.4;
	
	shared_puzzle[thread_x][thread_y]=sudoku_src[(thread_x * 9) + thread_y];

	if(thread_block_id == 0){
		
		for(int i = 0; i < 1000; i++){
			block_x = 3*(int)(3.0*curand_uniform(&r3state[block_num]));
			block_y = 3*(int)(3.0*curand_uniform(&r3state[block_num]));
			do
			{
				x1=(int)3.0*curand_uniform(&r3state[block_num]);
				y1=(int)3.0*curand_uniform(&r3state[block_num]);

			}while(mstate_d[((block_x+x1) * 9 )+(block_y+y1)] == 1);


			do{
				x2=(int)3.0*curand_uniform(&r3state[block_num]);
				y2=(int)3.0*curand_uniform(&r3state[block_num]);

			}while(mstate_d[((block_x+x2) * 9)+ (block_y+y2)] == 1);

	
			temp=shared_puzzle[block_x+x1][block_y+y1];
			shared_puzzle[block_x+x1][block_y+y1]=shared_puzzle[block_x+x2][block_y+y2];
			shared_puzzle[block_x+x2][block_y+y2]=temp;

			new_score=compute_score_d(shared_puzzle);
			if(new_score < current_score){
				current_score = new_score;
			}else{
			 if((exp((float)(current_score - new_score)/divisor)) > (curand_uniform(&r3state[block_num])))
				{
					current_score = new_score;
				}
				else{
					temp=shared_puzzle[block_x+x1][block_y+y1];
					shared_puzzle[block_x+x1][block_y+y1]=shared_puzzle[block_x+x2][block_y+y2];
					shared_puzzle[block_x+x2][block_y+y2]=temp;
				}
			}
			
			if(new_score == 0){
				break;
			}

		}
		
		for(int i=0;i<9;i++)
		{
			for(int j=0;j<9;j++)
			{
				if(block_num==0)
					block0[9*i+j]=shared_puzzle[i][j];
				if(block_num==1)
					block1[9*i+j]=shared_puzzle[i][j];
				if(block_num==2)
					block2[9*i+j]=shared_puzzle[i][j];
				if(block_num==3)
					block3[9*i+j]=shared_puzzle[i][j];
				if(block_num==4)
					block4[9*i+j]=shared_puzzle[i][j];
				if(block_num==5)
					block5[9*i+j]=shared_puzzle[i][j];
			}
		}
		
		score_block[block_num] = new_score;
	}
	
}

void init_sudoku(char const * filename ){
    FILE* file = fopen(filename, "r"); 
    char line[256];
    int a = 0;
    int b = 0;
    if(file == NULL){
      perror("error while opening the file. \n");
      exit(EXIT_FAILURE);
    }
    while (fgets(line, sizeof(line), file)) {
        if( a == 9){
            break;
        }
        for(b = 0; b < 9; b++){
            char str[2];
            str[0] = line[b];
            str[1] =  '\0';
            sudoku[a * 9 + b] = (int)strtol(str, (char **)NULL, 10);
	    if(sudoku[a * 9 + b] != 0){
		state[a * 9 + b] = 1;
	    }else{
		state[a * 9 + b] = 0;
	    }
           
        }
        a++;
    }
	printf("input puzzle \n");
	int row = 0;
	int column = 0;	
	for(row=0;row<9;row++)
    	{
	    for(column=0;column<9;column++)
		    printf("%d ",sudoku[row * 9 + column]);
    	    printf("\n");
    	}
	
	int x, y;
	int p, q;
	int idx;

	int nums_1[9],nums_2[9];

	for(int block_i=0;block_i<3;block_i++)
	{
		for(int block_j=0;block_j<3;block_j++)
		{
			for(int k=0;k<9;k++)
				nums_1[k]=k+1;

				for(int i=0;i<3;i++)
				{
					for(int j=0;j<3;j++)
					{
						x = block_i*3 + i;
						y = block_j*3 + j;

						if(sudoku[x*9+y]!=0){
							p = sudoku[x*9+y];
							nums_1[p-1]=0;
						}
					}
				}
				q = -1;
				for(int k=0;k<9;k++)
				{
					if(nums_1[k]!=0)
					{
						q+=1;
						nums_2[q] = nums_1[k];
					}
				}
				idx = 0;
				for(int i=0;i<3;i++)
				{
					for(int j=0;j<3;j++)
					{
						x = block_i*3 + i;
						y = block_j*3 + j;
						if(sudoku[x*9+y]==0)
						{
							sudoku[x*9+y] = nums_2[idx];
							idx+=1;
						}
					}
				}

			}
		}
}
void sudoku_call_gpu(char const * filename1){
	int device = 1;
	cudaGetDevice(&device);
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop,device);
	cudaSetDevice(1);
	
	bool solved = false;
	
	int * sudoku_db1;
	int * sudoku_db2;
	int * block10_d;
	int * block11_d;
	int * block12_d;
	int * block13_d;
	int * block14_d;
	int * block15_d;
	int * block20_d;
	int * block21_d;
	int * block22_d;
	int * block23_d;
	int * block24_d;
	int * block25_d;
	int * score1_block;
	int * score2_block;
	int  score1_host[4];
	int  score2_host[4];
	int * block_num_val;
	int block_num_val_h[162];
	
	int sudoku_hb1[81];
	int sudoku_hb2[81];
	int new_score;
	int puzz1_score = 0;
	int puzz2_score = 0;
	int puzz1_prev_score = 0;
	int puzz2_prev_score = 0;
	int left_energy = 1000;
	int min_score1 = 162;
	int min_score2 = 162;
	int min1_id = 0;
	int min2_id = 0;
	int itr = 100;
	int itr1 = 3;
	int itr2 = 4;
	int itr3 = 5;
	int copy = 0;
	float divisor = 0.4;
	int row = 0;
	int column = 0;
	size_t size = len * sizeof(int);

	gpuErrchk(cudaMalloc((void**) &sudoku_db1,size));
	gpuErrchk(cudaMemcpy(sudoku_db1,sudoku,size,cudaMemcpyHostToDevice));
	gpuErrchk(cudaMalloc((void**) &sudoku_db2,size));
	gpuErrchk(cudaMemcpy(sudoku_db2,sudoku,size,cudaMemcpyHostToDevice));
	gpuErrchk(cudaMalloc((void**)&mstate_d,size));
	gpuErrchk(cudaMemcpyToSymbol(mstate_d,state,size));
	
	gpuErrchk(cudaMalloc((void**) &block_num_val,(162*sizeof(int))));

	gpuErrchk(cudaMalloc((void**) &block10_d,size));
	gpuErrchk(cudaMalloc((void**) &block11_d,size));
	gpuErrchk(cudaMalloc((void**) &block12_d,size));
	gpuErrchk(cudaMalloc((void**) &block13_d,size));
	gpuErrchk(cudaMalloc((void**) &block14_d,size));
	gpuErrchk(cudaMalloc((void**) &block15_d,size));
	gpuErrchk(cudaMalloc((void**) &block20_d,size));
	gpuErrchk(cudaMalloc((void**) &block21_d,size));
	gpuErrchk(cudaMalloc((void**) &block22_d,size));
	gpuErrchk(cudaMalloc((void**) &block23_d,size));
	gpuErrchk(cudaMalloc((void**) &block24_d,size));
	gpuErrchk(cudaMalloc((void**) &block25_d,size));
	gpuErrchk(cudaMalloc((void**) &score1_block,sizeof(int)*6));
	gpuErrchk(cudaMalloc((void**) &score2_block,sizeof(int)*6));
	
	dim3 dimGrid(1,1);
	dim3 dimBlock(9,9);

	curandState *d_rstate_b1;
	curandState *d_rstate_b2;
	curandState *d_rstate_3;
	curandState *hstate;
	gpuErrchk(cudaMalloc(&d_rstate_b1, dimBlock.x* dimBlock.y * dimGrid.x * dimGrid.y * sizeof(curandState)));
	gpuErrchk(cudaMalloc(&d_rstate_b2, dimBlock.x* dimBlock.y * dimGrid.x * dimGrid.y * sizeof(curandState)));
	gpuErrchk(cudaMalloc(&d_rstate_3, dimBlock.x* dimBlock.y * dimGrid.x * dimGrid.y * sizeof(curandState)));
	
	int * d_score;
	int h_score[36];
	gpuErrchk(cudaMalloc((void**) &d_score,(36 * sizeof(int))));
	
	time_t t;
	srand((unsigned) time(&t));
	
	if(compute_score_h(sudoku) == 0){
		solved = true;
	}
		
	while(itr2 > 0){
		if(solved == true){
			break;
		}
		int seed1 = rand() % 10000;
		int seed2 = rand() % 10000;
		int seed3 = rand() % 10000;

		initCurand<<<dimGrid.x * dimGrid.y, dimBlock.x* dimBlock.y>>>(d_rstate_b1,d_rstate_b2,d_rstate_3 ,seed1,seed2,seed3);
		gpuErrchk(cudaPeekAtLastError());
		gpuErrchk(cudaDeviceSynchronize());

		dim3 dimGrid1(1,2);
		dim3 dimBlock1(9,9);

		init_rsudoku<<<dimGrid1, dimBlock1>>>(d_rstate_b1,d_rstate_b2,sudoku_db1,sudoku_db2);
		
		gpuErrchk(cudaMemcpy(sudoku_hb1,sudoku_db1,size,cudaMemcpyDeviceToHost));
		gpuErrchk(cudaMemcpy(sudoku_hb2,sudoku_db2,size,cudaMemcpyDeviceToHost));
	
		gpuErrchk(cudaMemcpy(block_num_val_h,block_num_val,(162*sizeof(int)),cudaMemcpyDeviceToHost));
		
		gpuErrchk(cudaPeekAtLastError());
		gpuErrchk(cudaDeviceSynchronize());
	
		dim3 dimGrid2(1,2);
		dim3 dimBlock2(2,9);
		
		int summed_score[12];
		int j = 0;
		int largest_grid_col = 0;
		int c = 0;
		int largest_grid_col_puzzle = 0;
		int largest_grid_row = 0;
		int largest_grid_row_puzzle = 0;
		int r = 0;

		compute_score<<<dimGrid2.x * dimGrid2.y,dimBlock2.x* dimBlock2.y>>>(sudoku_db1,sudoku_db2,d_score);
		gpuErrchk(cudaMemcpy(h_score,d_score,(36*sizeof(int)),cudaMemcpyDeviceToHost));

		puzz1_prev_score = puzz1_score;
		puzz2_prev_score = puzz2_score;
		puzz1_score = 0;
		puzz2_score = 0;

		for(int i = 0; i < 36; i++){
			if((i % 3) == 0){
				if(i != 0){
					j++;
				}		
			summed_score[j] = 0;
			}
			if(i < 18){
				puzz1_score += h_score[i];
			}else{
				puzz2_score += h_score[i];
			}	
		  summed_score[j] += h_score[i];
			
			
		}
		puzz1_score = 162 - puzz1_score;
		puzz2_score = 162 - puzz2_score;
		

		for(int i = 0; i < 12; i++){
			if(((i / 3) % 2) == 0){
				if(largest_grid_row < summed_score[i]){
					largest_grid_row = summed_score[i];
					r = (i % 3);
					largest_grid_row_puzzle = (i / 6);
				}
			}else{
				if(largest_grid_col < summed_score[i]){
					largest_grid_col = summed_score[i];
					c = (i % 3);
					largest_grid_col_puzzle = (i / 6);
				}

			}			
		
		}
	
		dim3 dimGrid3(1,1);
		dim3 dimBlock3(3,9);
		
		if((largest_grid_row_puzzle == 0) && (largest_grid_col_puzzle == 0)) {
		//crossover<<<dimGrid3.x * dimGrid3.y,dimBlock3.x* dimBlock3.y>>>(rsrc,rdest,csrc,cdest,r,c);
		crossover<<<dimGrid3.x * dimGrid3.y,dimBlock3.x* dimBlock3.y>>>(sudoku_db1,sudoku_db2,sudoku_db1,sudoku_db2,r,c);
		}else if((largest_grid_row_puzzle == 0) && (largest_grid_col_puzzle == 1)){
		crossover<<<dimGrid3.x * dimGrid3.y,dimBlock3.x* dimBlock3.y>>>(sudoku_db1,sudoku_db2,sudoku_db2,sudoku_db1,r,c);
		}else if((largest_grid_row_puzzle == 1) && (largest_grid_col_puzzle == 0)){
		crossover<<<dimGrid3.x * dimGrid3.y,dimBlock3.x* dimBlock3.y>>>(sudoku_db2,sudoku_db1,sudoku_db1,sudoku_db2,r,c);
		}else if((largest_grid_row_puzzle == 1) && (largest_grid_col_puzzle == 1)){
		crossover<<<dimGrid3.x * dimGrid3.y,dimBlock3.x* dimBlock3.y>>>(sudoku_db2,sudoku_db1,sudoku_db2,sudoku_db1,r,c);
		}

		compute_score<<<dimGrid2.x * dimGrid2.y,dimBlock2.x* dimBlock2.y>>>(sudoku_db1,sudoku_db2,d_score);
		gpuErrchk(cudaMemcpy(h_score,d_score,(36*sizeof(int)),cudaMemcpyDeviceToHost));
		
		puzz1_prev_score = puzz1_score;
		puzz2_prev_score = puzz2_score;
		puzz1_score = 0;
		puzz2_score = 0;
		j = 0;
		for(int i = 0; i < 36; i++){
			if((i % 3) == 0){
				if(i != 0){
					j++;
				}		
			summed_score[j] = 0;
			}
			if(i < 18){
				puzz1_score += h_score[i];
			}else{
				puzz2_score += h_score[i];
			}	
		  summed_score[j] += h_score[i];
			
			
		}
		puzz1_score = 162 - puzz1_score;
		puzz2_score = 162 - puzz2_score;
			
		if(puzz1_score < puzz1_prev_score){
			gpuErrchk(cudaMemcpy(sudoku_hb1,sudoku_db1,size,cudaMemcpyDeviceToHost));
		}else
		{
			gpuErrchk(cudaMemcpy(sudoku_db1,sudoku_hb1,size,cudaMemcpyHostToDevice));
			puzz1_score = puzz1_prev_score;
		}
		if(puzz2_score < puzz2_prev_score){
			gpuErrchk(cudaMemcpy(sudoku_hb2,sudoku_db2,size,cudaMemcpyDeviceToHost));
		}else{
			gpuErrchk(cudaMemcpy(sudoku_db2,sudoku_hb2,size,cudaMemcpyHostToDevice));
			puzz2_score = puzz2_prev_score;
		}

		dim3 dimGrid4(1,6);
		dim3 dimBlock4(9,9);

		
		while(itr3 > 0){
			mos<<<dimGrid4,dimBlock4>>>(sudoku_db1, d_rstate_3,puzz1_score,score1_block,block10_d,block11_d,block12_d,block13_d,block14_d, block15_d);

			gpuErrchk(cudaDeviceSynchronize());
			mos<<<dimGrid4,dimBlock4>>>(sudoku_db2, d_rstate_3,puzz2_score,score2_block,block20_d,block21_d,block22_d,block23_d,block24_d, block25_d);
			
			gpuErrchk(cudaDeviceSynchronize());
			cudaMemcpy(score1_host,score1_block,sizeof(int)*6,cudaMemcpyDeviceToHost);
			gpuErrchk(cudaDeviceSynchronize());
			cudaMemcpy(score2_host,score2_block,sizeof(int)*6,cudaMemcpyDeviceToHost);

			int g = 0;
			for(g=0;g<6;g++)
			{
				if(score1_host[g] < min_score1)
				{
					min_score1 = score1_host[g];
					min1_id = g;
				}
				if(score2_host[g] < min_score2)
				{
					min_score2 = score2_host[g];
					min2_id = g;
				}
			}
			puzz1_prev_score = puzz1_score;
			puzz2_prev_score = puzz2_score;
			if(min1_id==0)
			{
				cudaMemcpy(sudoku_db1,block10_d,size,cudaMemcpyDeviceToDevice);
				puzz1_score=min_score1;
			}
			if(min1_id==1)
			{
				cudaMemcpy(sudoku_db1,block11_d,size,cudaMemcpyDeviceToDevice);
				puzz1_score=min_score1;
			}
			if(min1_id==2)
			{
				cudaMemcpy(sudoku_db1,block12_d,size,cudaMemcpyDeviceToDevice);
				puzz1_score=min_score1;
			}
			if(min1_id==3)
			{
				cudaMemcpy(sudoku_db1,block13_d,size,cudaMemcpyDeviceToDevice);
				puzz1_score=min_score1;
			}
			if(min1_id==4)
			{
				cudaMemcpy(sudoku_db1,block14_d,size,cudaMemcpyDeviceToDevice);
				puzz1_score=min_score1;
			}
			if(min1_id==5)
			{
				cudaMemcpy(sudoku_db1,block15_d,size,cudaMemcpyDeviceToDevice);
				puzz1_score=min_score1;
			}
			if(min2_id==0)
			{
				cudaMemcpy(sudoku_db2,block20_d,size,cudaMemcpyDeviceToDevice);
				puzz2_score=min_score2;
			}
			if(min2_id==1)
			{
				cudaMemcpy(sudoku_db2,block21_d,size,cudaMemcpyDeviceToDevice);
				puzz2_score=min_score2;
			}
			if(min2_id==2)
			{
				cudaMemcpy(sudoku_db2,block22_d,size,cudaMemcpyDeviceToDevice);
				puzz2_score=min_score2;
			}
			if(min2_id==3)
			{
				cudaMemcpy(sudoku_db2,block23_d,size,cudaMemcpyDeviceToDevice);
				puzz2_score=min_score2;
			}
			if(min2_id==4)
			{
				cudaMemcpy(sudoku_db2,block24_d,size,cudaMemcpyDeviceToDevice);
				puzz2_score=min_score2;
			}
			if(min2_id==5)
			{
				cudaMemcpy(sudoku_db2,block25_d,size,cudaMemcpyDeviceToDevice);
				puzz2_score=min_score2;
			}
			
			if(puzz1_score == 0){
				copy = 1;
				break;
			}else if(puzz2_score == 0){
				copy = 1;
				cudaMemcpy(sudoku_db1,sudoku_db2,size,cudaMemcpyDeviceToDevice);
				break;
			}
			
			if((puzz1_score == puzz1_prev_score) || (puzz2_score == puzz2_prev_score)){
				itr3--;
			}else{
				itr3 = 5;
			}
		}
		
		if((puzz1_score == puzz1_prev_score) || (puzz2_score == puzz2_prev_score)){
			itr2--;
		}else{
			itr2 = 4;
		}
		if(itr2 < 0){
			cudaMemcpy(sudoku,sudoku_hb1,size,cudaMemcpyDeviceToHost);
			int array[3]={0,3,6};
			int tmp;
			int random1=random()%3;
			int random2=random()%3;

			int x1,y1,x2,y2;
			int block_x,block_y;

			for(int suffle=0;suffle<random()%10;suffle++)
			{
				block_x = array[random1];
				block_y = array[random2];
				do{
					x1=random()%3;
					y1=random()%3;;
				}while(state[((block_x+x1)*9)+(block_y+y1)]==1);

				do{
					x2=random()%3;;
					y2=random()%3;;
				}while(state[((block_x+x2)*9)+(block_y+y2)]==1);

				tmp = sudoku[((block_x+x1)*9)+(block_y+y1)];
				sudoku[((block_x+x1)*9)+(block_y+y1)]=sudoku[((block_x+x2)*9)+(block_y+y2)];
				sudoku[((block_x+x2)*9)+(block_y+y2)]=tmp;
			}
			cudaMemcpy(sudoku_db1,sudoku,size,cudaMemcpyHostToDevice);
			cudaMemcpy(sudoku_db2,sudoku,size,cudaMemcpyHostToDevice);
			new_score = compute_score_h(sudoku);
			if(new_score == 0){
				solved = true;
				//break;
			}
			puzz1_score = new_score;
			puzz2_score = new_score;
			
			
			itr2 = 4;
			divisor = divisor + 0.1;
		}
	}
	
	gpuErrchk(cudaMemcpy(sudoku_hb1,sudoku_db1,size,cudaMemcpyDeviceToHost));
	printf("\n");
	printf("Solved Puzzle \n");

	for(row=0;row<9;row++)
    	{
	    for(column=0;column<9;column++)
		    printf("%d ",sudoku_hb1[row * 9 + column]);
    	    printf("\n");
    	}

	cudaFree(sudoku_db1);
	cudaFree(sudoku_db2);
	cudaFree(block10_d);
	cudaFree(block11_d);
	cudaFree(block12_d);
	cudaFree(block13_d);
	cudaFree(block14_d);
	cudaFree(block15_d);
	cudaFree(block20_d);
	cudaFree(block21_d);
	cudaFree(block22_d);
	cudaFree(block23_d);
	cudaFree(block24_d);
	cudaFree(block25_d);
	cudaFree(score1_block);
	cudaFree(score2_block);
	
	char * resstr;
	char * lastdot;
	strcpy (resstr, filename1);
    lastdot = strrchr (resstr, '.');
    if (lastdot != NULL)
        *lastdot = '\0';
	resstr = strcat(resstr,".sol\0");
    //printf(resstr);
	
	FILE *f = fopen(resstr, "w+");
	if (f == NULL)
	{
		printf("Error opening outputfile!\n");
		exit(1);
	}
	for(row = 0; row < 9; row++){
		for(column = 0; column < 9; column++){
			if(copy == 1){
				fprintf(f,"%d",sudoku_hb1[row * 9 + column]);
			
			}
			else{
			
				fprintf(f,"%d",sudoku[row * 9 + column]);
			
			}
			
		
		}
		fprintf(f,"\n");
		
	}
	fclose(f);
	

	
}

int main(int argc, char *argv[])
{
	if(argc < 2){
		printf("Input and output file path required.");
		exit(-1);
	}
   init_sudoku(argv[1]);
   sudoku_call_gpu(argv[1]);

   return 0;
}

