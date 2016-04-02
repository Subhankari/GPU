#include<stdio.h>
#include<stdlib.h>
#include<time.h>

float calculate(float top, float left, float right, float bottom){
    float value = 0.25 * (top + left + right + bottom);
    return value;
}

void main(){
    clock_t tic;
    clock_t toc;
	tic = clock();
    long width = 10000;
    long iterations = 50;
    long k = 0;
    long i = 0;
    long j = 0;
    long l = 0;
    long m = 0;
    //FILE *f = fopen("outfileseq_test", "w");
    //FILE *fp = fopen("C:/Users/Subhankari/Desktop/GPU assignment/lab1.txt",'w');
    long len = width * width;
    float *g =(float*)malloc(len*sizeof(float));
    if(g == NULL){
	  printf("Out of memory\n");
	   exit(-1);
	}
	float *h = (float*)malloc(len*sizeof(float));
	if(g == NULL){
	  printf("Out of memory\n");
	   exit(-1);
	}


 for( j = 0; j < len; j++){
	if((j >= 10) && (j <= 30)){
        g[j] = 150;
    }else if((j < width) || ((j % width) == 0) || ((j % width) == (width - 1)) || (j >= (width * (width - 1)))){
        g[j] = 80;
    }else{
        g[j] = 0;
    }
 }



    for(k = 0; k < iterations ; k++){
        for(i = 0; i < len; i++){
            long row = (i / width);
            long col = i % width;
            long left = i - 1;
            long right = i + 1;
            long top = ((row - 1) * width) + col;
            long bottom = ((row + 1) * width + col);
            if(((i % width) == 0) || ((i % width) == (width - 1)) || (i < width) || (i >= (width * (width - 1)))){
                h[i] = g[i];
            }else{
                h[i] = 0.25 * (g[top] + g[left] + g[bottom] + g[right]);
            }
        }
        //fprintf(fp,"print after %i iterations.\n", k);

        for(l = 0; l < len ; l++){
				g[l] = h[l];
				printf("%f\n",g[l]);
            }
        }

/*        for(i = 0; i < len; i++){
                  //  printf("failed");
            printf("%f\n",g[i]);
                //fprintf(f,"%f\n",g[l]);
        }
            printf("failed1");*/

    toc = clock();
	double time_taken_seq = (double)(toc - tic)/CLOCKS_PER_SEC; // in seconds
	printf("time taken in seq: %f\n",time_taken_seq);


	free(g);
	free(h);

    }

