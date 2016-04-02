//The sequntial version for the sort is written using insertion sort which took 0.012s for execution.

#include <stdio.h>

void main(){
	int i,j,tmp,A[1024],B[1024],C[1024];

	for(i = 0; i < 1024;i++){
		
		A[i] = rand() % 10000;
		C[i] = A[i];
		B[i] = 0;
	}
	printf("\n");
	for (i = 0; i < 1024; ++i)
    	{
        	for (j = i + 1; j < 1024; ++j)
        	{
       		     if (C[i] > C[j])
            		{
		             	tmp =  C[i];
                		C[i] = C[j];
		                C[j] = tmp;
		         }
	        }
	 }
	for(i = 0;i < 1024; i++){
		B[i] = C[i];
		printf("A[%d] = %d B[%d] = %d\n", i, A[i], i, B[i]);
        }
        printf("\n");
}
