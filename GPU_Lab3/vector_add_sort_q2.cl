__kernel void vector_sort(__global int *A, int k, int j) {

	int i = get_global_id(0);
	int power_j =  i ^ j;

        if(power_j > i ){
                if ((i&k) == 0){
                        if (A[i]>A[power_j]){
                                int tmp = A[i];
                                A[i] = A[power_j];
                                A[power_j] = tmp;
                        }
                }
		if ((i&k)!=0) {
		      if (A[i]<A[power_j]) {
		        int tmp = A[i];
	        	A[i] = A[power_j];
	        	A[power_j] = tmp;
      			}
    		}
        }
	
}
