__kernel void vector_add(__global int*A, __global int *B) {

    
    // Get the index of the current element
    int i = get_global_id(0);

    // Do the operation
    B[i] = A[i] + B[i];
}
