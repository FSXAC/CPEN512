#include "ref.h"

#define BLOCK_SIZE 1024

 __global__
void scale_row(float *MAT, int pivot)
{
    int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (tid < N)
    {
        /* Assuming pivot col == pivot row always */
        int pivot_idx = pivot * N + pivot;
        int current_idx = pivot * N + tid;
        float scale = MAT[pivot_idx];
        __syncthreads();
        MAT[current_idx] /= scale;
    }
}

__global__
void subtract_single_row(float *MAT, int row, int pivot)
{
    int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (tid < N)
    {
        int pivot_start_idx = pivot * N + pivot;
        int row_start_idx = row * N + pivot;
        int current_idx = row * N + tid;
        int pivot_idx = pivot * N + tid;

        float f = MAT[row_start_idx] / MAT[pivot_start_idx];
        MAT[current_idx] -= f * MAT[pivot_idx];
    }
}

void ref_cuda(float *MAT)
{

    /* Allocate memory for the device */
    float *MATD;
    cudaMalloc(&MATD, sizeof(float) * M * N);
    cudaMemcpy(MATD, (void *) MAT, sizeof(float) * M * N, cudaMemcpyHostToDevice);

    struct timeval begin, end;
    gettimeofday(&begin, 0);

    /* Loop through rows */
    for (int row = 0; row < M; row++)
    {
        /* Assuming square positive matrix always */
        int col = row;

        /* Block size is the number of blocks required 
         * 
         * So if we have 32 elements in the row and BLOCK_SIZE is 16, we need
         * 2 blocks to process this row
         *
         * Optimization can be made since we don't need to process
         * all the elements before the pivot, since
         * we can assume they're zeros
         *
         * Normally we just do M / BLOCK_SIZE
         * But this is modified so that we always have at least 1 block
         */
        int elements_to_process = N;
        int num_blocks = (int) ceil((float) elements_to_process / BLOCK_SIZE);
        scale_row<<<num_blocks, BLOCK_SIZE>>>(MATD, row);
        cudaDeviceSynchronize();

        #ifdef DEBUG_GPU
        printf("Scaling row %d\n", row);
        cudaMemcpy(MAT, (void *) MATD, sizeof(float) * M * N, cudaMemcpyDeviceToHost);
        print_mat(MAT);
        #endif

        for (int subrow = row + 1; subrow < M; subrow++)
            subtract_single_row<<<num_blocks, BLOCK_SIZE>>>(MATD, subrow, row);
        cudaThreadSynchronize();

        #ifdef DEBUG_GPU
        printf("Eliminating rows after row %d\n", row);
        cudaMemcpy(MAT, (void *) MATD, sizeof(float) * M * N, cudaMemcpyDeviceToHost);
        print_mat(MAT);
        #endif
    }

    gettimeofday(&end, 0);
    double time_parallel = (end.tv_sec - begin.tv_sec) + (end.tv_usec - begin.tv_usec) * 1e-6;
    printf("Time parallel: %.6e s\n", time_parallel);

    /* Copy back from device to host */
    cudaMemcpy(MAT, (void *) MATD, sizeof(float) * M * N, cudaMemcpyDeviceToHost);

    /* Free device memroy */
    cudaFree(MATD);
    
    
}

int main(void)
{
    /* Time keeping */
    struct timeval begin, end;
    double time_serial;

    /* Malloc matrices */
    MAT   = (float *) malloc(sizeof(float) * N * M);
    MAT_B = (float *) malloc(sizeof(float) * N * M);

    printf("(M x N)=(%d x %d)\n", M, N);
    init_array(MAT);
    memcpy(MAT_B, MAT, sizeof(float) * N * M);
    print_mat(MAT);

    /* Run single threaded */
    printf("Running serial . . .\n");
    gettimeofday(&begin, 0);
    ref_old_noswap(MAT_B);
    gettimeofday(&end, 0);
    time_serial = (end.tv_sec - begin.tv_sec) + (end.tv_usec - begin.tv_usec) * 1e-6;
    print_mat(MAT_B);

    /* Run parallel ref */
    printf("Running parallel . . .\n");
    // gettimeofday(&begin, 0);
    ref_cuda(MAT);
    // gettimeofday(&end, 0);
    // double time_parallel = (end.tv_sec - begin.tv_sec) + (end.tv_usec - begin.tv_usec) * 1e-6;
    // print_mat(MAT);

    /* Run verification (if enabled) */
    #ifdef RUN_VERIF
    printf("Running verification . . .\n");
    int errors = verify_ref(MAT, MAT_B);
    printf("MISMATCH=%d\n", errors);
    #endif

    printf("SERIAL TIME=%.6e s\n", time_serial);
    // printf("PARALL TIME=%.6e s\n", time_parallel);

    return 0;
}
