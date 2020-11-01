#include "ref.h"
// #include "cuda.h"
// #include "cuda_runtime_api.h"

#define TPB 8

 __global__
void scale_row(float *MAT, int pivot)
{
    int thread_x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int thread_y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (thread_y > 0)
        printf("thread y: %d \n", thread_y);

    int tid = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
    
    if (thread_x < N)
    {
        /* Assuming pivot col == pivot row always */
        int pivot_idx = pivot * N + pivot;
        int current_idx = pivot * N + thread_x;
        float scale = MAT[pivot_idx];
        __syncthreads();
        printf("tid %d, array index %d (%d, %d)\n", tid, current_idx, pivot, thread_x); 
        MAT[current_idx] /= scale;
    }
}

__global__
void subtract_row(float *MAT, int pivot)
{
    int thread_x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int thread_y = (blockIdx.y * blockDim.y) + threadIdx.y;
    int tid = thread_y * N + thread_x;

    if (thread_y != pivot && thread_y < M && thread_x < N)
    {
        int pivot_idx = pivot * N + pivot;

        /* The factor to divide by */
        // float f = MAT[thread_y * N] / MAT[pivot_idx];
        // MAT[tid] -= f * MAT[pivot_idx];
    }
}

#define DEBUG_GPU
void ref_cuda(float *MAT)
{

    /* Allocate memory for the device */
    float *MATD;
    cudaMalloc(&MATD, sizeof(float) * M * N);
    cudaMemcpy(MATD, (void *) MAT, sizeof(float) * M * N, cudaMemcpyHostToDevice);

    /* Loop through rows */
    for (int row = 0; row < M; row++)
    {
        /* Assuming square positive matrix always */
        int col = row;

        /* Block size is the number of blocks required 
         * 
         * So if we have 32 elements in the row and TPB is 16, we need
         * 2 blocks to process this row
         *
         * Optimization can be made since we don't need to process
         * all the elements before the pivot, since
         * we can assume they're zeros
         *
         * Normally we just do M / TPB
         * But this is modified so that we always have at least 1 block
         */
        // int elements_to_process = N - col;
        int elements_to_process = N;
        int block_size = (elements_to_process - 1) / TPB + 1;
        printf("blocksize: %d\n", block_size);
        scale_row<<<block_size, TPB>>>(MATD, row);
        cudaThreadSynchronize();

        #ifdef DEBUG_GPU
        printf("Scaling row %d\n", row);
        cudaMemcpy(MAT, (void *) MATD, sizeof(float) * M * N, cudaMemcpyDeviceToHost);
        print_mat(MAT);
        #endif

        /* Block size is now the remining elements */
        // elements_to_process = (N - row) * M;
        // // elements_to_process = (N - row) * (M - col);
        // block_size = (elements_to_process - 1) / TPB + 1;
        // subtract_row<<<block_size, TPB>>>(MATD, row);
        // cudaThreadSynchronize();

        // #ifdef DEBUG_GPU
        // printf("Eliminating rows after row %d\n", row);
        // cudaMemcpy(MAT, (void *) MATD, sizeof(float) * M * N, cudaMemcpyDeviceToHost);
        // print_mat(MAT);
        // #endif
    }

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
    // printf("Running serial . . .\n");
    // gettimeofday(&begin, 0);
    // ref_old_noswap(MAT_B);
    // gettimeofday(&end, 0);
    // time_serial = (end.tv_sec - begin.tv_sec) + (end.tv_usec - begin.tv_usec) * 1e-6;
    // print_mat(MAT_B);

    /* Run parallel ref */
    printf("Running parallel . . .\n");
    gettimeofday(&begin, 0);
    ref_cuda(MAT);
    gettimeofday(&end, 0);
    double time_parallel = (end.tv_sec - begin.tv_sec) + (end.tv_usec - begin.tv_usec) * 1e-6;
    print_mat(MAT);

    /* Run verification (if enabled) */
    #ifdef RUN_VERIF
    printf("Running verification . . .\n");
    int errors = verify_ref(MAT, MAT_B);
    printf("MISMATCH=%d\n", errors);
    #endif

    printf("SERIAL TIME=%.6e s\n", time_serial);
    printf("PARALL TIME=%.6e s\n", time_parallel);

    /* Make sure we exit pthread */
    // pthread_exit(NULL);

    return 0;
}
