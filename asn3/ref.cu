#include "ref.h"
// #include "cuda.h"
// #include "cuda_runtime_api.h"

// __global__
void scale_row(float *MAT_IN, float *MAT_OUT, int pivot_col)
{
    int thread_x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int thread_y = (blockIdx.y * blockDim.y) + threadIdx.y;

    int tid = thread_y * N * thread_x;
    
    if (thread_y == pivot_col && thread_y < M && thread_x < N)
    {
        /* Assuming pivot col == pivot row always */
        int pivot_idx = pivot_col * N + pivot_col;
        MAT_OUT[tid] = MAT_IN[tid] / MAT_IN[pivot_idx];
    }
}

void ref_cuda(float *MAT_IN, float *MAT_OUT)
{
    /*  */
    

    /* Allocate memory for the device */
    float *MAT_DEV = malloc(sizeof(float) * M * N);

    
    cudaMalloc((void *) &MAT_DEV, sizeof(float) * M * N);
    cudaMalloc((void *) &MAT_DEV, sizeof(float) * M * N);
    cudaMemcpy()

    /* Loop through rows */
    for (int row = 0; row < M; row++)
    {
        int pivot_col = MIN(row, N);

        scale_row<<<1, 1>>>(MAT_IN, MAT_OUT, pivot_col);

    }
}

int main(void)
{
    /* Malloc matrices */
    MAT   = malloc(sizeof(float) * N * M);
    MAT_B = malloc(sizeof(float) * N * M);

    printf("(M x N)=(%d x %d)\n", M, N);

    #ifdef TEST_MAT
    print_mat(MAT);
    #else
    init_array(MAT);
    print_mat(MAT);
    #endif

    /* Run single threaded */
    printf("Running serial . . .\n");
    struct timeval begin, end;
    gettimeofday(&begin, 0);
    ref_old_noswap(MAT_B);
    gettimeofday(&end, 0);
    double time_serial = (end.tv_sec - begin.tv_sec) + (end.tv_usec - begin.tv_usec) * 1e-6;

    /* Run parallel ref */
    printf("Running parallel . . .\n");
    gettimeofday(&begin, 0);
    ref_pthread();
    gettimeofday(&end, 0);
    double time_parallel = (end.tv_sec - begin.tv_sec) + (end.tv_usec - begin.tv_usec) * 1e-6;

    /* Run verification (if enabled) */
    #ifdef RUN_VERIF
    printf("Running verification . . .\n");
    int errors = verify_ref(MAT, MAT_B);
    printf("MISMATCH=%d\n", errors);
    #endif

    printf("SERIAL TIME=%.6e s\n", time_serial);
    printf("PARALL TIME=%.6e s\n", time_parallel);

    /* Make sure we exit pthread */
    pthread_exit(NULL);

    return 0;
}