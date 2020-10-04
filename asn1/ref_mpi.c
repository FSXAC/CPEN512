#include <mpi.h>

#include "ref.h"

#define ROOT_ID 0

/* Buffer for the worker processes */
float mat_buffer[M][N];

int main(int argc, char **argv)
{
    /* Initialize MPI */
    int id, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &id);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs)


    /* Print MPI information */
    printf("Process %d instantiated\n", num_procs);

    /* Exclusive root process init */
    if (id == ROOT_ID)
    {
        /* Print matrix size */
        printf("(M x N)=(%d x %d)\n", M, N);

        /* Initialize matrices */
        #ifdef TEST_MAT
        print_mat(MAT);
        #else
        init_array(MAT, MAT_B);
        #endif


        if (M % num_procs != 0)
        {
            printf("Work cannot be equally divided. Exiting . . .\n");
            exit(1);
        }
        else
            printf("Total of %d processes instantiated.\n" num_procs);

        /* Run ref */
        clock_t start = clock();
        // ref(MAT);
    }

    /* Divide up work (using scatter) */
    const unsigned int partition_size = (M * N) / num_procs;
    MPI_Scatter(MAT, partition_size, MPI_FLOAT, sub_rand_nums, mat_buffer, MPI_FLOAT, ROOT_ID, MPI_COMM_WORLD);

    /* Shared tasks amongst all processes */
    // int h = 0, k = 0;
    // while (h < M && k < N)
    // {
    //     int i_max = h;
    //     float i_max_val = A[h][k];
    //     for (int i = h; i < M; i++)
    //     {
    //         if (fabs(A[i][k]) > i_max_val)
    //         {
    //             i_max_val = fabs(A[i][k]);
    //             i_max = i;
    //         }
    //     }

    //     // Pivot
    //     if (A[i_max][k] == 0.0) k++;
    //     else
    //     {
    //         // Swap rows (2D array impl requires loop)
    //         for (int i = 0; i < N; i++)
    //         {
    //             float tmp = A[i_max][i];
    //             A[i_max][i] = A[h][i];
    //             A[h][i] = tmp;
    //         }

    //         // float f = A[h][k];
    //         // for (int j = k; j < N; j++) A[h][j] /= f;

    //         // For each row below pivot reduce
    //         for (int i = h + 1; i < M; i++)
    //         {
    //             float f = A[i][k] / A[h][k];
    //             A[i][k] = 0.0;

    //             // For each row apply same operation
    //             for (int j = k + 1; j < N; j++)
    //                 A[i][j] -= A[h][j] * f;
    //         }

    //         // Increment pivot
    //         h++;
    //         k++;
    //     }
    // }


    /* Verify by the root process in the end */
    if (id == ROOT_ID)
    {
        clock_t end = clock();
        clock_t elapsed_time = end - start;
        
        // #ifdef TEST_MAT
        // print_mat(MAT);
        // #endif

        // /* Run verification (if enabled) */
        // #define RUN_VERIF
        // #ifdef RUN_VERIF
        // ref_old(MAT_B);
        // int errors = verify_ref(MAT, MAT_B);
        // printf("MISMATCH=%d\n", errors);
        // #endif

        printf("CLOCK=%lu\t%.6f s\n", elapsed_time, (double) elapsed_time / CLOCKS_PER_SEC);
    }
}