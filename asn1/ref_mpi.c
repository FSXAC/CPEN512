#include <mpi.h>

#include "ref.h"

#define ROOT_ID 0

/* Buffer for the worker processes */
float mat_buffer[M][N];

int main(int argc, char **argv)
{
    /* Time measures */
    double start, end;

    /* Initialize MPI */
    int id, num_procs;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &id);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    /* Print MPI information */
    printf("Process %d instantiated\n", id);

    /* Exclusive root process init */
    if (id == ROOT_ID)
    {
        /* Print matrix size */
        printf("(M x N)=(%d x %d)\n", M, N);

        /* Initialize matrices */
        #ifdef TEST_MAT
        print_mat(MAT);
        #else
        srand(13);
        init_array(MAT, MAT_B);
        print_mat(MAT);
        #endif


        if (M % num_procs != 0)
        {
            printf("Work cannot be equally divided. Exiting . . .\n");
            exit(1);
        }
        else
            printf("Total of %d processes instantiated.\n", num_procs);

        start = MPI_Wtime();

        /* Put matrix into "good"-enough form */
        // int h = 0;
        // int k = 0;
        // while (h < N && k < M)
        // {
        //     if (MAT[h][k] == 0.0)
        //     {
        //         int h_max = find_max_row_index(MAT, h, k);
        //         if (MAT[h_max][k] != 0.0)
        //             swap_rows(MAT, h, h_max);
        //         else
        //             k++;
        //     }

        //     h++;
        //     k++;
        // }
        // print_mat(MAT);
    }

    /* Divide up work (using scatter) */
    const unsigned int partition_size = (M * N) / num_procs;
    MPI_Scatter(MAT, partition_size, MPI_FLOAT, mat_buffer, partition_size, MPI_FLOAT, ROOT_ID, MPI_COMM_WORLD);

    // DEBUG: TEST
    // for each process, we will do a print mat on the buffer
    // print_mat2(mat_buffer, M / num_procs, N);

    /* Our worker ID gives a hint to how we orchestrate communications */

    /* Verify by the root process in the end */
    if (id == ROOT_ID)
    {
        end = MPI_Wtime();
        const double elapsed_time = end - start;

        // Sanity check
        ref_noswap(MAT);
        print_mat(MAT);

        /* Run verification (if enabled) */
        #define RUN_VERIF
        #ifdef RUN_VERIF
        ref_old_noswap(MAT_B);
        int errors = verify_ref(MAT, MAT_B);
        printf("MISMATCH=%d\n", errors);
        #endif

        printf("TIME=%.6f s\n", elapsed_time);
    }

    return MPI_Finalize();
}