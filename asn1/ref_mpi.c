#include <mpi.h>

#include "ref.h"

#define ROOT_ID 0

int main(int argc, char **argv)
{
    /* Buffer for the worker processes */
    float mat_buffer[M][N];
    float row_buffer[N];
    
    /* Time measures */
    double start, end;

    /* Initialize MPI */
    int id, num_procs;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &id);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    // if (num_procs == 1)
    // {
    //     printf("Note: running baseline\n");
    //     printf("(M x N)=(%d x %d)\n", M, N);
    //     start = MPI_Wtime();
    //     ref_noswap(MAT);
    //     end = MPI_Wtime();
    //     const double elapsed_time = end - start;

    //     MPI_Finalize();

    //     #ifdef RUN_VERIF
    //     ref_old_noswap(MAT_B);
    //     print_mat(MAT_B);
    //     int errors = verify_ref(MAT, MAT_B);
    //     printf("MISMATCH=%d\n", errors);
    //     #endif

    //     printf("TIME=%.6e s\n", elapsed_time);
    //     return 0;
    // }

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


        if (M % num_procs != 0 || num_procs > M)
        {
            printf("Work cannot be equally divided. Exiting . . .\n");
            MPI_Finalize();
            exit(1);
        }
        else
            printf("Total of %d processes instantiated.\n", num_procs);
    }

    /* Divide up work (using scatter) */
    const int partition_rows = M / num_procs;
    const int partition_size = N * partition_rows;
    MPI_Scatter(MAT, partition_size, MPI_FLOAT, mat_buffer, partition_size, MPI_FLOAT, ROOT_ID, MPI_COMM_WORLD);

    /* Start timer AFTER work has been distributed */
    if (id == ROOT_ID) start = MPI_Wtime();

    /* Debug print */
    // print_mat2(mat_buffer, M / num_procs, N);

    /* Our worker ID gives a hint to how we orchestrate communications */
    /* What does the worker do when they receive a row from above */

    /* For every row "above" this processe's we will receive partitision size times
     * number of processes from before (num_received_rows), where root (id == 0)
     * don't receive from anyone so this loop doesn't get executed.
     */

    /* Let h be the column of current pivot */
    int h;

    const int num_received_rows = id * partition_rows;
    for (int row_index = 0; row_index < num_received_rows; row_index++)
    {
        /* Use Broadcast to receive and put it in receiving buffer (row_buffer)
         * Note that the sender/root id can be determined from row index
         */
        const int sender_id = row_index / partition_rows;
        MPI_Bcast(row_buffer, N, MPI_FLOAT, sender_id, MPI_COMM_WORLD);

        /* "Global" pivot column is just the row index (on the diagonal)*/
        h = row_index;

        /* Once the received row is in the buffer, let's do elimination/reduction on all rows 
         * assigned for our current worker
         */

        /* Iterate through each row of the Matrix buffer */
        for (int i = 0; i < partition_rows; i++)
        {
            /* For each column of the row starting from pivot */
            const float f = mat_buffer[i][h] / row_buffer[h];
            for (int col = h; col < N; col++)
            {
                mat_buffer[i][col] -= f * row_buffer[col];
            }
        }
    }

    /* What does the worker do when it's their turn to reduce the row and send it to others */
    /* For each row in current partion */
    for (int local_row = 0; local_row < partition_rows; local_row++)
    {
        /* "Global" pivot column can be calculated using ID */
        h = (id * partition_rows) + local_row;

        /* For each element to the right of pivot, divide by the pivot value */
        for (int col = h + 1; col < N; col++)
            mat_buffer[local_row][col] /= mat_buffer[local_row][h];
        mat_buffer[local_row][h] = 1.0;

        /* Ok at this point we're done with current row, so let rows below it know */
        MPI_Bcast(&mat_buffer[local_row][0], N, MPI_FLOAT, id, MPI_COMM_WORLD);

        /* Partially reduce the remaining rows below for this worker */
        /* ?? */
        for (int row = local_row + 1; row < partition_rows; row++)
        {
            const float f = mat_buffer[row][h] / mat_buffer[local_row][h];
            mat_buffer[row][h] = 0.0;
            for (int col = h + 1; col < N; col++)
                mat_buffer[row][col] -= f * mat_buffer[local_row][col];
        }
    }

    /* Resolve broadcast receives for workers that no longer needs it */
    for (int i = num_received_rows + 1; i < M; i++)
    {
        /* FIXME: this is redundant copying */
        const int sender_id = i / partition_rows;
        MPI_Bcast(row_buffer, N, MPI_FLOAT, sender_id, MPI_COMM_WORLD);
    }

    /* Let faster workers wait and synchronize before moving on */
    MPI_Barrier(MPI_COMM_WORLD);

    /* Stop timer BEFORE we gather the data */
    if (id == ROOT_ID) end = MPI_Wtime();

    /* Gather (opposite of scatter) */
    MPI_Gather(mat_buffer, partition_size, MPI_FLOAT, MAT, partition_size, MPI_FLOAT, ROOT_ID, MPI_COMM_WORLD);

    /* Verify by the root process in the end */
    if (id == ROOT_ID)
    {
        const double elapsed_time = end - start;

        // Sanity check
        // ref_noswap(MAT);
        print_mat(MAT);

        /* Run verification (if enabled) */
        #ifdef RUN_VERIF
        ref_old_noswap(MAT_B);
        print_mat(MAT_B);
        int errors = verify_ref(MAT, MAT_B);
        printf("MISMATCH=%d\n", errors);
        #endif

        printf("TIME=%.6e s\n", elapsed_time);
    }

    return MPI_Finalize();
}