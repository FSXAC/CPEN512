#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define MAX_N           100000
#define DISTRIBUTE_TAG  2001
#define RETURN_TAG      2002

#define N 4096

float array[MAX_N];
float array2[MAX_N];

/* MPI macros */
#define ROOT_ID 0

int main(int argc, char **argv)
{
    /* Initialize MPI */
    MPI_Init(&argc, &argv);

    /* Get current process ID and total number of processes */
    int id, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &id);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    printf("Current ID: %d of %d total processes\n", id, num_procs);

    /* Split function if we're the orchestrator or the worker */
    if (id == ROOT_ID)
    {
        /* Do orchestrator stuff if we're the root process */
        /* Initialize the array with random values */
        for (int i = 0; i < N; i++)
            // array[i] = (float) rand() / RAND_MAX;
            array[i] = 0.1 * i;


        /* START OF OPERATION */
        clock_t start = clock();

        /* Distribute roughly balanced work to all workers */
        /* Find out the boundary of the array in which we can work on */
        /* We can split the array size equally to amount of processes */
        /*
        * Suppose we have an array of 22 elements and splitting across
        * 4 processes including the root process
        * 
        * Then each process should get 22/4 = 5 elements each
        * But what about the remaining elements
        *
        *  0                   1                   2
        * [0|1|2|3|4|5|6|7|8|9|0|1|2|3|4|5|6|7|8|9|0|1]
        *  ↑       ↑ ↑       ↑ ↑       ↑ ↑       ↑ ↑ ↑
        *  0-------0 1-------1 2-------2 3-------3 1-1
        * 
        */

        int start_index = -1;
        int end_index = -1;
        int split_size = N / num_procs;

        for (int worker_id = 1; worker_id < num_procs; worker_id++)
        {
            /* Then each worker id gets one piece of the pie */
            start_index = worker_id * split_size;
            end_index = (worker_id + 1) * split_size - 1;
            
            /* To worker:
             *  -   1 value of:     number of elements to add
             *  -   n values of:    actual values for those indicies in the array
             */
            MPI_Send(&split_size, 1, MPI_INT, worker_id, DISTRIBUTE_TAG, MPI_COMM_WORLD);
            MPI_Send(&array[start_index], split_size, MPI_FLOAT, worker_id, DISTRIBUTE_TAG, MPI_COMM_WORLD);
        }

        /* Do orchestrator's share of the work */
        float sum = 0;
        for (int i = 0; i < split_size; i++)
        {
            sum += array[i];
        }

        /* NOTE: commented out assuming num process align */
        /* Check if the end index reaches the end, if not then we can just finish the work */
        // if (end_index != -1) {
        //     for (int i = end_index + 1; i < N; i++)
        //     {
        //         sum += array[i];
        //     }
        // }

        /* Collect sum from workers */
        MPI_Status status;
        float partial_sum;
        for (int worker_id = 1; worker_id < num_procs; worker_id++)
        {
            MPI_Recv(&partial_sum, 1, MPI_LONG, MPI_ANY_SOURCE, RETURN_TAG, MPI_COMM_WORLD, &status);
            printf("Received partial sum from worker id %d\n", status.MPI_SOURCE);
            sum += partial_sum;
        }

        /* End of operation */
        clock_t end = clock();
        clock_t elapsed_time = end - start;
        printf("CLOCK=%lu\t%.6f s\n", elapsed_time, (double) elapsed_time / CLOCKS_PER_SEC);

        printf("Total sum is %.4f\n", sum);
    }
    else
    {
        /* Do worker stuff */
        /* Here we are expected to receive MPI signals */
        int split_size;
        MPI_Status status;

        MPI_Recv(&split_size, 1, MPI_INT, ROOT_ID, DISTRIBUTE_TAG, MPI_COMM_WORLD, &status);
        MPI_Recv(&array2, split_size, MPI_FLOAT, ROOT_ID, DISTRIBUTE_TAG, MPI_COMM_WORLD, &status);
          
        /* Calculate the partial sum */
        float partial_sum = 0;
        for (int i = 0; i < split_size; i++)
        {
            partial_sum += array2[i];
        }

        /* Send partial sum back to orchestrator */
        MPI_Send(&partial_sum, 1, MPI_FLOAT, ROOT_ID, RETURN_TAG, MPI_COMM_WORLD);
    }

    /* Close */
    return MPI_Finalize();
}