#include <mpi.h>
#include <stdio.h>

#define MAX_N           100000
#define DISTRIBUTE_TAG  2001
#define RETURN_TAG      2002

float array[MAX_N];
float array2[MAX_N];

/* MPI macros */
#define ROOT_ID 0

main(int argc, char **argv)
{
    /* Initialize MPI */
    MPI_Init(&argc, &argv);

    /* Get current process ID and total number of processes */
    int id, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &id);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    /* Split function if we're the orchestrator or the worker */
    if (id == ROOT_ID)
    {
        /* Do orchestrator stuff if we're the root process */

        /* Ask the user for input size */
        int num_size;
        printf("please enter the number of numbers to sum: ");
        scanf("%i", &num_size);
        if (num_size > MAX_N)
        {
            printf("N too large!\n");
            exit(1);
        } 

        /* Initialize the array with random values */
        for (int i = 0; i < num_size; i++)
            array[i] = (float) rand() / RAND_MAX;

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

        int start_index;
        int end_index;
        int split_size = num_size / num_procs;

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

        /* Check if the end index reaches the end, if not then we can just finish the work */
        for (int i = end_index + 1; i < num_size; i++)
        {
            sum += array[i];
        }

        /* Collect sum from workers */
        MPI_Status status;
        float partial_sum;
        for (int worker_id = 1; worker_id < num_procs; worker_id++)
        {
            MPI_Recv(&partial_sum, 1, MPI_LONG, MPI_ANY_SOURCE, RETURN_TAG, MPI_COMM_WORLD, &status);
            printf("Received partial sum from worker id %d\n", status.MPI_SOURCE);
            sum += partial_sum;
        }

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
        MPI_SEND(&partial_sum, 1, MPI_FLOAT, ROOT_ID, RETURN_TAG, MPI_COMM_WORLD);
    }

    /* Close */
    MPI_Finalize();
    
}