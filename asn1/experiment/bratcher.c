#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define M 1024
#define N 1024

float MAT[M][N];
float MAT_B[M][N];

/* MPI macros */
#define ROOT_ID 0

/* This initializes the A array with size MxN with random integers casted as float */
void init_array(float A[][N], float A_backup[][N])
{
    for (int row = 0; row < M; row++)
        for (int col = 0; col < N; col++)
        {
            A[row][col] = (float) (rand() % 18 - 9);
            A_backup[row][col] = A[row][col];
        }
}

/* Assuming number of process > 1 and number of processes matches with array row size */
int main(int argc, char **argv)
{
    /* Initialize matrices */
    init_array(MAT, MAT_B);

    /* Initialize MPI */
    MPI_Init(&argc, &argv);

    int id, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &id);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    printf("Current ID: %d of %d total processes\n", id, num_procs);

    if (id == ROOT_ID)
    {

    }
    else
    {

    }

    /* Close */
    return MPI_Finalize();
}