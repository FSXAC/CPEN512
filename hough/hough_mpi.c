#include "common.h"

#include <mpi.h>

#define ROOT_RANK 0

/* Difference: Since MPI have multiple programs -- only use root 
 * to read and save images
 */
int main(int argc, char **argv) {

    int rank;
    int num_procs;

    /* Initialize MPI */
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    /* Vars */
    uint8_t* bin_image;
    int acc_width = THETA_STEPS;
    int acc_height = 2 * MAX_R;
    float *acc;

    /* Timer */
    double start, end;

    /* Buffer for the work-threads */
    uint8_t *partial_img = malloc(sizeof(uint8_t) * IMG_SIZE * IMG_SIZE);
    float *partial_acc = calloc(acc_width * acc_height, sizeof(float));

    /* Main rank loads image and divide work */
    if (rank == ROOT_RANK)
    {
        /* MPI Check */
        if (IMG_SIZE % num_procs != 0)
        {
            printf("Uneven workload to distribute.\n");
            MPI_Finalize();
            exit(1);
        }
        else
        {
            printf("%d threads ready to work.\n", num_procs);
        }

        /* Read image */
        int width, height, bpp;
        bin_image = stbi_load(IMG_FILE, &width, &height, &bpp, 1);
        printf("Image size: %d px by %d px, bpp: %d\n", width, height, bpp);
        
        if (width != IMG_SIZE || height != IMG_SIZE)
        {
            printf("Error! invalid image size\n");
            return 1;
        }

        /* Set up accumulator */
        acc = (float *) malloc(sizeof(float) * acc_height * acc_width);

        printf("transforming to %d by %d acc...\n", acc_width, acc_height);
    }

    /* Divide up the work */
    int partition_row = IMG_SIZE / num_procs;
    int partition_size = IMG_SIZE * partition_row;
    MPI_Scatter(bin_image, partition_size, MPI_UNSIGNED_CHAR, partial_img, partition_size, MPI_UNSIGNED_CHAR, ROOT_RANK, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);

    /* Root starts the clock */
    if (rank == ROOT_RANK)
        start = MPI_Wtime();

    for (int j = 0; j < partition_row; j++)
    {
        for (int u = 0; u < IMG_SIZE; u++)
        {

            /* Offet 'v' for partition row */
            int v = j + (rank * partition_row);

            /* For each theta in accumulator */
            for (int col = 0; col < acc_width; col++)
            {
                float theta = MIN_THETA + (D_THETA * col);
                float rho = v * sin(theta) + u * cos(theta);
                int row = floor(rho - MIN_R);

                /* Cast vote */
                if (row >= 0 && row < acc_height)
                    GETACC(partial_acc, row, col) += GETIM(partial_img, u, v) > THRESHOLD;
            }
        }
    }

    /* Synchronize */
    MPI_Barrier(MPI_COMM_WORLD);
    
    /* Stop timer */
    end = MPI_Wtime();

    /* Reduce all the partial results to the same summed accumulator */
    MPI_Reduce(
        partial_acc,
        acc,
        acc_width * acc_height,
        MPI_FLOAT,
        MPI_SUM,
        ROOT_RANK,
        MPI_COMM_WORLD
    );

    /* Root cleans up */
    if (rank == ROOT_RANK)
    {
        /* Execution time */
        printf("Execution time (MPI with n=%d): %.6f s\n", num_procs, end - start);

        /* Normalize and out */
        uint8_t* out_acc = (uint8_t *) malloc(sizeof(uint8_t) * acc_height * acc_width);
        printf("normalizing and copying output...\n");
        float max = normalize_image(acc, out_acc, acc_width, acc_height);
        printf("maximum value in acc: %.1f\n", max);

        /* Write out image to file */
        stbi_write_jpg("out_mpi.jpg", acc_width, acc_height, 1, out_acc, 90);

        /* Close image */
        stbi_image_free(bin_image);

        /* Free memory */
        free(out_acc);
        free(acc);
    }

    MPI_Finalize();

    free(partial_img);
    free(partial_acc);

    return 0;
}
