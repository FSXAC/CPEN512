#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#include "init.h"

void print_mat(float A[][N])
{
    for (int i = 0; i < M; i++, printf("\n"))
        for (int j = 0; j < N; j++)
            printf("%6.1f", A[i][j]);

    printf("\n");
}

void ref_old(float A[][N])
{
    int h = 0, k = 0;

    while (h < M && k < N)
    {
        int i_max = h;
        float i_max_val = A[h][k];

        for (int i = h; i < M; i++)
        {
            if (fabs(A[i][k]) > i_max_val)
            {
                i_max_val = fabs(A[i][k]);
                i_max = i;
            }
        }

        // Pivot
        if (A[i_max][k] == 0.0)
        {
            k++;
        }
        else
        {
            // Swap rows (2D array impl requires loop)
            for (int i = 0; i < N; i++)
            {
                float tmp = A[i_max][i];
                A[i_max][i] = A[h][i];
                A[h][i] = tmp;
            }

            float f = A[h][k];
            for (int j = k; j < N; j++)
            {
                A[h][j] /= f;
            }

            // For each row below pivot reduce
            for (int i = h + 1; i < M; i++)
            {
                float f = A[i][k] / A[h][k];
                A[i][k] = 0.0;

                // For each row apply same operation
                for (int j = k + 1; j < N; j++)
                {
                    A[i][j] -= A[h][j] * f;
                }
            }

            // Increment pivot
            h++;
            k++;
        }
    }
}

int main(void)
{
    init_array(MAT);
    // print_mat(A);


    clock_t start = clock();
    ref(MAT);
    clock_t end = clock();

    // print_mat(A);

    clock_t elapsed_time = end - start;
    printf("CLOCK=%lu\t%.6f s\n", elapsed_time, (double) elapsed_time / CLOCKS_PER_SEC);
}