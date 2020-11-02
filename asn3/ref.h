#ifndef REF_H
#define REF_H

#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>

// #define RUN_VERIF
// #define DEBUG_PRINT
// #define DEBUG_GPU

#define BLOCK_SIZE 1024

#ifndef M
#define M 4096
#endif 

#ifndef N
#define N M
#endif

float *MAT;
float *MAT_B;

/* Array access macro */
#define GET(A, row, col) A[row * N + col]
#define MIN(A, B) ((A > B) ? B : A)

/* Prints matrix */
void print_mat(float *A)
{
    #ifdef DEBUG_PRINT
    int i, j;
    for (i = 0; i < MIN(M, 20); i++, printf("\n"))
        for (j = 0; j < MIN(N, 20); j++)
        {
            // printf("%d %d / %d %d\n", i, j, M, N);
            if (i == j) printf("\033[0;32m");
            printf("%6.1f", GET(A, i, j));
            if (i == j) printf("\033[0m");
        }

    printf("\n");
    #endif
}

/* This initializes the A array with size MxN with random integers casted as float */
void init_array(float *A)
{
    for (int row = 0; row < M; row++)
        for (int col = 0; col < N; col++)
        {
            /* Make sure (1,1) element is never 0 */
            // GET(A, row, col) = (float) (rand() % 3 - 1);
            GET(A, row, col) = (float) 0.1 * (rand() % 400 );
        }
}

/* Find the row index of value with highest index */
int find_max_row_index(float *A, int h, int k)
{
    int i_max = h;
    float val = fabs(GET(A, h, k));
    float i_max_val = val;

    /* Loop through each row */
    for (int i = h; i < M; i++)
    {
        val = fabs(GET(A, i, k));
        if (val > i_max_val)
        {
            i_max_val = val;
            i_max = i;
        }
    }

    return i_max;
}

/* Swap two rows */
void swap_rows(float *A, int row1, int row2)
{
    for (int i = 0; i < N; i++)
    {
        float tmp = GET(A, row1, i);
        GET(A, row1, i)= GET(A, row2, i);
        GET(A, row2, i) = tmp;
    }
}

/* Normalize the row */
void normalize_row(float *A, int row, int start_col)
{
    float f = GET(A, row, start_col);

    for (int j = start_col; j < N; j++)
        GET(A, row, j) /= f;
}

/* Cleaned up ref function */
void ref(float *A)
{
    int h = 0;  /* pivot row */
    int k = 0;  /* pivot col */

    while (h < M && k < N)
    {
        /* Find the row index with the highest mag */
        int i_max = find_max_row_index(A, h, k);

        if (GET(A, i_max, k) != 0.0)
        {
            /* Swap current pivot row with imax row */
            swap_rows(A, h, i_max);

            /* Make the left most number in current row 1 */
            normalize_row(A, h, k);

            /* Update everything under pivot */
            for (int i = h + 1; i < M; i++)
            {
                /* Find the factor to multiply so that it becomes 0 */
                float f = GET(A, i, k) / GET(A, h, k);

                /* Set the first num to 0 to reduce computation */
                GET(A, i, k) = 0.0;

                /* Subtract f times the first row */
                for (int j = k + 1; j < N; j++)
                    GET(A, i, j) -= f * GET(A, h, j);
            }

            /* Update pivot */
            h++;
            k++;
        }
        else k++;
    }
}

void ref_noswap(float *A)
{
    int h = 0;  /* pivot row */
    int k = 0;  /* pivot col */

    while (h < M && k < N)
    {
        if (GET(A, h, k) != 0.0)
        {
            /* Make the left most number in current row 1 */
            normalize_row(A, h, k);

            /* Update everything under pivot */
            for (int i = h + 1; i < M; i++)
            {
                /* Find the factor to multiply so that it becomes 0 */
                float f = GET(A, i, k) / GET(A, h, k);

                /* Set the first num to 0 to reduce computation */
                GET(A, i, k) = 0.0;

                /* Subtract f times the first row */
                for (int j = k + 1; j < N; j++)
                    GET(A, i, j) -= f * GET(A, h, j);
            }

            /* Update pivot */
            h++;
            k++;
        }
        else k++;
    }
}

/* This is used as a backup/checker */
void ref_old(float *A)
{
    int h = 0, k = 0;
    while (h < M && k < N)
    {
        int i_max = h;
        float i_max_val = GET(A, h, k);
        for (int i = h; i < M; i++)
        {
            if (fabs(GET(A, i, k)) > i_max_val)
            {
                i_max_val = fabs(GET(A, i, k));
                i_max = i;
            }
        }

        // Pivot
        if (GET(A, i_max, k) == 0.0) k++;
        else
        {
            // Swap rows (2D array impl requires loop)
            for (int i = 0; i < N; i++)
            {
                float tmp = GET(A, i_max, i);
                GET(A, i_max, i) = GET(A, h, i);
                GET(A, h, i) = tmp;
            }

            float f = GET(A, h, k);
            for (int j = k; j < N; j++) GET(A, h, j) /= f;

            // For each row below pivot reduce
            for (int i = h + 1; i < M; i++)
            {
                float f = GET(A, i, k) / GET(A, h, k);
                GET(A, i, k) = 0.0;

                // For each row apply same operation
                for (int j = k + 1; j < N; j++)
                    GET(A, i, j) -= GET(A, h, j) * f;
            }

            // Increment pivot
            h++;
            k++;
        }
    }
}

void ref_old_noswap(float *A)
{
    int h = 0, k = 0;
    while (h < M && k < N)
    {
        // Pivot
        if (GET(A, h, k) == 0.0) k++;
        else
        {
            float f = GET(A, h, k);
            for (int j = k; j < N; j++) GET(A, h, j) /= f;

            // For each row below pivot reduce
            for (int i = h + 1; i < M; i++)
            {
                float f = GET(A, i, k) / GET(A, h, k);
                GET(A, i, k) = 0.0;

                // For each row apply same operation
                for (int j = k + 1; j < N; j++)
                    GET(A, i, j) -= GET(A, h, j) * f;
            }

            // Increment pivot
            h++;
            k++;
        }

        // print_mat(A);
    }
}

/* Floating point comparison */
int nearlyEqual(float a, float b) {
    return (fabs(a - b) < 0.005);
}

#define PRINT_RED(X) printf("\033[0;31m%6.2f\033[0m", X);
// #define PRINT_GREEN(X) printf("\033[0;32m%6.2f\033[0m", X);
// #define PRINT_RED(X) printf("%6.2f", X);
#define PRINT_GREEN(X) printf("%6.2f", X);

/* This varifies the answer */
/* A is to be tested, B is reference */
int verify_ref(float *A, float *B)
{
    int done = 0;
    for (int i = 1; i < M && !done; i++) {
        for (int j = 0; j < i && !done; j++)
        {
            if (GET(A, i, j) != 0.0)
            {
                printf("Not in REF!\n");
                done = 1;
            }
        }
    }

    /* Print diff and count errors */
    int errors = 0;
    
    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < N; j++)
        {
            if (!nearlyEqual(GET(A, i, j), GET(B, i, j))) {
                #ifdef DEBUG_PRINT
                PRINT_RED(GET(A, i, j));
                #endif

                errors++;
            }
            else if (i == j)
            {
                #ifdef DEBUG_PRINT
                PRINT_GREEN(GET(A, i, j));
                #endif
            }
            else
            {
                #ifdef DEBUG_PRINT
                printf("%6.2f", GET(A, i, j));
                #endif
            }
        }

        // Print correct answer
        #ifdef DEBUG_PRINT
        printf("\t");
        for (int j = 0; j < N; j++)
            printf("%6.2f", GET(B, i, j));
        printf("\n");
        #endif
    }

    #ifdef DEBUG_PRINT
    printf("\n");
    #endif

    return errors;
}

#endif