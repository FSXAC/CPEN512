#ifndef MM_H
#define MM_H

#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>

#define RUN_VERIF
#define DEBUG_PRINT

#ifndef N
#define N 3
#endif

/* Matrices */
int *A;
int *B;
int *C;
int *C_serial;

/* Array access macro */
#define GET(A, row, col) A[row * N + col]
#define MIN(A, B) ((A > B) ? B : A)

/* Prints matrix */
void print_mat(int *A)
{
#ifdef DEBUG_PRINT
    int i, j;
    for (i = 0; i < MIN(N, 20); i++, printf("\n"))
        for (j = 0; j < MIN(N, 20); j++)
        {
            if (i == j)
                printf("\033[0;32m");
            printf("%4d", GET(A, i, j));
            if (i == j)
                printf("\033[0m");
        }

    printf("\n");
#endif
}

/* This initializes the A array with size MxN with random integers casted as float */
void init_array(int *A)
{
    for (int row = 0; row < N; row++)
        for (int col = 0; col < N; col++)
            GET(A, row, col) = rand() % 3 - 1;
}

/* This performs the basic tiled matrix multiplication */
void mm(int *A, int *B, int *C, int n, int tile_size)
{
    int II2, II3, i1, i2, i3;
    for (II2 = 0; II2 < n; II2 += tile_size)
        for (II3 = 0; II3 < n; II3 += tile_size)
            for (i1 = 0; i1 < n; i1++)
                for (i2 = II2; i2 < MIN(II2 + tile_size, n); i2++)
                    for (i3 = II3; i3 < MIN(II3 + tile_size, n); i3++)
                        GET(C, i1, i3) += GET(A, i1, i2) * GET(B, i2, i3);
}

// #define PRINT_RED(X) printf("\033[0;31m%6.2f\033[0m", X);
// #define PRINT_GREEN(X) printf("\033[0;32m%6.2f\033[0m", X);
#define PRINT_RED(X) printf("%4d", X);
#define PRINT_GREEN(X) printf("%4d", X);

/* This varifies the answer */
/* A is to be tested, B is reference */
int verify_mm(int *A, int *B)
{
    /* Print diff and count errors */
    int errors = 0;
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            if (GET(A, i, j) == GET(B, i, j))
            {
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
                printf("%4d", GET(A, i, j));
#endif
            }
        }

// Print correct answer
#ifdef DEBUG_PRINT
        printf("\t");
        for (int j = 0; j < N; j++)
            printf("%4d", GET(B, i, j));
        printf("\n");
#endif
    }

#ifdef DEBUG_PRINT
    printf("\n");
#endif

    return errors;
}

#endif
