#include <math.h>

// #define TEST_MAT
#ifdef TEST_MAT

#define M 3
#define N 4

float MAT[M][N] = {
    {2.0, 1.0, -1.0, 8},
    {-3.0, -1.0, 2.0, -11},
    {-2.0, 1.0, 2.0, -3}
};

#else

#define M 1024
#define N 1025

float MAT[M][N];

#endif

/* This initializes the A array with size MxN with random integers casted as float */
inline void init_array(float A[][N])
{
    for (int row = 0; row < M; row++)
        for (int col = 0; col < N; col++)
            A[row][col] = (float) (rand() % 18 - 9);
}


/* Find the row index of value with highest index */
inline int find_max_row_index(float A[][N], int h, int k)
{
    int i_max = h;
    float val = fabs(A[h][k]);
    float i_max_val = val;

    /* Loop through each row */
    for (int i = 0; i < M; i++)
    {
        val = fabs(A[i][k]);
        if (val > i_max_val)
        {
            i_max_val = val;
            i_max = i;
        }
    }

    return i_max;
}

/* Swap two rows */
inline void swap_rows(float A[][N], int row1, int row2)
{
    for (int i = 0; i < N; i++)
    {
        float tmp = A[row1][i];
        A[row1][i] = A[row2][i];
        A[row2][i] = tmp;
    }
}

/* Normalize the row */
inline void normalize_row(float A[][N], int row, int start_col)
{
    float f = A[row][start_col];

    for (int j = start_col; j < N; j++)
        A[row][j] /= f;
}

/* Cleaned up ref function */
void ref(float A[][N])
{
    int h = 0;  /* pivot row */
    int k = 0;  /* pivot col */

    while (h < M && k < N)
    {
        /* Find the row index with the highest mag */
        int i_max = find_max_row_index(A, h, k);

        if (A[i_max][k] != 0.0)
        {
            /* Swap current pivot row with imax row */
            swap_rows(A, h, i_max);

            /* Make the left most number in current row 1 */
            normalize_row(A, h, k);

            /* Update everything under pivot */
            for (int i = h + 1; i < M; i++)
            {
                /* Find the factor to multiply so that it becomes 0 */
                float f = A[i][k] / A[h][k];

                /* Set the first num to 0 to reduce computation */
                A[i][k] = 0.0;

                /* Subtract f times the first row */
                for (int j = k + 1; j < N; j++)
                    A[i][j] -= f * A[h][j];
            }

            /* Update pivot */
            h++;
            k++;
        }
        else k++;
    }
}