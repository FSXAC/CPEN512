// #define TEST_MAT
#ifdef TEST_MAT

#define M 3
#define N 4

float A[M][N] = {
    {2.0, 1.0, -1.0, 8},
    {-3.0, -1.0, 2.0, -11},
    {-2.0, 1.0, 2.0, -3}
};

#else

#define M 1024
#define N 1025

float A[M][N];

#endif

/* This initializes the A array with size MxN with random integers casted as float */
void init_array(void)
{
    for (int row = 0; row < M; row++)
        for (int col = 0; col < N; col++)
            A[row][col] = (float) (rand() % 18 - 9);
}