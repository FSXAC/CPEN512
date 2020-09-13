#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// Print matrix in output (doesn't affect benchmark)
#define PRINT_MATRIX

// Which type to use for benchmark by default
#if !defined(USE_INT) && !defined(USE_FLOAT) && !defined(USE_DOUBLE)
#define USE_INT
#endif

/* Which matrix multiplcation function to use */
#if !defined(MATMUL_NAIVE) && !defined(MATMUL_TILED)
// #define MATMUL_NAIVE
#define MATMUL_TILED
#endif

// What is the size of matrix in benchmark
#if !defined(N_SIZE)
#define N_SIZE 256
#endif

// How many times to run the benchmark to report an average value
#if !defined(N_TRIALS)
#define N_TRIALS 1
#endif

// Typedef based on which benchmark defined
#if defined(USE_INT)
typedef int t;
#elif defined(USE_FLOAT)
typedef float t;
#elif defined(USE_DOUBLE)
typedef double t;
#endif

/* Tile size for tiled matrix multiplication (override in gcc) theoretical optimal is 512 */
#ifndef TILE_SIZE
#define TILE_SIZE 512
#endif

/* Helpers */
#define MIN(x, y) ((x > y) ? y : x)					/* Returns the minimum of (x, y) */
#define GET(X, row, col) X[row * N_SIZE + col]		/* Refernces the element in 1D array using 2D coords */

/* Initializes the memory of a matrix with random values
 * Returns: the type pointer of the 2D matrix rolled into a 1D array
 */
t* mat_init (void) {
	int idx = 0;
	t* matrix = (t*)malloc(sizeof(t) * N_SIZE * N_SIZE);

	while (idx <  N_SIZE * N_SIZE)
	{
		#if defined(USE_INT)
		matrix[idx++] = rand() % 10;
		#else
		matrix[idx++] = (t) rand() / RAND_MAX;
		#endif
	}

	return matrix;
}

/* Naive/triple-loop matrix multiplication
 */
void matmul(t* A, t* B, t* C)
{
	int row, col, i;
	t sum;

	for (row = 0; row < N_SIZE; row++) {
		for (col = 0; col < N_SIZE; col++) {
			sum = 0;
			for (i = 0; i < N_SIZE; i++) {
				sum += GET(A, row, i) * GET(B, i, col);
			}
			GET(C, row, col) = sum;
		}
	}
}

/* Tiled matrix multiplication
 */
void matmul_tiled(t* A, t* B, t* C)
{
	int I, J, K;
	int i, j, k;
	t sum;

	// Stride by tile
	for (I = 0; I < N_SIZE; I += TILE_SIZE)
	{
		for (J = 0; J < N_SIZE; J += TILE_SIZE)
		{
			for (K = 0; K < N_SIZE; K += TILE_SIZE)
			{

				// Calculate the inner products for each tile
				for (i = I; i < MIN(I + TILE_SIZE, N_SIZE); i++)
				{
					for (j = J; j < MIN(J + TILE_SIZE, N_SIZE); j++)
					{
						sum = 0;
						for (k = K; k < MIN(k + TILE_SIZE, N_SIZE); k++)
						{
							sum += GET(A, i, k) * GET(B, k, j);
						}
						GET(C, i, j) = sum;
					}
				}
			}
		}
	}
}

void print_mat(t* A)
{
	int row = 0;
	int col = 0;

	if (N_SIZE <= 128) {
		printf("Matrix: \n");

		for (row = 0; row < N_SIZE; row++, printf("\n")) {
			for (col = 0; col < N_SIZE; col++) {
				#if defined(USE_INT)
				printf("%6d", A[row * N_SIZE + col]);
				#else
				printf("%6.2f", A[row * N_SIZE + col]);
				#endif
			}
		}
	} else {
		printf("Matrix (first 5 elements): ");

		for (row = 0; row * N_SIZE + col < 5; row++) {
			for (col = 0; row * N_SIZE + col < 5; col++) {

				#if defined(USE_INT)
				printf("%6d", A[row * N_SIZE + col]);
				#else
				printf("%6.2f", A[row * N_SIZE + col]);
				#endif
			}
		}
		printf(" ... (truncated)\n");
	}
}

clock_t matmul_benchmark()
{
	t *A, *B, *C;
	
	A = mat_init();
	B = mat_init();
	C = mat_init();

	#ifdef PRINT_MATRIX
	print_mat(A);
	print_mat(B);
	#endif

	clock_t start = clock();
	#if defined(MATMUL_NAIVE)
	matmul(A, B, C);
	#elif defined(MATMUL_TILED)
	matmul_tiled(A, B, C);
	#endif
	clock_t end = clock();

	#ifdef PRINT_MATRIX
	print_mat(C);
	#endif

	free(A);
	free(B);
	free(C);

	return end - start;
}

int main(void) {
	if (N_SIZE > 4096)
	{
		printf("N larger than 4096 not supported.\n");
		return 1;
	}

	printf("N_SIZE = %d\n", N_SIZE);
	#if defined(__APPLE__)
	printf("CLOCK_PER_SEC = %d\n", CLOCKS_PER_SEC);
	#elif __linux__
	printf("CLOCK_PER_SEC = %lu\n", CLOCKS_PER_SEC);
	#endif

	if (N_TRIALS > 1) {
		clock_t* times = malloc(sizeof(clock_t) * N_TRIALS);
		
		for (int i = 0; i < N_TRIALS; i++) {
			times[i] = matmul_benchmark();
			printf("CLOCK=%lu\n", times[i]);
		}

		// Compute average
		clock_t sum = 0;
		for (int i = 0; i < N_TRIALS; i++) {
			sum += times[i];
		}

		printf("AVERAGE=%lu\n", sum / N_TRIALS);

	} else {
		clock_t time = matmul_benchmark();
		printf("CLOCK=%lu\n", time);
	}

	return 0;
}
