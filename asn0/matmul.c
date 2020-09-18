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
#define MATMUL_NAIVE
#endif

// What is the size of matrix in benchmark
#if !defined(N_SIZE)
#define N_SIZE 256
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
#define TILE_SIZE 724
#endif

/* Helpers */
#define MIN(x, y) ((x > y) ? y : x)					/* Returns the minimum of (x, y) */
#define GET(X, row, col) X[row * N_SIZE + col]		/* Refernces the element in 1D array using 2D coords */

/* Initializes the memory of a matrix with random values
 * Returns: the type pointer of the 2D matrix rolled into a 1D array
 */
t* mat_init (void) {
	int idx = 0;
	t* matrix = malloc(sizeof(t) * N_SIZE * N_SIZE);

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

t* mat_init_zero(void) {
	int idx = 0;
	t* matrix = calloc(N_SIZE * N_SIZE, sizeof(t));
	return matrix;
}


#if defined(MATMUL_NAIVE)
/* Naive/triple-loop matrix multiplication
 */
void matmul(t* A, t* B, t* C)
{
	int row, col, i;
	for (row = 0; row < N_SIZE; row++) {
		for (col = 0; col < N_SIZE; col++) {
			for (i = 0; i < N_SIZE; i++) {
				GET(C, row, col) += GET(A, row, i) * GET(B, i, col);
			}
		}
	}
}

#elif defined(MATMUL_TILED)


/* Manual optimization 1: Tiled matrix multiplication
 */
void matmul(t* A, t* B, t* C)
{
	int II2, II3, i1, i2, i3;
	for (II2 = 0; II2 < N_SIZE; II2 += TILE_SIZE) {
		for (II3 = 0; II3 < N_SIZE; II3 += TILE_SIZE) {
			for (i1 = 0; i1 < N_SIZE; i1++) {
				for (i2 = II2; i2 < MIN(II2 + TILE_SIZE, N_SIZE); i2++) {
					for (i3 = II3; i3 < MIN(II3 + TILE_SIZE, N_SIZE); i3++) {
						GET(C, i1, i3) += GET(A, i1, i2) * GET(B, i2, i3);
					}
				}
			}
		}
	}
}

#elif defined(MATMUL_DO2)

/* Manual optimization 2a: manual unrolling (size 2)
 */
void matmul(t* A, t* B, t* C)
{
	int row, col, i;
	for (row = 0; row < N_SIZE; row++) {
		for (col = 0; col < N_SIZE; col++) {
			for (i = 0; i < N_SIZE; i += 2) {
				GET(C, row, col) += GET(A, row, i) * GET(B, i, col) + 
									GET(A, row, i + 1) * GET(B, i + 1, col);
			}
		}
	}
}

#elif defined(MATMUL_DO4)
/* Manual optimization 2b: manual unrolling (size 4)
 */
void matmul(t* A, t* B, t* C)
{
	int row, col, i;
	for (row = 0; row < N_SIZE; row++) {
		for (col = 0; col < N_SIZE; col++) {
			for (i = 0; i < N_SIZE; i += 4) {
				GET(C, row, col) += GET(A, row, i) * GET(B, i, col) +
									GET(A, row, i + 1) * GET(B, i + 1, col) +
									GET(A, row, i + 2) * GET(B, i + 2, col) +
									GET(A, row, i + 3) * GET(B, i + 3, col);
			}
		}
	}
}

#elif defined(MATMUL_DO8)
/* Manual optimization 2c: manual unrolling (size 8)
 */
void matmul(t* A, t* B, t* C)
{
	int row, col, i;
	for (row = 0; row < N_SIZE; row++) {
		for (col = 0; col < N_SIZE; col++) {
			for (i = 0; i < N_SIZE; i += 8) {
				GET(C, row, col) += GET(A, row, i) * GET(B, i, col) +
									GET(A, row, i + 1) * GET(B, i + 1, col) +
									GET(A, row, i + 2) * GET(B, i + 2, col) +
									GET(A, row, i + 3) * GET(B, i + 3, col) +
									GET(A, row, i + 4) * GET(B, i + 4, col) +
									GET(A, row, i + 5) * GET(B, i + 5, col) +
									GET(A, row, i + 6) * GET(B, i + 6, col) +
									GET(A, row, i + 7) * GET(B, i + 7, col) +
			}
		}
	}
}

#endif

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
	C = mat_init_zero();

	#ifdef PRINT_MATRIX
	print_mat(A);
	print_mat(B);
	#endif

	clock_t start = clock();
	matmul(A, B, C);
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

	#if defined(MATMUL_TILED)
	printf("MATMUL_TILED: TILE_SIZE = %d\n", TILE_SIZE);
	#elif defined(MATMUL_DO2)
	printf("MATMUL_DO2\n");
	#elif defined(MATMUL_DO4)
	printf("MATMUL_DO4\n");
	#endif

	clock_t time = matmul_benchmark();
	printf("CLOCK=%lu\t%.6f s\n", time, (double) time / CLOCKS_PER_SEC);

	return 0;
}
