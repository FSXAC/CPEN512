#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define PRINT_MATRIX

#if !defined(USE_INT) && !defined(USE_FLOAT) && !defined(USE_DOUBLE)
#define USE_INT
#endif

#if !defined(N_SIZE)
#define N_SIZE 4096
#endif

#if defined(USE_INT)
typedef int t;
#elif defined(USE_FLOAT)
typedef float t;
#elif defined(USE_DOUBLE)
typedef double t;
#endif

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

void matmul(t* A, t* B, t* C)
{
	int row, col, i;
	t sum;

	for (row = 0; row < N_SIZE; row++) {
		for (col = 0; col < N_SIZE; col++) {
			sum = 0;
			for (i = 0; i < N_SIZE; i++) {
				sum += A[row * N_SIZE + i] * B[i * N_SIZE + col];
			}
			
			C[row * N_SIZE + col] = sum;
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
		printf("Matrix (first 5 elements): \n");

		for (row = 0; row * N_SIZE + col < 5; row++, printf("\n")) {
			for (col = 0; row * N_SIZE + col < 5; col++) {

				#if defined(USE_INT)
				printf("%6d", A[row * N_SIZE + col]);
				#else
				printf("%6.2f", A[row * N_SIZE + col]);
				#endif
			}
		}
		printf("... (truncated)\n");
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
	printf("N_SIZE = %d\n", N_SIZE);
	#if defined(__APPLE__)
	printf("CLOCK_PER_SEC = %d\n", CLOCKS_PER_SEC);
	#elif __linux__
	printf("CLOCK_PER_SEC = %lu\n", CLOCKS_PER_SEC);
	#endif
	clock_t time = matmul_benchmark();
	printf("\nCLOCK=%lu\n", time);

	return 0;
}
