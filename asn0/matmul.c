#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// #define USE_INT
// #define USE_FLOAT
// #define USE_DOUBLE

#if defined(USE_INT)
typedef int t;
#elif defined(USE_FLOAT)
typedef float t;
#elif defined(USE_DOUBLE)
typedef double t;
#endif

t* mat_init (int N) {
	int idx = 0;
	t* matrix = (t*)malloc(sizeof(t) * N * N);

	while (idx <  N * N)
	{
		#if defined(USE_INT)
		matrix[idx++] = rand() % 10;
		#else
		matrix[idx++] = (t) rand() / RAND_MAX;
		#endif
	}

	return matrix;
}

void matmul(t* A, t* B, t* C, int N)
{
	int row, col, i;
	t sum;

	for (row = 0; row < N; row++) {
		printf("%d\n", row);
		for (col = 0; col < N; col++) {
			sum = 0;

			for (i = 0; i < N; i++) {
				t x = A[row * N + i];
				t y = B[i * N + col];
				sum += x * y;
			}

			C[row * N + col] = sum;
		}
	}
}

void print_mat(t* A, int N)
{
	int row = 0;
	int col = 0;

	printf("Matrix: \n");

	for (row = 0; row < N; row++, printf("\n")) {
		for (col = 0; col < N; col++) {
			#if defined(USE_INT)
			printf("%6d", A[row * N + col]);
			#else
			printf("%6.2f", A[row * N + col]);
			#endif
		}
	}
}

clock_t matmul_benchmark(int N)
{
	t *A, *B, *C;
	
	A = mat_init(N);
	B = mat_init(N);
	C = mat_init(N);

	#ifdef DEBUG_PRINT
	print_mat(A, N);
	print_mat(B, N);
	#endif

	clock_t start = clock();
	matmul(A, B, C, N);
	clock_t end = clock();

	#ifdef DEBUG_PRINT
	print_mat(C, N);
	#endif

	return end - start;
}

int main(void) {
	int n;

	printf("Enter size (N): ");
	scanf("%d", &n);

	clock_t time = matmul_benchmark(n);
	printf("It took %lu clock ticks\n", time );

	return 0;
}
