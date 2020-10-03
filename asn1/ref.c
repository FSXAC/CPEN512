#include "ref.h"

void print_mat(float A[][N])
{
    for (int i = 0; i < M; i++, printf("\n"))
        for (int j = 0; j < N; j++)
            printf("%6.1f", A[i][j]);

    printf("\n");
}

main(void)
{
    printf("(M x N)=(%d x %d)\n", M, N);

    #ifdef TEST_MAT
    print_mat(MAT);
    #else
    init_array(MAT, MAT_B);
    #endif

    /* Run ref */
    clock_t start = clock();
    ref(MAT);
    clock_t end = clock();
    clock_t elapsed_time = end - start;

    #ifdef TEST_MAT
    print_mat(MAT);
    #endif

    /* Run verification (if enabled) */
    #define RUN_VERIF
    #ifdef RUN_VERIF
    ref_old(MAT_B);
    int errors = verify_ref(MAT, MAT_B);
    printf("MISMATCH=%d\n", errors);
    #endif

    printf("CLOCK=%lu\t%.6f s\n", elapsed_time, (double) elapsed_time / CLOCKS_PER_SEC);
}