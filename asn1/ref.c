#include "ref.h"
#include <time.h>

int main(void)
{
    /* Malloc matrices */
    MAT   = malloc(sizeof(float) * N * M);
    MAT_B = malloc(sizeof(float) * N * M);

    printf("(M x N)=(%d x %d)\n", M, N);

    #ifdef TEST_MAT
    print_mat(MAT);
    #else
    init_array(MAT, MAT_B);
    #endif

    /* Run ref */
    clock_t start = clock();
    ref_noswap(MAT);
    clock_t end = clock();
    clock_t elapsed_time = (double) (end - start) / CLOCKS_PER_SEC;

    #ifdef TEST_MAT
    print_mat(MAT);
    #endif

    /* Run verification (if enabled) */
    #define RUN_VERIF
    #ifdef RUN_VERIF
    ref_old_noswap(MAT_B);
    int errors = verify_ref(MAT, MAT_B);
    printf("MISMATCH=%d\n", errors);
    #endif

    printf("CLOCK=%.6e s\n", elapsed_time);

    return 0;
}