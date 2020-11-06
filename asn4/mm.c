#include <stdio.h>
#include "vbx.h"
#include "mm.h"

#define NONE 0

double mm_vbx(float *A, float *B, float *C, int n)
{
#if VBX_SIMULATOR==1
    /* Initialize the simulator */
    vbxsim_init(
        4,      /* number of lanes */
        64,     /* scratchpad memory capacity (kB) */
        256,    /* maximum masked waves */
        6,      /* fxp word frac bits */
        5,      /* fxp half frac bits */
        4,      /* fxp byte frac bits */
        0,      /* unpopulated ALU lanes */
        0       /* unpopulated multiplier lanes */
    )
#endif

	/* Allocate vectors on the scratchpad */
	vbx_word_t* a = vbx_sp_malloc(N * N * sizeof(vbx_word_t));
	vbx_word_t* b = vbx_sp_malloc(N * N * sizeof(vbx_word_t));
	vbx_word_t* c = vbx_sp_malloc(N * N * sizeof(vbx_word_t));

	//Set vector length, then compute 4*[1,2,3,...,10]
	vbx_set_vl(num_elements);
	vbx(SEW, VADD, a, 2, NONE); //a = [1,2,3,...,10]
	vbx(SVW, VMOV, b, 4, NONE); //b = [4,4,....,4]
	vbx(VVW, VMUL, c, a, b); //c = a * b

	//wait for all vector instructions to finish
	vbx_sync();

	//print out vector c
	int i;
	for( i=0; i<num_elements; i++ ) {
		printf( "%6d ", a[i] );
	}
	printf( "\n" );
	// vbxsim_print_stats();
	return 0;
}

int main(void)
{
    /* Time keeping */
    struct timeval begin, end;
    double time_serial;
    double time_parallel;

    /* Initalize matrices */
    srand(0);
    A = (float *) malloc(sizeof(float) * N * N);
    B = (float *) malloc(sizeof(float) * N * N);
    C = (float *) malloc(sizeof(float) * N * N);
    C_serial = (float *) malloc(sizeof(float) * N * N);
    init_array(A);
    init_array(B);

    /* Run serial */
    mm(A, B, C_serial, N, 8);

    /* Run parallel */
    time_parallel = mm_vbx(A, B, C, N);

    /* Verify */
    #ifdef RUN_VERIF
    int errors = verify_mm(C, C_serial);
    printf("Mismatches: %d/%d\n", errors, N * N);
    #endif

    /* Print results */
    printf("Serial time: %.6e s\n", time_serial);
    printf("Parallel time: %.6e s\n", time_parallel);

    return 0;
}
