#include <stdio.h>
#include "vbx.h"
#include "ref.h"

const int num_elements=10;

#define NONE 0

int main()
{

#if VBX_SIMULATOR==1
	//initialize with 4 lanes,and 64kb of sp memory
	//word,half,byte fraction bits 16,15,4 respectively
	vbxsim_init( 4, 64, 256,6,5, 4 , 0, 0);
#endif

	//Allocate vectors in scratchpad
	vbx_word_t* a = vbx_sp_malloc( num_elements*sizeof(vbx_word_t) );
	vbx_word_t* b = vbx_sp_malloc( num_elements*sizeof(vbx_word_t) );
	vbx_word_t* c = vbx_sp_malloc( num_elements*sizeof(vbx_word_t) );

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

}

int main(void)
{
    /* Time keeping */
    struct timeval begin, end;
    double time_serial;

    /* Malloc matrices */
    MAT   = (float *) malloc(sizeof(float) * N * M);
    MAT_B = (float *) malloc(sizeof(float) * N * M);

    /* Initialize both matrices */
    printf("(M x N)=(%d x %d)\n", M, N);
    init_array(MAT);
    memcpy(MAT_B, MAT, sizeof(float) * N * M);
    print_mat(MAT);

    /* Run serial */

    /* Run parallel */
    printf("Running parallel . . .\n");
    ref_cuda(MAT);
    double time_parallel = ref_cuda(MAT);

    /* Run verification (if enabled) */
    #ifndef CUDA_ONLY
    #ifdef RUN_VERIF
    printf("Running verification . . .\n");
    int errors = verify_ref(MAT, MAT_B);
    printf("MISMATCH=%d\n", errors);
    #endif
    #endif

    #ifndef CUDA_ONLY
    printf("SERIAL TIME=%.6e s\n", time_serial);
    #endif
    
    printf("PARALL TIME=%.6e s\n", time_parallel);

    return 0;
}
