#include <pthread.h>
#include <time.h>

#include "ref.h"

#define NUM_THREADS 4

void *thread_do(void *thread_id)
{
    long tid;
    tid = (long) thread_id;

    printf("Hello and goodbye! It's me, thread #%ld!\n", tid);
    pthread_exit(NULL);
}

int main(void)
{
    /* Pthread stuff */
    /* Make an array of threads */
    pthread_t threads[NUM_THREADS];

    /* Make thread attributes (here I'm explicitly saying they need to be joinable) */
    pthread_attr_t thread_attr;

    /* Initialize thread attr with default values */
    pthread_attr_init(&thread_attr);
    
    /* Set joinable */
    pthread_attr_setdetachstate(&thread_attr, PTHREAD_CREATE_JOINABLE);

    /* Make threads */
    for (long t = 0; t < NUM_THREADS; t++)
    {
        printf("Main: creating thread %ld\n", t);

        /* Arguments to creating a new thread:
         * 1. the thread identifier
         * 2. thread attribute (NULL for default attribute)
         * 3. start routine
         * 4. arguments to the routine
         */
        int rc = pthread_create(&threads[t], &thread_attr, thread_do, (void *) t);

        /* Check return code */
        if (rc)
        {
            printf("Error: non zero return code (%d)\n", rc);
            exit(-1);
        }
    }

    /* Free attribute (no longer need it) */
    pthread_attr_destroy(&thread_attr);

    /* Wait for all threads to finish and join */
    for (long t = 0; t < NUM_THREADS; t++)
    {
        void *status;
        int rc = pthread_join(threads[t], &status);

        if (rc)
        {
            printf("Error: non zero return code while joining (%d)\n", rc);
            exit(-1);
        }

        printf("%d joined!\n", t);
    }

    /* Make sure we exit pthread */
    // pthread_exit(NULL);

    // return 0;

    /* Old stuff */
    /* Malloc matrices */
    MAT   = malloc(sizeof(float) * N * M);
    MAT_B = malloc(sizeof(float) * N * M);

    printf("(M x N)=(%d x %d)\n", M, N);

    #ifdef TEST_MAT
    print_mat(MAT);
    #else
    init_array(MAT, MAT_B);
    print_mat(MAT);
    #endif

    /* Run ref */
    clock_t start = clock();
    ref_noswap(MAT);
    clock_t end = clock();
    double elapsed_time = (double) (end - start) / CLOCKS_PER_SEC;

    // #ifdef TEST_MAT
    print_mat(MAT);
    // #endif

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