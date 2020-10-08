#include <pthread.h>

#ifdef __APPLE__
#include "pthread_barrier.h"
#endif

#include <time.h>
#include "ref.h"

#define NUM_THREADS 2

/* Pthread struct that contains information about what the thread is going to do */
/* This is passed as arugment to the starting routine of each thread */
struct thread_arg_t {
    /* Pointer to the matrix */
    float *matrix;

    /* These values show the boundary of where the thread is operating on */
    /* [start_index, end_index] <- inclusive */
    int start_index;
    int end_index;

    /* Thread id */
    int tid;

    /* Shared barrier for synchronization */
    pthread_barrier_t *barrier;
};

/* Thread's starting routine */
void *thread_do(void *thread_arg)
{
    /* First we need to obtain the values inside the struct */
    /* By casting the void pointer to a thread arg pointer */
    struct thread_arg_t *args = (struct thread_arg_t *) thread_arg;

    float *matrix = args->matrix;
    int start_index = args->start_index;
    int end_index = args->end_index;
    int tid = args->tid;

    /* Synchronization barrier */
    pthread_barrier_t *barrier = args->barrier;

    #ifdef DEBUG_PRINT
    printf("Thread %d started\n", tid);
    #endif

    /* Loop through all the rows */
    /* (we need to do this since threads can be assumed to be independent) */
    for (int row_index = 0; row_index < M; row_index++)
    {
        int col_index = row_index;

        /* Only manipulate if the row index is within this thread's assignment */
        if (row_index >= start_index && row_index <= end_index)
        {
            /* Partially the same as ref() in ref.h of the serialized case*/
            normalize_row(matrix, row_index, col_index);

            /* Wait for other threads */
            pthread_barrier_wait(barrier);

            /* Also partially reduce every row below (in this thread) */
            for (int i = row_index + 1; i <= end_index; i++)
            {
                /* Find the factor to multiply so that it becomes 0 */
                float f = GET(matrix, i, col_index);

                /* Set the first num to 0 to reduce computation */
                GET(matrix, i, col_index) = 0.0;

                /* Subtract f times the first row */
                for (int j = col_index + 1; j < N; j++)
                    GET(matrix, i, j) -= f * GET(matrix, row_index, j);
            }
        }

        printf("%d: h=%d, k=%d\n", tid, row_index, col_index);
    }

    return NULL;
}

void ref_pthread()
{
    /* Make an array of threads */
    pthread_t threads[NUM_THREADS];

    /* And their corresponding args structs */
    struct thread_arg_t thread_args[NUM_THREADS];

    /* Add a barrier for synchronization */
    pthread_barrier_t barrier;

    /* Initialize barrier */
    pthread_barrier_init(&barrier, NULL, NUM_THREADS);

    /* Make thread attributes (here I'm explicitly saying they need to be joinable) */
    pthread_attr_t thread_attr;

    /* Initialize thread attr with default values */
    pthread_attr_init(&thread_attr);
    
    /* Set joinable */
    pthread_attr_setdetachstate(&thread_attr, PTHREAD_CREATE_JOINABLE);

    /* Make threads */
    for (int t = 0; t < NUM_THREADS; t++)
    {
        printf("Main: creating thread %d\n", t);

        /* Prepare the struct arg */
        thread_args[t].tid = t;
        thread_args[t].start_index = (M / NUM_THREADS) * t;
        thread_args[t].end_index = (M / NUM_THREADS) * (t + 1) - 1;
        thread_args[t].matrix = MAT;

        thread_args[t].barrier = &barrier;

        /* Arguments to creating a new thread:
         * 1. the thread identifier
         * 2. thread attribute (NULL for default attribute)
         * 3. start routine
         * 4. arguments to the routine
         */
        int rc = pthread_create(&threads[t], &thread_attr, thread_do, (void *) &thread_args[t]);

        /* Check return code */
        if (rc)
        {
            printf("Error: non zero return code (%d)\n", rc);
            exit(-1);
        }
    }

    /* Free attribute (no longer need it) */
    pthread_attr_destroy(&thread_attr);

    /* Ending operation: Wait for all threads to finish and join */
    for (int t = 0; t < NUM_THREADS; t++)
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
}

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
    print_mat(MAT);
    #endif

    /* Run ref */
    clock_t start = clock();
    ref_pthread();
    clock_t end = clock();
    double elapsed_time = (double) (end - start) / CLOCKS_PER_SEC;

    /* Run verification (if enabled) */
    #define RUN_VERIF
    #ifdef RUN_VERIF
    ref_old_noswap(MAT_B);
    int errors = verify_ref(MAT, MAT_B);
    printf("MISMATCH=%d\n", errors);
    #endif

    printf("CLOCK=%.6e s\n", elapsed_time);

    /* Make sure we exit pthread */
    pthread_exit(NULL);

    return 0;
}