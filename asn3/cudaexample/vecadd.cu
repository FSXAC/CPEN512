// Serialized program

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define N 4096

__global__
void vector_add(float *out, float *a, float *b)
{
    for (int i = 0; i < N; i++)
    {
        out[i] = a[i] + b[i];
    }
}

void vector_add_ref(float *out, float *a, float *b)
{
    for (int i = 0; i < N; i++)
    {
        out[i] = a[i] + b[i];
    }
}

int main(void)
{
    float *a, *b, *out;
    float *x, *y, *z;

    a = (float *) malloc(sizeof(float) * N);
    b = (float *) malloc(sizeof(float) * N);
    out = (float *) malloc(sizeof(float) * N);

    for (int i = 0; i < N; i++) {
        a[i] = 1.0 * i;
        b[i] = 2.0 * i;
    }

    /* Allocate device memory */
    cudaMalloc((void**) &x, sizeof(float) * N);
    cudaMalloc((void**) &y, sizeof(float) * N);
    cudaMalloc((void**) &z, sizeof(float) * N);

    /* Do memory copy */
    cudaMemcpy(x, a, sizeof(float) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(y, b, sizeof(float) * N, cudaMemcpyHostToDevice);

    // Vector add function
    vector_add<<<1, 1>>>(z, x, y);

    // Sync
    cudaDeviceSynchronize();

    /* Copy again */
    cudaMemcpy(out, z, sizeof(float) * N, cudaMemcpyDeviceToHost);

    for (int i = 0; i < 6; i++)
    {
        printf("%8.3f + %8.3f = %8.3f\n", a[i], b[i], out[i]);
    }

    cudaFree(x);
    cudaFree(y);
    cudaFree(z);

    free(a);
    free(b);
    free(out);
}