// Serialized program

#include <stdio.h>
#include <stdlib.h>

#define N 4096

void vector_add(float *out, float *a, float *b)
{
    for (int i = 0; i < N; i++)
    {
        out[i] = a[i] + b[i];
    }
}

int main(void)
{
    float *a, *b, *out;

    a = (float *) malloc(sizeof(float) * N);
    b = (float *) malloc(sizeof(float) * N);
    out = (float *) malloc(sizeof(float) * N);

    for (int i = 0; i < N; i++) {
        a[i] = 1.0 * i;
        b[i] = 2.0 * i;
    }

    // Vector add function
    vector_add(out, a, b);

    for (int i = 0; i < 6; i++)
    {
        printf("%8.3f + %8.3f = %8.3f\n", a[i], b[i], out[i]);
    }
}