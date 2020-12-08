#ifndef COMMON_H
#define COMMON_H

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "stb_image.h"
#include "stb_image_write.h"

#define PI 3.141592653589723234
#define RAD(deg) (deg * PI / 180)

#define IMG_SIZE 1024

#if IMG_SIZE == 256
#define MAX_R 362
#define IMG_FILE "img/ipad256.jpg"
#elif IMG_SIZE == 512
#define MAX_R 724
#define IMG_FILE "img/testx.jpg"
#elif IMG_SIZE == 1024
#define MAX_R 1448
#define IMG_FILE "img/ipad1024.jpg"
#else
#define MAX_R 2896
#define IMG_FILE "img/ipad2048.jpg"
#endif

#define MIN_R -MAX_R
#define MAX_THETA RAD(89)
#define MIN_THETA RAD(-89)
#define THETA_STEPS MAX_R
#define D_THETA (MAX_THETA - MIN_THETA) / THETA_STEPS

// Macro to access array easier
#define GETIM(img, x, y) img[y * IMG_SIZE + x]
#define GETACC(acc, row, col) acc[row * THETA_STEPS + col]

// Bound check
#define IN_BOUND(x, y) (x < IMG_SIZE && x >= 0 && y < IMG_SIZE && y >= 0)

// Threshold image to (0, 1)
void threshold_image(uint8_t* img, int threshold)
{
    for (int row = 0; row < IMG_SIZE; row++)
        for (int col = 0; col < IMG_SIZE; col++)
            GETIM(img, row, col) = GETIM(img, row, col) > threshold ? 1 : 0;
}

// Normalize image to (0, 255)
float normalize_image(float *in, uint8_t *out, int w, int h)
{
    float max = 0;
    for (int row = 0; row < h; row++)
        for (int col = 0; col < w; col++)
            if (GETACC(in, row, col) > max)
                max = GETACC(in, row, col);

    for (int row = 0; row < h; row++)
        for (int col = 0; col < w; col++)
            GETACC(out, row, col) = 255 * (GETACC(in, row, col) / max);

    return max;
}

#endif