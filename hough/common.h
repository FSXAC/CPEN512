#ifndef COMMON_H
#define COMMON_H

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "stb_image.h"
#include "stb_image_write.h"

#include "params.h"

// Time measuring
#define TIME(begin, end) ((end.tv_sec - begin.tv_sec) + (end.tv_usec - begin.tv_usec) * 1e-6)

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
