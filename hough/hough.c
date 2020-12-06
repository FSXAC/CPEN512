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

#define IMG_SIZE 256

#if IMG_SIZE == 256
#define MAX_R 362
#define IMG_FILE "img/ipad256.jpg"
#elif IMG_SIZE == 512
#define MAX_R 724
#elif IMG_SIZE == 1024
#define MAX_R 1448
#else
// Maximum radius (diagonal to diagnal of 2048 by 2048 image)
#define MAX_R 2896
#endif

#define MIN_R -MAX_R
#define MAX_THETA RAD(89)
#define MIN_THETA RAD(-89)
#define THETA_STEPS 512
#define D_THETA (MAX_THETA - MIN_THETA) / THETA_STEPS

// Macro to access array easier
#define GETIM(img, x, y) img[y * IMG_SIZE + x]
#define GETACC(acc, row, col) acc[row * THETA_STEPS + col]

int main() {

    /* Read image */
    int width, height, bpp;
    uint8_t* bin_image = stbi_load(IMG_FILE, &width, &height, &bpp, 1);
    printf("Image size: %d px by %d px, bpp: %d\n", width, height, bpp);
    
    if (width != IMG_SIZE || height != IMG_SIZE)
    {
        printf("Error! invalid image size\n");
        return 1;
    }

    /* Set up accumulator */
    int acc_width = THETA_STEPS;
    int acc_height = 2 * MAX_R;
    uint32_t *acc = (uint32_t *) malloc(sizeof(uint32_t) * acc_height * acc_width);


    // For each radius
    for (int acc_row_idx = 0; acc_row_idx < acc_height; acc_row_idx++)
    {
        int r = acc_row_idx + MIN_R;

        // FOr each theta
        for (int acc_col_idx = 0; acc_col_idx < acc_width; acc_col_idx++)
        {
            float theta = MIN_THETA + (D_THETA * acc_col_idx);

            // For each pixel on this line (radius, theta)
            if (theta < -(PI / 4) || theta > PI / 4)
            {
                for (int x = 0; x < IMG_SIZE; x++)
                {
                    int y = (r - cos(theta) * x) / sin(theta);
                    GETACC(acc, acc_row_idx, acc_col_idx) += GETIM(bin_image, x, y);
                }
            }
            else
            {
                for (int y = 0; y < IMG_SIZE; y++)
                {
                    int x = (r - sin(theta) * y) / cos(theta);
                    GETACC(acc, acc_row_idx, acc_col_idx) += GETIM(bin_image, x, y);
                }
            }
        }
    }

    /* Find maximum value in accumulator */
    uint32_t max = 0;
    for (int row = 0; row < acc_height; row++)
    {
        for (int col = 0; col < acc_width; col++)
        {
            if (GETACC(acc, row, col) > max)
                max = GETACC(acc, row, col);
        }
    }

    /* Normalized accumulator */
    uint8_t* out_acc = (uint8_t *) malloc(sizeof(uint8_t) * acc_height * acc_width);
    for (int row = 0; row < acc_height; row++)
    {
        for (int col = 0; col < acc_width; col++)
        {
            GETACC(out_acc, row, col) = GETACC(acc, row, col) / max;
        }
    }

    /* Write out image to file */
    stbi_write_jpg("out.jpg", acc_width, acc_height, 1, out_acc, 85);

    /* Close image */
    stbi_image_free(bin_image);

    return 0;
}