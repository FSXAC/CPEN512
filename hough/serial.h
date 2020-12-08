#ifndef SERIAL_H
#define SERIAL_H

#include "common.h"

void hough_serial(uint8_t *img, float *acc, int acc_width, int acc_height)
{
    /* Iterate through each pixel in the accumulator
     * which corresponds to each (rho, theta) value
     */
    for (int row = 0; row < acc_height; row++)
    {
        int r = row + MIN_R;
        for (int col = 0; col < acc_width; col++)
        {
            float theta = MIN_THETA + (D_THETA * col);

            // For each pixel on this line (radius, theta)
            if (theta < -(PI / 4) || theta > PI / 4)
            {
                for (int x = 0; x < IMG_SIZE; x++)
                {
                    int y = (r - cos(theta) * x) / sin(theta);
                    if (IN_BOUND(x, y))
                        GETACC(acc, row, col) += GETIM(bin_image, x, y);
                }
            }
            else
            {
                for (int y = 0; y < IMG_SIZE; y++)
                {
                    int x = (r - sin(theta) * y) / cos(theta);
                    if (IN_BOUND(x, y))
                        GETACC(acc, row, col) += GETIM(bin_image, x, y);
                }
            }
        }
    }
}

#endif