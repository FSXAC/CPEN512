/* Keep -I option (apparently) */
/* https://stackoverflow.com/questions/30514189/how-to-include-header-correctly-in-the-opencl-kernel */
#include "common.h"

__kernel void acc_vote(
    __global const uint8_t *img,
    __global float *acc,
    __global int acc_width,
    __global int acc_height
    )
{
    /* Get global id (like thread id) */
    int id = get_global_id(0);

    int row = id / acc_width;
    int col = id % acc_width;

    /* If within bound of the output space */
    if (row < acc_height && col < acc_width)    
    {
        int r = row + MIN_R;
        float theta = MIN_THETA + (D_THETA * col);
        
        if (theta < -(PI / 4) || theta > PI / 4)
        {
            for (int x = 0; x < IMG_SIZE; x++)
            {
                int y = (r - cos(theta) * x) / sin(theta);
                if (IN_BOUND(x, y))
                    GETACC(acc, row, col) += GETIM(img, x, y) > THRESHOLD;
            }
        }
        else
        {
            for (int y = 0; y < IMG_SIZE; y++)
            {
                int x = (r - sin(theta) * y) / cos(theta);
                if (IN_BOUND(x, y))
                    GETACC(acc, row, col) += GETIM(img, x, y) > THRESHOLD;
            }
        }
    }
}