/* Keep -I option (apparently) */
/* https://stackoverflow.com/questions/30514189/how-to-include-header-correctly-in-the-opencl-kernel */

/* Also every time the main program compiles for some reason on Linux + Nvidia 
 * This kernel needs to be modified a bit (because I think there's some pre-compiled caching)
 * which causes constants such as image size, boundaries to be out-of-date and cause segment fault issues
 *
 * Also found https://forums.developer.nvidia.com/t/disable-caching-by-the-opencl-compiler/23752/2
 * do export CUDA_DISABLE_CACHE=1 to disable caching of kernel code
 */
#include "params.h"

__kernel void acc_vote(
    __global const unsigned char *img,
    __global float *acc,
    int acc_width,
    int acc_height
    )
{
    /* Get global id (like thread id) */
    int id = get_global_id(0);

    int row = id / acc_width;
    int col = id % acc_width;

    /* If within bound of the output space */
    if (row < acc_height && col < acc_width)    
    {
        //GETACC(acc, row, col) = (row / 16 % 2) * (col);
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
