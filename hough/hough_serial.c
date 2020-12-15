#include "common.h"

double hough_serial(uint8_t *img, float *acc, int acc_width, int acc_height)
{
    struct timeval begin, end;
    gettimeofday(&begin, 0);

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

    gettimeofday(&end, 0);
    return TIME(begin, end);
}

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
    /* Using float because int could run into overflow issues */
    int acc_width = THETA_STEPS;
    int acc_height = 2 * MAX_R;
    float *acc = (float *) malloc(sizeof(float) * acc_height * acc_width);

    // For each radius
    printf("transforming to %d by %d acc...\n", acc_width, acc_height);
    double t = hough_serial(bin_image, acc, acc_width, acc_height);
    printf("Execution time (serial): %.6f s\n", t);
    
    /* Normalize and out */
    uint8_t* out_acc = (uint8_t *) malloc(sizeof(uint8_t) * acc_height * acc_width);
    printf("normalizing and copying output...\n");
    float max = normalize_image(acc, out_acc, acc_width, acc_height);
    printf("maximum value in acc: %.1f\n", max);

    /* Write out image to file */
    stbi_write_jpg("out.jpg", acc_width, acc_height, 1, out_acc, 90);

    /* Close image */
    stbi_image_free(bin_image);

    /* Free memory */
    free(out_acc);
    free(acc);

    return 0;
}