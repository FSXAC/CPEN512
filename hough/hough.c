#include "common.h"
#include "serial.h"

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

    /* Create threshold image */
    threshold_image(bin_image, 50);

    /* Set up accumulator */
    /* Using float because int could run into overflow issues */
    int acc_width = THETA_STEPS;
    int acc_height = 2 * MAX_R;
    float *acc = (float *) malloc(sizeof(float) * acc_height * acc_width);

    // For each radius
    printf("transforming to %d by %d acc...\n", acc_width, acc_height);
    hough_serial(bin_image, acc, acc_width, acc_height);
    
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