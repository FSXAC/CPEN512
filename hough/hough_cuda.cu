#include "common.h"

#define BLOCK_SIZE 1024

__global__
void acc_vote(uint8_t *img, float *acc, int acc_width, int acc_height)
{
    int thread_id = blockIdx.y * gridDim.x + blockIdx.x * blockDim.x + threadIdx.x;
    int row = thread_id / acc_width;
    int col = thread_id % acc_width;

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

double hough_cuda(uint8_t *img, float *acc, int acc_width, int acc_height)
{
    struct timeval begin, end;
    // gettimeofday(&begin, 0);
    
    /* allocate device memory */
    float *device_img;
    float *device_acc;
    
    cudaMalloc(&device_img, sizeof(uint8_t) * IMG_SIZE * IMG_SIZE);
    cudaMalloc(&device_acc, sizeof(float) * acc_width * acc_height);
    
    /* Copy input image file */
    cudaMemcpy(device_img, (void *) img, sizeof(uint8_t) * IMG_SIZE * IMG_SIZE, cudaMemcpyHostToDevice);
    
    gettimeofday(&begin, 0);
    int num_output_elements = acc_width * acc_height;
    acc_vote<<<num_output_elements, BLOCK_SIZE>>>(img, acc, acc_width, acc_height);
    cudaDeviceSynchronize();

    /* Get the output */
    cudaMemcpy(acc, (void *) device_acc, sizeof(float) * acc_width, acc_height, cudaMemcpyDeviceToHost);

    /* Free device memory */
    cudaFree(device_img);
    cudaFree(device_acc);

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
    double t = hough_cuda(bin_image, acc, acc_width, acc_height);
    printf("Execution time: %.2f s\n", t);
    
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
