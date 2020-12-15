#include "common.h"

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#define KERNEL_FILE "hough_opencl_kernel.cl"
#define KERNEL_FUNC "acc_vote"

#include "cl_error_check.h"

void hough_opencl(uint8_t *img, float *acc, int acc_width, int acc_height)
{
    struct timeval t_ready, t_begin, t_end, t_done;

    /* Get platform and device information */
    cl_platform_id platform_id = NULL;
    cl_device_id device_id = NULL;
    cl_uint ret_num_devices;
    cl_uint ret_num_platforms;
    cl_int ret;
    
    ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
    ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_ALL, 1, &device_id, &ret_num_devices);

    /* Create OpenCL context */
    cl_context context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &ret);

    /* Create command queue */
    cl_command_queue command_queue = clCreateCommandQueue(context, device_id, 0, &ret);

    /* Create program from loaded kernel source */
    /* Open and load kernel file */    
    FILE *fp = fopen(KERNEL_FILE, "r");
    if (fp == NULL) 
    {
        fprintf(stderr, "Failed to load kernel.\n");
        exit(1);
    }

    /* Load kernel source bytes */
    fseek(fp, 0, SEEK_END);
    size_t source_size = ftell(fp);
    rewind(fp);
    char *source_str = (char *) malloc(source_size + 1);
    source_str[source_size] = '\0';
    fread(source_str, sizeof(char), source_size, fp);
    fclose(fp);

    /* Use program souce to make program */
    cl_program program = clCreateProgramWithSource(context, 1, (const char **) &source_str, (const size_t *) &source_size, &ret);

    /* Build program */
    // ret = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    RC(clBuildProgram(program, 1, &device_id, "-I ./", NULL, NULL));

    /* Create memory buffer on device for the required I/O */
    cl_mem img_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(uint8_t) * IMG_SIZE * IMG_SIZE, NULL, &ret);
    cl_mem acc_mem_obj = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * acc_width * acc_height, NULL, &ret);
    gettimeofday(&t_ready, 0);

    /* Copy input data to input buffer */
    RC(clEnqueueWriteBuffer(command_queue, img_mem_obj, CL_TRUE, 0, sizeof(uint8_t) * IMG_SIZE * IMG_SIZE, img, 0, NULL, NULL));

    /* Create kernel */
    cl_kernel kernel = clCreateKernel(program, KERNEL_FUNC, &ret);
    RETURN_CHECK

    /* Set kernel arguments */
    RC(clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *) &img_mem_obj));
    RC(clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *) &acc_mem_obj));
    RC(clSetKernelArg(kernel, 2, sizeof(int), &acc_width));
    RC(clSetKernelArg(kernel, 3, sizeof(int), &acc_height));

    /* Find local and global size */
    // size_t local_size = 128;
    size_t local_size;
    RC(clGetKernelWorkGroupInfo(kernel, device_id, CL_KERNEL_WORK_GROUP_SIZE, sizeof(local_size), &local_size, NULL));
    size_t global_size = ceil(acc_width * acc_height / (float) (local_size)) * local_size;
    // size_t global_size = 2 * local_size;
    printf("Local size %zu, global size %zu\n", local_size, global_size);


    /* Begin time */
    gettimeofday(&t_begin, 0);

    /* Execute kernel */
    RC(clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, &global_size, &local_size, 0, NULL, NULL));
    clFinish(command_queue);    

    /* Stop clock */
    gettimeofday(&t_end, 0);

    /* Read from buffer after it's finished */
    RC(clEnqueueReadBuffer(command_queue, acc_mem_obj, CL_TRUE, 0, sizeof(float) * acc_width * acc_height, acc, 0, NULL, NULL));
    gettimeofday(&t_done, 0);

    /* Clean up */
    RC(clFlush(command_queue));
    RC(clFinish(command_queue));
    RC(clReleaseKernel(kernel));
    RC(clReleaseProgram(program));
    RC(clReleaseMemObject(img_mem_obj));
    RC(clReleaseMemObject(acc_mem_obj));
    RC(clReleaseCommandQueue(command_queue));
    RC(clReleaseContext(context));

    printf("Total execution time (including read and write): %.6f s\n", TIME(t_ready, t_done));
    printf("Breakdown:\n");
    printf("    Ready -> Begin : %.6f s\n", TIME(t_ready, t_begin));
    printf("    Begin -> End   : %.6f s\n", TIME(t_begin, t_end));
    printf("    End   -> Done  : %.6f s\n", TIME(t_end, t_done));
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
    hough_opencl(bin_image, acc, acc_width, acc_height);
    
    /* Normalize and out */
    uint8_t* out_acc = (uint8_t *) malloc(sizeof(uint8_t) * acc_height * acc_width);
    printf("normalizing and copying output...\n");
    float max = normalize_image(acc, out_acc, acc_width, acc_height);
    printf("maximum value in acc: %.1f\n", max);

    /* Write out image to file */
    stbi_write_jpg("out_opencl.jpg", acc_width, acc_height, 1, out_acc, 90);

    /* Close image */
    stbi_image_free(bin_image);

    /* Free memory */
    free(out_acc);
    free(acc);

    return 0;
}
