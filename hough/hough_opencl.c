#include "common.h"

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#define MAX_SOURCE_SIZE (0x100000)

double hough_opencl(uint8_t *img, float *acc, int acc_width, int acc_height)
{
    struct timeval begin, end;

    /* Open and load kernel file */    
    FILE *fp = fopen("hough_opencl_kernel.cl", "r");
    if (fp == NULL) 
    {
        fprintf(stderr, "Failed to load kernel.\n");
        exit(1);
    }


    /* Load bytes */
    char *source_str = (char *) malloc(MAX_SOURCE_SIZE);
    size_t source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fp);
    fclose(fp);

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

    /* Create memory buffer on device for the required I/O */
    cl_mem img_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(uint8_t) * IMG_SIZE * IMG_SIZE, NULL, &ret);
    cl_mem acc_mem_obj = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * acc_width * acc_height, NULL, &ret);
    cl_mem acc_width_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(int), NULL, &ret);
    cl_mem acc_height_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(int), NULL, &ret);

    /* Create program from loaded kernel source */
    cl_program program = clCreateProgramWithSource(context, 1, (const char **) &source_str, (const size_t *) &source_size, &ret);

    /* Build program */
    ret = clBuildProgram(program, 1, &device_id, "-I ./", NULL, NULL);

    /* Create kernel */
    cl_kernel kernel = clCreateKernel(program, "hough_opencl", &ret);

    /* Set kernel arguments */
    ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *) &img_mem_obj);
    ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *) &acc_mem_obj);
    ret = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *) &acc_width_mem_obj);
    ret = clSetKernelArg(kernel, 3, sizeof(cl_mem), (void *) &acc_height_mem_obj);

    /* Start clock */
    gettimeofday(&begin, 0);

    /* Execute kernel */
    size_t glboal_item_size = acc_width * acc_height;
    size_t local_item_size = 64;
    ret = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, &glboal_item_size, &local_item_size, 0, NULL, NULL);

    /* Stop clock */
    gettimeofday(&end, 0);

    /* Read from buffer after it's finished */
    ret = clEnqueueReadBuffer(command_queue, acc_mem_obj, CL_TRUE, 0, sizeof(float) * acc_width * acc_height, acc, 0, NULL, NULL);

    /* Clean up */
    ret = clFlush(command_queue);
    ret = clFinish(command_queue);
    ret = clReleaseKernel(kernel);
    ret = clReleaseProgram(program);
    ret = clReleaseMemObject(img_mem_obj);
    ret = clReleaseMemObject(acc_mem_obj);
    ret = clReleaseMemObject(acc_width_mem_obj);
    ret = clReleaseMemObject(acc_height_mem_obj);
    ret = clReleaseCommandQueue(command_queue);
    ret = clReleaseContext(context);

    return TIME(begin, end);
}