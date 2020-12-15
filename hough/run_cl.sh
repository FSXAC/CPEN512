gcc hough_opencl.c -o hough_opencl.out -arch x86_64 -framework OpenCL -O2 $1 && ./hough_opencl.out
# rm hough_opencl.out
