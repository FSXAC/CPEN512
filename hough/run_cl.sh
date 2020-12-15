gcc -arch x86_64 -o hough_opencl.out hough_opencl.c -framework OpenCL -O2 $1 && ./hough_opencl.out
