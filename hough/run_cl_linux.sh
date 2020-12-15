gcc -m64 -march=native -o hough_opencl.out hough_opencl.c -l OpenCL -lm -O2 $1 && ./hough_opencl.out
