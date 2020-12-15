nvcc hough_cuda.cu -o hough_cuda.out -O2 $1 -m64 && ./hough_cuda.out
