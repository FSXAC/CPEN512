nvcc hough_cuda.cu -o hough_cuda.out -O2 -D IMG_SIZE=256 -m64 && ./hough_cuda.out
