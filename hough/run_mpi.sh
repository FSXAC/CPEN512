mpicc hough_mpi.c -o hough_mpi.out -g -O0 $1 && mpirun -np 1 ./hough_mpi.out
