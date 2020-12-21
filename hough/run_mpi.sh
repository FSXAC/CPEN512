mpicc hough_mpi.c -o hough_mpi.out -O2 && mpirun -np $1 ./hough_mpi.out
