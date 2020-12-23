mpicc hough_mpi.c -o hough_mpi.out -m64 -march=barcelona -lm -O2 && mpirun -np $1 ./hough_mpi.out
