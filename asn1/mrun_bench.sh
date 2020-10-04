#!/usr/bin/zsh

# first parameter is the size of matrix then it runs the program against using multiple nodes

mpicc ref_mpi.c -o ref_mpi.out -O3 -D M=$1 -D N=$1 \
    -m64 -march=barcelona -mtune=barcelona -ftree-vectorize -finline-functions \
    -fmerge-all-constants -fmodulo-sched -faggressive-loop-optimizations \
    -floop-interchange -funroll-all-loops -ftree-loop-distribution -funswitch-loops \

mpirun -np 1 ./ref_mpi.out
mpirun -np 2 ./ref_mpi.out
mpirun -np 4 ./ref_mpi.out
mpirun -np 8 ./ref_mpi.out
mpirun -np 16 ./ref_mpi.out