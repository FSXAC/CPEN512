mpicc ref_mpi.c -o ref_mpi.out -O3 \
    -m64 -march=barcelona -mtune=barcelona -ftree-vectorize -finline-functions \
    -fmerge-all-constants -fmodulo-sched -faggressive-loop-optimizations \
    -floop-interchange -funroll-all-loops -ftree-loop-distribution -funswitch-loops \
    && mpirun -np $1 ./ref_mpi.out