mpicc sumarray.c -o sumarray.out -O3 \
    -m64 -march=barcelona -mtune=barcelona -ftree-vectorize -finline-functions \
    -fmerge-all-constants -fmodulo-sched -faggressive-loop-optimizations \
    -floop-interchange -funroll-all-loops -ftree-loop-distribution -funswitch-loops \
    && ./sumarray.out