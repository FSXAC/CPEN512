gcc ref.c -o ref.out -O3 -D M=$1 -D N=$1 \
    -m64 -march=barcelona -mtune=barcelona -ftree-vectorize -finline-functions \
    -fmerge-all-constants -fmodulo-sched -faggressive-loop-optimizations \
    -floop-interchange -funroll-all-loops -ftree-loop-distribution -funswitch-loops \
    && ./ref.out