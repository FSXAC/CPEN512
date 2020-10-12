# Run with optimizer
gcc -pthread ref_pthread.c -o ref.out -O3 -finline-functions -funroll-loops -m64 -D M=$1 -D N=$1 -D NUM_THREADS=$2 && ./ref.out

# Debug
# gcc -pthread ref_pthread.c -o ref.out -g -D M=$1 -D N=$1 && gdb ./ref.out

# Disabled flags
# -m64 -march=barcelona -mtune=barcelona -ftree-vectorize -finline-functions \
# -fmerge-all-constants -fmodulo-sched -faggressive-loop-optimizations \
# -floop-interchange -funroll-all-loops -ftree-loop-distribution -funswitch-loops \