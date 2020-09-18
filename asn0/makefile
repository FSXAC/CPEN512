CC=/usr/bin/gcc

EXTRA_OPTS=-O3 -m64 -march=barcelona -mtune=barcelona
CUSTOM_OPTS=-Ofast -m64 -march=barcelona -mtune=barcelona -ftree-vectorize -fmerge-all-constants -fmodulo-sched -faggressive-loop-optimizations -floop-interchange -funroll-all-loops -ftree-loop-distribution -funswitch-loops

N_SIZE=256

EXTRA_DEFINES=-D NONE
DEFINES=-D N_SIZE=$(N_SIZE) $(EXTRA_DEFINES)

# Test
test-int: matmul.c
	gcc -o out/matmul_test_int matmul.c -D USE_INT $(DEFINES)
test-float: matmul.c
	gcc -o out/matmul_test_float matmul.c -D USE_FLOAT $(DEFINES)
test-double: matmul.c
	gcc -o out/matmul_test_double matmul.c -D USE_DOUBLE $(DEFINES)
test-tiled-int: matmul.c
	gcc -o out/matmul_test_tiled_int matmul.c -D USE_INT $(DEFINES) -D MATMUL_TILED
test-tiled-float: matmul.c
	gcc -o out/matmul_test_tiled_float matmul.c -D USE_FLOAT $(DEFINES) -D MATMUL_TILED
test-tiled-double: matmul.c
	gcc -o out/matmul_test_tiled_double matmul.c -D USE_DOUBLE $(DEFINES) -D MATMUL_TILED

# None
build: matmul.c
	gcc -o out/matmul matmul.c -D USE_INT $(DEFINES)
build-float: matmul.c
	gcc -o out/matmul_f matmul.c -D USE_FLOAT $(DEFINES)
build-double: matmul.c
	gcc -o out/matmul_d matmul.c -D USE_DOUBLE $(DEFINES)

# Optimization level 1
build-o: matmul.c
	gcc -o out/matmul_o1 matmul.c -O -D USE_INT $(DEFINES)
build-float-o: matmul.c
	gcc -o out/matmul_f_o1 matmul.c -O -D USE_FLOAT $(DEFINES)
build-double-o: matmul.c
	gcc -o out/matmul_d_o1 matmul.c -O -D USE_DOUBLE $(DEFINES)

# Optimization level 2
build-o2: matmul.c
	gcc -o out/matmul_o2 matmul.c -O2 -D USE_INT $(DEFINES)
build-float-o2: matmul.c
	gcc -o out/matmul_f_o2 matmul.c -O2 -D USE_FLOAT $(DEFINES)
build-double-o2: matmul.c
	gcc -o out/matmul_d_o3 matmul.c -O2 -D USE_DOUBLE $(DEFINES)

# Optimization level 3
build-o3: matmul.c
	gcc -o out/matmul_o3 matmul.c -O3 -D USE_INT $(DEFINES)
build-float-o3: matmul.c
	gcc -o out/matmul_f_o3 matmul.c -O3 -D USE_FLOAT $(DEFINES)
build-double-o3: matmul.c
	gcc -o out/matmul_d_o4 matmul.c -O3 -D USE_DOUBLE $(DEFINES)

# Extra options
build-extra: matmul.c
	gcc -o out/matmul_e matmul.c $(EXTRA_OPTS) -D USE_INT $(DEFINES)
build-float-extra: matmul.c
	gcc -o out/matmul_f_e matmul.c $(EXTRA_OPTS) -D USE_FLOAT $(DEFINES)
build-double-extra: matmul.c
	gcc -o out/matmul_d_e matmul.c $(EXTRA_OPTS) -D USE_DOUBLE $(DEFINES)

# Custom build options
build-custom: matmul.c
	gcc -o out/matmul_custom matmul.c $(CUSTOM_OPTS) -D USE_INT $(DEFINES)
build-float-custom: matmul.c
	gcc -o out/matmul_f_custom matmul.c $(CUSTOM_OPTS) -D USE_FLOAT $(DEFINES)
build-double-custom: matmul.c
	gcc -o out/matmul_d_custom matmul.c $(CUSTOM_OPTS) -D USE_DOUBLE $(DEFINES)

clean:
	rm -rf out/*
	rm -rf results/*
