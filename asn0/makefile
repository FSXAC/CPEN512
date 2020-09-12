CC=/usr/bin/gcc
EXTRA_OPTS=-O3 -m64 -march=barcelona -mtune=barcelona
N_SIZE = 256

# Test
test-int: matmul.c
	gcc -o out/matmul_test_int matmul.c -D USE_INT -D N_SIZE=$(N_SIZE) -D DEBUG_PRINT
test-float: matmul.c
	gcc -o out/matmul_test_float matmul.c -D USE_FLOAT -D N_SIZE=$(N_SIZE) -D DEBUG_PRINT
test-double: matmul.c
	gcc -o out/matmul_test_double matmul.c -D USE_DOUBLE -D N_SIZE=$(N_SIZE) -D DEBUG_PRINT

# None
build: matmul.c
	gcc -o out/matmul matmul.c -D USE_INT -D N_SIZE=$(N_SIZE)
build-float: matmul.c
	gcc -o out/matmul_f matmul.c -D USE_FLOAT -D N_SIZE=$(N_SIZE)
build-double: matmul.c
	gcc -o out/matmul_d matmul.c -D USE_DOUBLE -D N_SIZE=$(N_SIZE)

# Optimization level 1
build-o: matmul.c
	gcc -o out/matmul_o1 matmul.c -O -D USE_INT -D N_SIZE=$(N_SIZE)
build-float-o: matmul.c
	gcc -o out/matmul_f_o1 matmul.c -O -D USE_FLOAT -D N_SIZE=$(N_SIZE)
build-double-o: matmul.c
	gcc -o out/matmul_d_o1 matmul.c -O -D USE_DOUBLE -D N_SIZE=$(N_SIZE)

# Optimization level 2
build-o2: matmul.c
	gcc -o out/matmul_o2 matmul.c -O2 -D USE_INT -D N_SIZE=$(N_SIZE)
build-float-o2: matmul.c
	gcc -o out/matmul_f_o2 matmul.c -O2 -D USE_FLOAT -D N_SIZE=$(N_SIZE)
build-double-o2: matmul.c
	gcc -o out/matmul_d_o3 matmul.c -O2 -D USE_DOUBLE -D N_SIZE=$(N_SIZE)

# Optimization level 3
build-o3: matmul.c
	gcc -o out/matmul_o3 matmul.c -O3 -D USE_INT -D N_SIZE=$(N_SIZE)
build-float-o3: matmul.c
	gcc -o out/matmul_f_o3 matmul.c -O3 -D USE_FLOAT -D N_SIZE=$(N_SIZE)
build-double-o3: matmul.c
	gcc -o out/matmul_d_o4 matmul.c -O3 -D USE_DOUBLE -D N_SIZE=$(N_SIZE)

# Extra options
build-extra: matmul.c
	gcc -o out/matmul_e matmul.c $(EXTRA_OPTS) -D USE_INT -D N_SIZE=$(N_SIZE)
build-float-extra: matmul.c
	gcc -o out/matmul_f_e matmul.c $(EXTRA_OPTS) -D USE_FLOAT -D N_SIZE=$(N_SIZE)
build-double-extra: matmul.c
	gcc -o out/matmul_d_e matmul.c $(EXTRA_OPTS) -D USE_DOUBLE -D N_SIZE=$(N_SIZE)

# Custom build options

clean:
	rm -f out/*
	rm -f results/*