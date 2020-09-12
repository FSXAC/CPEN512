CC=/usr/bin/gcc

# None
build: matmul.c
	gcc -o out/matmul matmul.c -D USE_INT
build-float: matmul.c
	gcc -o out/matmul_f matmul.c -D USE_FLOAT
build-double: matmul.c
	gcc -o out/matmul_d matmul.c -D USE_DOUBLE

# Optimization level 1
build-o: matmul.c
	gcc -o out/matmul_o1 matmul.c -O -D USE_INT
build-float-o: matmul.c
	gcc -o out/matmul_f_o1 matmul.c -O -D USE_FLOAT
build-double-o: matmul.c
	gcc -o out/matmul_d_o1 matmul.c -O -D USE_DOUBLE

# Optimization level 2
build-o2: matmul.c
	gcc -o out/matmul_o2 matmul.c -O2 -D USE_INT
build-float-o2: matmul.c
	gcc -o out/matmul_f_o2 matmul.c -O2 -D USE_FLOAT
build-double-o2: matmul.c
	gcc -o out/matmul_d_o3 matmul.c -O2 -D USE_DOUBLE

# Optimization level 3
build-o3: matmul.c
	gcc -o out/matmul_o3 matmul.c -O3 -D USE_INT
build-float-o3: matmul.c
	gcc -o out/matmul_f_o3 matmul.c -O3 -D USE_FLOAT
build-double-o3: matmul.c
	gcc -o out/matmul_d_o4 matmul.c -O3 -D USE_DOUBLE

build-extra: matmul.c
	gcc -o out/matmul_e matmul.c -O3 -m64 -march=barcelona -mtune=barcelona -D USE_INT
build-float-extra: matmul.c
	gcc -o out/matmul_f_e matmul.c -O3 -m64 -march=barcelona -mtune=barcelona -D USE_FLOAT
build-double-extra: matmul.c
	gcc -o out/matmul_d_e matmul.c -O3 -m64 -march=barcelona -mtune=barcelona -D USE_DOUBLE

# Custom build options

clean:
	rm -f out/*