CC=/usr/bin/gcc

# None
build: matmul.c
	gcc -o matmul matmul.c -D USE_INT
build-float: matmul.c
	gcc -o matmul matmul.c -D USE_FLOAT
build-double: matmul.c
	gcc -o matmul matmul.c -D USE_DOUBLE

# Optimization level 1
build-o: matmul.c
	gcc -o matmul matmul.c -O -D USE_INT
build-float-o: matmul.c
	gcc -o matmul matmul.c -O -D USE_FLOAT
build-double-o: matmul.c
	gcc -o matmul matmul.c -O -D USE_DOUBLE

# Optimization level 2
build-o2: matmul.c
	gcc -o matmul matmul.c -O2 -D USE_INT
build-float-o2: matmul.c
	gcc -o matmul matmul.c -O2 -D USE_FLOAT
build-double-o2: matmul.c
	gcc -o matmul matmul.c -O2 -D USE_DOUBLE

# Optimization level 3
build-o3: matmul.c
	gcc -o matmul matmul.c -O3 -D USE_INT
build-float-o3: matmul.c
	gcc -o matmul matmul.c -O3 -D USE_FLOAT
build-double-o3: matmul.c
	gcc -o matmul matmul.c -O3 -D USE_DOUBLE

# Custom build options
build-extra: matmul.c
	gcc -o matmul matmul.c -O3 -m64 -march=barcelona -mtune=barcelona -D USE_INT
build-float-extra: matmul.c
	gcc -o matmul matmul.c -O3 -m64 -march=barcelona -mtune=barcelona -D USE_FLOAT
build-double-extra: matmul.c
	gcc -o matmul matmul.c -O3 -m64 -march=barcelona -mtune=barcelona -D USE_DOUBLE


clean:
	rm -f $(ODIR)/*.o *~ core $(INCDIR)/*~ matmul