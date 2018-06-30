all: solver_avx1 solver_avx2 quickbench_avx1 quickbench_avx2

../asm/equihash_avx1.o:
	../asm/assmeble.sh

../asm/equihash_avx2.o:
	../asm/assmeble.sh

solver_avx1: solver.c ../asm/equihash_avx1.o
	gcc -ggdb -fopenmp -o solver_avx1 solver.c ../asm/equihash_avx1.o

solver_avx2: solver.c ../asm/equihash_avx2.o
	gcc -ggdb -fopenmp -o solver_avx2 solver.c ../asm/equihash_avx2.o

quickbench_avx1: quickbench.c ../asm/equihash_avx1.o
	gcc -ggdb -fopenmp -o quickbench_avx1 quickbench.c ../asm/equihash_avx1.o

quickbench_avx2: quickbench.c ../asm/equihash_avx2.o
	gcc -ggdb -fopenmp -o quickbench_avx2 quickbench.c ../asm/equihash_avx2.o

.PHONY: clean
clean:
	rm solver_avx1 solver_avx2 quickbench_avx1 quickbench_avx2
