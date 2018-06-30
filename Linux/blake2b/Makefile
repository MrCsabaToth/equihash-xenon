all: example_avx1 example_avx2

asm/zblake2_avx1.o:
	asm/assemble.sh

asm/zblake2_avx2.o:
	asm/assemble.sh

example_avx1: example_avx1.c asm/zcblake2_avx1.o
	gcc -o example_avx1 example_avx1.c asm/zcblake2_avx1.o

example_avx2: example_avx2.c asm/zcblake2_avx2.o
	gcc -o example_avx2 example_avx2.c asm/zcblake2_avx2.o

.PHONY: clean
clean:
	rm example_avx1 example_avx2
