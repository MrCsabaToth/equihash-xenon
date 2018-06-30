//compile with
//gcc -o quickbench quickbench.c equihash_avx2.o
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
// To sooth VS Code
#include <time.h>
#include <omp.h>

#define CONTEXT_SIZE 178033152
#define ITERATIONS 100

//Linkage with assembly
//EhPrepare takes in 136 bytes of input. The remaining 4 bytes of input is fed as nonce to EhSolver.
//EhPrepare saves the 136 bytes in context, and EhSolver can be called repeatedly with different nonce.
void EhPrepare(void *context, void *input);
int32_t EhSolver(void *context, uint32_t nonce);
extern char testinput[];

int main(void)
{
	int thread_count = omp_get_max_threads();
	void *context_alloc;
	void* contexts[thread_count];
	uint32_t *pu32;
	uint64_t *pu64, previous_rdtsc;
	uint8_t inputheader[144];	//140 byte header
	FILE *infile, *outfile;
	struct timespec time0, time1;
	long t0, t1;
	int32_t numsolutions[thread_count], thread_solutions[thread_count], total_solutions;
	uint32_t nonce, delta_time[thread_count], total_time[thread_count], total_times;
	int i, j;

	printf("Thread count %d\n", thread_count);

	context_alloc = malloc((CONTEXT_SIZE + 4096) * thread_count);
	infile = 0;
	infile = fopen("input.bin", "rb");
	if (infile) {
		puts("Reading input.bin");
		fread(inputheader, 140, 1, infile);
		fclose(infile);
	} else {
		puts("input.bin not found, use sample data (beta1 testnet block 2)");
		memcpy(inputheader, testinput, 140);
	}

	#pragma omp parallel for
	for (i = 0; i < thread_count; i++) {
		contexts[i] = (void*) (((long) context_alloc + 4095) & -4096 + i * CONTEXT_SIZE);
		printf("context %d assigned %p, %ld\n", i, contexts[i], ((long) context_alloc + 4095) & -4096 + i * CONTEXT_SIZE);
		printf("context will be prepared for %d\n", i);
		EhPrepare(contexts[i], (void *) inputheader);
		printf("context prepared for %d\n", i);
	}

	//Warm up, timing not taken into average
	nonce = 0;
	clock_gettime(CLOCK_MONOTONIC, &time0);
	#pragma omp parallel for
	for (i = 0; i < thread_count; i++) {
		numsolutions[i] = EhSolver(contexts[i], nonce);
		thread_solutions[i] = 0;
		total_time[i] = 0;
	}
	clock_gettime(CLOCK_MONOTONIC, &time1);
	delta_time[0] = (uint32_t) ((time1.tv_sec * 1000000000 + time1.tv_nsec)
			- (time0.tv_sec * 1000000000 + time0.tv_nsec))/1000000;
	printf("(Warm up) Time: %u ms, solutions: %u\n", delta_time, numsolutions[i]);

	printf("Running %d iterations...\n", ITERATIONS);
	nonce = 58;	//arbritary number to get 19 solutions in 10 iterations (to match 1.88 solutions per run)
	total_solutions = 0;
	#pragma omp parallel for
	for (i = 0; i < ITERATIONS; i++) {
		clock_gettime(CLOCK_MONOTONIC, &time0);
		int tid = omp_get_num_threads();
		numsolutions[tid] = EhSolver(contexts[i % tid], nonce + i);
		clock_gettime(CLOCK_MONOTONIC, &time1);
		delta_time[tid] = (uint32_t) ((time1.tv_sec * 1000000000 + time1.tv_nsec)
				- (time0.tv_sec * 1000000000 + time0.tv_nsec))/1000000;
		total_time[tid] += delta_time[tid];
		thread_solutions[tid] += numsolutions[tid];
		printf("Time: %u ms, solutions: %u\n", delta_time, numsolutions[tid]);
	}


	for (i = 0; i < thread_count; i++) {
		printf("Average time for %d: %d ms; %.3f Sol/s\n",
				i,
				total_time[i] / ITERATIONS / thread_count,
				(double) 1000.0 * thread_solutions[i] / total_time[i]);
		total_solutions += thread_solutions[i];
		total_times += total_time[i];
	}
	printf("Average time: %d ms; %.3f Sol/s\n",
			total_times / ITERATIONS, (double) 1000.0 * total_solutions / total_times);

	free(context_alloc);
	return 0;
}
