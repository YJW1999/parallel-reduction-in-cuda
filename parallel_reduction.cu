#pragma once
#ifdef __INTELLISENSE__
void __syncthreads();
#endif

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>

#define size 100000 // set the length of input array

__device__ int parallel_reduction(int val, int* smem) {
	smem[threadIdx.x] = val;
	__syncthreads();
	for (int i = 1; i < blockDim.x; i *= 2)
	{
		if (threadIdx.x < (threadIdx.x ^ i)) //idea of XOR to acheive divide and conquer
			smem[threadIdx.x] += smem[threadIdx.x ^ i];
		__syncthreads();
	}

	return smem[0];
}

__global__ void ParallelReductionKernel(const int* input, int* output, int _size)
{
	extern __shared__ int smem[]; //using shared memory

	int tid = threadIdx.x;
	int gid = blockIdx.x * blockDim.x + tid;

	int sum = (gid < size) ? input[gid] : 0;

	int blockSum = parallel_reduction(sum, smem);

	if (tid == 0)
	{
		output[blockIdx.x] = blockSum;
	}
}


int main() {
	
	int* arr = new int[size];
	int sum = 0;

	for (int i = 0; i < size; ++i) {
		arr[i] = rand();
		sum += arr[i];
	}

	int* input_data, * output_data;
	cudaMalloc((void**)&input_data, sizeof(int) * size);
	cudaMalloc((void**)&output_data, sizeof(int) * size);
	cudaMemcpy(input_data, arr, size*sizeof(int), cudaMemcpyHostToDevice);

	const int blockSize = 256;
	const int gridSize = (size + blockSize - 1) / blockSize;

	ParallelReductionKernel << <gridSize, blockSize>> > (input_data, output_data, size);

	int* output = new int[size];
	cudaMemcpy(output, output_data, size*sizeof(int), cudaMemcpyDeviceToHost);

	int result = 0;
	for (int i = 0; i < size; ++i) {
		result += output[i];
	}
	
	std::cout << "The expected sum is: " << sum << std::endl;
	std::cout << "The output is: " << result << std::endl;

	delete[] arr;
	delete[] output;
	cudaFree(input_data);
	cudaFree(output_data);

	return 0;
}
