#include "nearest_neighbour.cuh"

#include <cmath>

#define NREPS 10 // number of repetations for time calculations

#define THREADS_PER_BLOCK 1024

__global__ void NearestNeighbourKernel(Point *train, Point *test, int *result, int trainSize, int testSize)
{
	int blockId = blockIdx.x + blockIdx.y * gridDim.x;
	unsigned int i = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;
	if (i < testSize)
	{
		__shared__ int minDist = INT32_MAX;
		__shared__ int minID = -1;
		__shared__ int dist = 0;

		for (int j = 0; j < trainSize; j++)
		{
			dist = 0;
			// Calculate distance between points
			for (int k = 0; k < 8; k++)
			{
				dist += train[j][k].first > test[i][k].first ? (train[j][k].first - test[i][k].first) * (train[j][k].first - test[i][k].first)
					: (test[i][k].first - train[j][k].first) * (test[i][k].first - train[j][k].first);
				dist += train[j][k].second > test[i][k].second ? (train[j][k].second - test[i][k].second) * (train[j][k].second - test[i][k].second)
					: (test[i][k].second - train[j][k].second) * (test[i][k].second - train[j][k].second);
			}
			if (dist < minDist)
			{
				minDist = dist;
				minID = j;
			}
		}
		result[i] = minID;
	}
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t CudaNearestNeighbour(Point *train, Point *test, int *result, int trainSize, int testSize)
{
	Point *dev_train, *dev_test;
	int *dev_result;
	cudaError_t cudaStatus;

	int numThreads = (int)sqrt(THREADS_PER_BLOCK);
	dim3 dimBlock(numThreads, numThreads, 1);

	//===========================================================================================================================
	// Allocate GPU buffers for three vectors (two input, one output)
	//
	cudaStatus = cudaMalloc((void**)&dev_result, testSize * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_train, trainSize * sizeof(Point));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_test, testSize * sizeof(Point));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	//===========================================================================================================================
	// Copy input vectors from host memory to GPU buffers.
	//
	cudaStatus = cudaMemcpy(dev_train, train, trainSize * sizeof(Point), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_test, test, testSize * sizeof(Point), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	//===========================================================================================================================
	// Launch a kernel on the GPU with one thread for each element, and check for errors.
	//
	printf("%d, %d\n", testSize, testSize / numThreads);
	NearestNeighbourKernel<<<(testSize+THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK, dimBlock>>>(dev_result, dev_rowPtr, dev_colInd, testSize);

	//===========================================================================================================================
	// Check for any errors launching the kernel
	//
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		goto Error;
	}

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(result, dev_result, testSize * sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

Error:
	cudaFree(dev_result);
	cudaFree(dev_train);
	cudaFree(dev_test);

	return cudaStatus;
}