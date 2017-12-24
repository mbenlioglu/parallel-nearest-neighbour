/*
	Created on: December 18, 2017
		Author: mbenlioglu


*/

#ifndef _CLOSENESS_CENTRALITY_
#define _CLOSENESS_CENTRALITY_


#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#ifdef __cplusplus
extern "C" {
#endif /*__cplusplus*/
#include <stdio.h> // stdio functions are used since C++ streams aren't necessarily thread safe
#include <stdlib.h>
#include <string.h>
#ifdef __cplusplus
}
#endif /*__cplusplus*/

#include "util.h"

//#define DEBUG;

// kernels
__global__ void NearestNeighbourKernel(Point *train, Point *test, int *result, int *trainSize, int *testSize);

// drivers
cudaError_t CudaNearestNeighbour(Point *train, Point *test, int *result, int trainSize, int testSize);

#endif // !_CLOSENESS_CENTRALITY_
