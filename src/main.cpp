/*
	Created on: December 18, 2017
		Author: mbenlioglu

	Includes main function that runs the nearest neighbour calculation tests which will run on GPU device and host CPU for performance
	comparisons.
*/

#ifdef __cplusplus
extern "C" {
#endif /*__cplusplus*/
#include <stdio.h> // stdio functions are used since C++ streams aren't necessarily thread safe
#include <stdlib.h>
#include <string.h>
#ifdef __cplusplus
}
#endif /*__cplusplus*/

#include <cstring>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <omp.h>

#include "util.h"
#include "nearest_neighbour.cuh"

#define DEBUG

void CalculateNearestNeighbour(Point *train, Point *test, int *result, int trainSize, int testSize)
{
	static int minID;
	static int minDist;
#pragma omp threadprivate(minDist, minID)
#pragma omp parallel
	{
		minDist = INT32_MAX;
		minID = -1;
	}

	int dist;
#pragma omp parallel for private(dist)
	for (int i = 0; i < testSize; i++)
	{
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
		minDist = INT32_MAX;
		minID = -1;
	}
}

void PrintUsage(std::string appName)
{
	std::string firstExp = "Uses ./train.txt, ./test.txt and GPU-0 as default";
	std::string secondVer = appName + " <GPU deviceID>";
	std::string secondExp = "Uses ./train.txt, ./test.txt as default, and given input as GPU device ID";
	std::string thirdVer = appName + " <train file> <test file> <GPU deviceID>";
	std::string thirdExp = "Uses given inputs as train file, test file, and GPU ID";
	const char *str = "USAGE\n./%-55s %s\n./%-55s %s\n./%-55s %s\n";
	printf(str, appName.c_str(), firstExp.c_str(), secondVer.c_str(), secondExp.c_str(), thirdVer.c_str(), thirdExp.c_str());
}

int main(int argc, char *argv[])
{
	std::string baseName = std::string(argv[0]);
	std::string fillerAsterisk(100, '*');
	std::string fillerDashes(100, '-');

	// Get executable name from path
#ifdef _WIN32
	baseName = baseName.substr(baseName.rfind('\\') + 1);
	char *trainFile = "../data/train.txt", *testFile = "../data/test.txt"; // train and test files
#else
	baseName = baseName.substr(baseName.rfind('/') + 1);
	char *trainFile = "./data/train.txt", *testFile = "./data/test.txt"; // train and test files
#endif // WIN32

	int devID = 0; // selected device id
	switch (argc)
	{
	case 1:
		break;
	case 2:
		if (std::string(argv[1]) == "-h" || std::string(argv[1]) == "--help")
		{
			PrintUsage(baseName);
			return EXIT_SUCCESS;
		}
		else
			devID = atoi(argv[1]);
	case 4:
		trainFile = argv[1];
		testFile = argv[2];
		devID = atoi(argv[3]);
	default:
		PrintUsage(baseName);
		return EXIT_FAILURE;
	}

	printf((fillerAsterisk + "\n").c_str());
	printf("Starting %s ...\n", baseName.c_str());
	printf((fillerAsterisk + "\n").c_str());

	printf("\nInitializing Device...\n");

	//===========================================================================================================================
	// set the CUDA capable GPU to be used
	//
	int num_gpus = 0;   // number of CUDA GPUs

	cudaGetDeviceCount(&num_gpus);

	if (num_gpus < 1)
	{
		printf("no CUDA capable devices were detected\n");
		return EXIT_FAILURE;
	}
	else if (devID > num_gpus || devID < 0)
	{
		printf("Invalid device ID\n");
		return EXIT_FAILURE;
	}

	cudaDeviceProp dprop;
	cudaError_t cudaStatus = cudaGetDeviceProperties(&dprop, devID);
	printf("   %s #%d: %s\n\n", "Selected Device", devID, dprop.name);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
	}
	cudaSetDevice(devID);

	//===========================================================================================================================
	// Read data
	//
	std::cout << "Reading data...";
	std::ifstream fTrain(trainFile), fTest(testFile);
	if (!fTrain)
	{
		std::cout << "Unable to open train file!\n";
		return EXIT_FAILURE;
	}
	if (!fTest)
	{
		std::cout << "Unable to open test file!\n";
		return EXIT_FAILURE;
	}

	Point *testPoints, *trainPoints;

	// get size of test and train data
	unsigned int trainSize, testSize, dimension = 16;
	trainSize = std::count(std::istreambuf_iterator<char>(fTrain), std::istreambuf_iterator<char>(), '\n');
	testSize = std::count(std::istreambuf_iterator<char>(fTest), std::istreambuf_iterator<char>(), '\n');

	// create test and train arrays
	testPoints = new Point[testSize];
	trainPoints = new Point[trainSize];

	// reset files
	fTrain.clear();
	fTrain.seekg(0);
	fTest.clear();
	fTest.seekg(0);

	// fill train array
	std::string line = "";
	size_t i = 0;
	while (std::getline(fTrain, line))
	{
		std::istringstream iss(line);
		std::string token;

		for (uint8_t j = 0; j < 8; j++)
		{
			std::getline(iss, token, ',');
			trainPoints[i][j].first = atoi(token.c_str());
			std::getline(iss, token, ',');
			trainPoints[i][j].second = atoi(token.c_str());
		}
		++i;
	}
	fTrain.close();

	// fill test array
	line = "";
	i = 0;
	while (std::getline(fTest, line))
	{
		std::istringstream iss(line);
		std::string token;

		for (uint8_t j = 0; j < 8; j++)
		{
			std::getline(iss, token, ',');
			testPoints[i][j].first = atoi(token.c_str());
			std::getline(iss, token, ',');
			testPoints[i][j].second = atoi(token.c_str());
		}
		++i;
	}
	fTest.close();

	std::cout << " done\n" << fillerDashes << "\n";

	//===========================================================================================================================
	// Calculate nearest neighbour in CPU.
	//
	int *result = new int[testSize]();
	double elapsed_sequential, start, end;
	// Sequential
	omp_set_num_threads(1);
	std::cout << "Starting sequantial algorithm...";
	start = omp_get_wtime();
	CalculateNearestNeighbour(trainPoints, testPoints, result, trainSize, testSize);
	end = omp_get_wtime();
	elapsed_sequential = end - start;
	std::cout << " done\n";

	// print to file
	std::ofstream output("output.txt");
	if (output.is_open() && output.good())
	{
		for (size_t i = 0; i < testSize; i++)
			output << result[i] << "\n";
		output.close();
	}
	else std::cout << "Unable to open file";

	// Parallel
	int *result2 = new int[testSize]();
	double elapsed_parallel[4];
	for (int i = 1; i < 5; i++)
	{
		omp_set_num_threads((2 << i) / 2);
		std::cout << "Starting parallel algorithm with " << (2 << i) / 2 << " threads...";
		start = omp_get_wtime();
		CalculateNearestNeighbour(trainPoints, testPoints, result2, trainSize, testSize);
		end = omp_get_wtime();
		elapsed_parallel[i - 1] = end - start;
		std::cout << " ended\n";
#ifdef DEBUG
		std::cout << "Running correctness check...";
		std::string s = std::memcmp(result, result2, testSize * sizeof(int)) == 0 ? " correct\n" : " wrong!\n";
		std::cout << s;
#endif // DEBUG
	}

	//===========================================================================================================================
	// Calculate nearest neighbour in GPU.
	//
	std::cout << "Starting on GPU...";
	start = omp_get_wtime();
	cudaStatus = CudaNearestNeighbour(trainPoints, testPoints, result2, trainSize, testSize);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "CudaNearestNeighbour failed!");
		return EXIT_FAILURE;
	}
	end = omp_get_wtime();
	double elapsed_gpu = end - start;
	std::cout << " ended.\n";
#ifdef DEBUG
	std::cout << "Running correctness check...";
	std::string s = std::memcmp(result, result2, testSize * sizeof(int)) == 0 ? " correct\n" : " wrong!\n";
	std::cout << s;
#endif // DEBUG

	std::cout << " ended\n";
	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		return EXIT_FAILURE;
	}

	delete[] result2;
	// Print results
	std::cout << fillerAsterisk << "\nResults:\n" << fillerAsterisk << "\n";
	printf("| %-15s | %-12s | %-15s |\n", "GPU / CPU", "Thread Count", "Exec. Time");
	std::cout << fillerDashes << "\n";
	printf("| %-15s | %-12d | %-12.10f s |\n", "CPU", 1, elapsed_sequential);
	for (int i = 1; i < 5; i++)
		printf("| %-15s | %-12d | %-12.10f s |\n", "CPU", (2 << i) / 2, elapsed_parallel[i - 1]);
	printf("| %-15s | %-12d | %-12.10f s |\n", "GPU", testSize, elapsed_gpu);
	std::cout << "\n";

	return EXIT_SUCCESS;
}