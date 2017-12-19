/*
	Created on: December 19, 2017
		Author: mbenlioglu

	Common header for data structures used in the project
*/

#pragma once

#include <stdint.h>

// struct for two 4-bit unsigned integers
struct uint4s {
	uint8_t first : 4, second : 4;
};
// point with 16 dimensions (2 * 8) where each dimension is between 0..15 (4 bits)
typedef uint4s Point[8];
