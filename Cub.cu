// CUB = Cuda UnBound
// its header library file

// device wide primitives
// block wide collective primitives
// warp wide primitives
// thread and resource utilities

// device level
// histogram, partition, radix sort

// block level
// block discontinutiy provides methods for flaggin discontinuties within 
// an ordered set of items partitioned across a cuda block
// block exchange methods for rearrange data partitioned across a thread block

#include <cub/cub.cub>

__global__ void ExampleKernel()

typedef cub::BlockDiscontinuity<int, 128> BlockDiscontinuity;

__shared__ typename BlockDiscontinuity::TempStorage temp_storage;

int thread_data[4]; // data spread in registers

int head_flags[4]; // returns discontinuities

BlockDiscontinuity(temp_storage).FlagHeads(head_flags, thread_data, cub::Inequality());

// CUB = not friendly, amazing performance

// OpenMP = runtime shared memory multithreading execution of code
// Standardizes task and loop level parallelism


#pragma omp simd // use vectorization
    for (int n = 0; n < size; n++)
        sinTable[n] = 

#pragma omp target teams distribute parallel for map(from:sinTable[0:size])
    for (int n = 0; n<size; ++n) // unload onto an accelerate

#include "omp.h"