// scan = add each element to the sum of the elements before it
// prefix scan used a lot
// parallel scan solution #1: Hills and Steele
// look 1 element near first then 2 elements and so on for 2^M elements
// do M adds and 1 input update
// n(logn -1)+1
// n = 2^M
// harris algorithm
// leaves to root and root to leaves sweeps and 2n-2 sweeps
// a lot better

// 1D stencil algorithm used a lot for weighted sums in machine learning
// cudaMemcpy is blocking
// cudaMemcpyAsync posts it and moves on requires pinned host memory
// use cudaHostAlloc()

// For high performance:
// 1. Parallelize sequential code
// 2. Minimize data transfer
// 3. Alignned and coalesced global memory accesses
// 4. Minimize the use of global memory. Prefer shared memory access where possible

// Bank conflicts causes serialization
// Sufficient number of active threads
// keep number of threads per block multiple of 32 to avoid wasted lanes
// use fast math library
// avoid thread divergence as one thread has to wait for others

// parallel reduction
// has no global synchronization
// Roofline model: Performance is constrained by Bandwith or Number Crunching
// unroll the loop and multiple elements per thread for best parallel reduction

switch(threads) { // based on threads call the corresponding reduce
    case 512: // multiple elements per thread
        reduce<<<512>>>... break;
}

// should have a bw close to 84.56
// all threads in a single block can communicate
// grid is a collection of thread blocks

// dim3 abc(10, 10, 10) = a 3d structure or vector type with
// three integers x, y and z that can be initiallized

// kernel invocation
// blocks and thread per block
kernel<<<100, 256>>>(..)
// maximu of 1024 threads per block
// can launch 2^32 -1 blocks

// to read: prefix scan, parallel scan, 1D stencil, parallel reduction
// Warp Divergence, Latency Hiding, template parameters, occupancy
// working on cuda-gdb and visual profiler

