// concurrency through streams
// CUDA Stream: a queue of GPU operations
// by default stream 0 that doesn't allow overtaking

cudaStream_t stream[2];
for (int i = 0; i < 2; i++)
    cudaStreamCreate(&stream[i]);
float* hostPtr;
cudaMallocHost(&hostPtr, 2 * size); // two halves

for (int i - 0; i < 2; i++) {  // chunkification
    // move half of the data to device
    cudaMemcpyAsync(inputDevptr + i * size, hostPtr + i * size, size, cudaMemcpyHostToDevice, stream[i]);
    // call the kernel, gets output
    MyKernel<<<100, 512, 0, stream[i]>>>(outputDevPtr + i *size, inputDevPtr + i * size, size);
    // take output and fill back stream
    cudaMemcpyAsync(hostPtr + i * size, outputDevPtr + i * size, size, cudaMemcpyDeviceToHost, stream[i]);
}

for (int i = 0; i < 2; i++)
    cudaStreamDestroy(stream[i]); // fold back the streams

// Stream 0 is special can not overlap other streams
// Streams with non blocking flag are exception

// cudaDeviceSynchronize(): Hosts waits till all tasks prior to call are done
// cudaStreamSynchronize(): Allows host to synchronize with a specific stream
// cudaStreamWaitEvent(): So stream waits on an event and event can be executed by another stream
// cudaStreamQuery(): A way to know if all preceeding commands in a stream have completed

for () { // overlap solutions breadth first that gives better performance
    // some stream0 operation
    // some stream1 operation
    // some stream0 operation
    // some stream1 operation
}

// thread focus dictates which thread you are looking at
// print local or shared variables, print registers, print stack contents
// applies only to threads in focus
// attributes: kernel index, block index, thread index

// run time error checker
// cuda-memcheck like valgrind
// detect stack overflow etc...

// check return code of CUDA API routines
// compiler uses name mangling to differentiate between overloaded functions
// change the function name a little, also happens in classes

// profiler get info on app
// CUPTI very low level ECE