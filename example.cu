#include<cuda.h>
#include<iostream>

// __device__ is executed on the device and only callable from the device
// __global__ is executed on the device and only callable on the host
// __host__ is executed on the host and only callable on the host

// host is your cpu executing master thread
// device is gpu card through pcie bus
// host instructs device to execute kernel

// dim3 DimGrid(100, 50); // 2D grid with 5000 thread blocks
// dim3 DimBlock(4, 8, 8); // 3D block structure with 256 threads per block
// kernelFoo<<< DimGrid, DimBlock>>> // 5000 * 256 threads

__global__ void simpleKernel(int *data)
{
    // so function runs for every thread and updates this array
    data[threadIDX.x] += 2*(blockIdx.x + threadIDx.x); 
}

int main() {
    
    const int numElems = 4;
    int hostArray[numElems], *devArray;
    
    // allocate memory on the device (GPU); zero out all entries in this device array
    cudaMalloc((void **) &devArray, sizeof(int) * numElems);
    cudaMemset(devArray, 0, numElems * sizeof(int));
    
    // defining execution configuration
    // invoke GPU kernel, with one block that has four threads
    simpleKernel<<<1,numElems>>>(devArray);
    
    // bring the result back from the GPU into the hostArray
    cudaMemcpy(&hostArray, devArray, sizeof(int) * numElems, cudaMemcpyDeviceToHost);
    
    // print out the result to confirm that things are looking good
    std::cout << "Values stored in hostArray: " << std::endl;
    for (int i = 0; i < numElems; i++)
        std::cout << hostArray[i] << std::endl;

    cudaFree(devArray);
    return 0;
}
