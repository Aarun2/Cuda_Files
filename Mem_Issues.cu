// user defined pipelining
// unified virtual addressing
// GPU and
// Pool of memory use example 49 bits one more for GPU or CPU
// Unified Virtual Adressing UVA (one virtual memory space)
// frame or page number
// page fault page not present

// make gpu read from cpu memory
cudaHostAlloc() // pinned memory, good for small memory
// negative performance for many values

// mapping of host pinned memory into the memory space of the device
// don't need to copy back and forth

cudaHostAlloc(pointer, size, flag);
// flag = use "cudaHostAllocMapped" maps for direct access

// Zero copy GPU-CPU interaction
// can access from pinned and mapped host memory by a thread running on the GPU
// without a CUDA runtime copy call to move data onto the GPU
// zero copy memory
// Written through the PCI-e pipe

// Device is memory ballooned virtually
// cudaHostGetDevicePointer() call eliminated
// returned pointer to access on kernel

// cudaMemcpy sets parameter to cudaMemcpyDefault to determine location from the pointers
// cudaPointerGetAttributes() will give you this data
// cudaHostAlloc() makes data portable across all devices

// so now runtime fugures out with cudaMemcpyDefault flag

cudaMemcpy(gpu1Dst_memPntr, host_memPntr, byteSize1, cudaMemcpyDefault);

cudaMemcpy(host_memPntr, gpu1Dst_memPntr, byteSize1, cudaMemcpyDefault);

// needed for peer to peer (P2P) inter GPU transfer
// Z-C key accomplishment use pointer within device function access host data
// UVA data access and data transfer component
// data access: GPU can access data on different GPU
// data transfer: copy data in between GPUs

// unsimplified version

#define SZ 8

__global__ void AplusB(int *ret, int a, int b) {
    ret[threadIdx.x] = a + b + threadIdx.x; 
}

int main() { // earlier version
    int *ret;
    cudaMalloc(&ret, SZ*sizeof(int));
    AplusB<<<1, SZ>>>(ret, 10, 100); // call function
    int *host_ret = (int *)malloc(SZ * sizeof(int)); // allocate memory on host
    cudaMemcpy(host_ret, SZ*sizeof(int), cudaMemcpyDefault); // copy result back from memory to host
    for (int i = 0; i < SZ; i++)
        printf("%d: A+B = %d\n", i, host_ret[i]); // print data
    fre(host_ret); // free memory on host
    cudaFree(ret); // free memory on device
    return 0;
}

int main_2() { // better version
    int *ret;
    cudaMallocManaged(&ret, SZ*sizeof(int)); // can give flag as a hint
    AplusB<<<1, SZ>>>(ret, 10, 100); // kernel code is asynchronous
    cudaDeviceSynchronize(); // so need to wait for others
    for (int i = 0; i < SZ; i++)
        printf("%d: A+B = %d\n", i, host_ret[i]); // print data
    cudaFree(ret); // free memory on device
    return 0;
}

__device__ __managed__ int ret[52]; // a global memory on device
__global__ void AplusB(int *ret, int a, int b) {
    ret[threadIdx.x] = a + b + threadIdx.x; 
}

int main_2() { // even better version
     AplusB<<<1, SZ>>>(10, 100); // kernel code is asynchronous
     cudaDeviceSynchronize();
     for (int i = 0; i < SZ; i++) // migrated to host
        printf("%d: A+B = %d\n", i, ret[i]); // print data
    return 0;
}

// Any allocation created in the managed memory space automatically migrated to 
// where it is needed

__device__ __managed__ int x, y = 2;
__global__ void kernel() {
    x = 10;
}

int main() {
    kernel<<<1, 1, 1>>>();
    y = 20; // will crash as not synchrnoized need to add cuda synchronize before
    return 0;
}
// migrate pages as needed from device to host etc

__global__ void add_plus (int a, int b, int * result) {
    result[threadIdx.x] = a + b + threadIdx.x;
}

int main_3() { // another way to modify
    int ret[52];
    add_plus<<<1, SZ>>>(10, 100, ret);
    cudaDeviceSynchronize();
    for (int i = 0; i < SZ; i++) // migrated to host
        printf("%d: A+B = %d\n", i, ret[i]); // print data
    return 0;
}

cudaMemPrefetchAsync(data, N, GPU); // so prefetch pages needed
// also can do for CPU

