dim3 threads(256); // x is set to 256 and y, z are 1
// so on more numbers initialize other dimensions

// 1024 max threads per block
// 2^32-1 blocks in a single launch

// GPU has limit on threads per block
// each thread: threadIdx, blockIdx, blockDim, gridDim
// index within block, block index within grid, block dimensions in threads, grid dimensions in blocks

// each thread has an unique id = blockIdx.X * blockDim.X + threadIdx.X

__global__ void (int *a, int *b, int *c) {
    for (int i = 0; i < 10; i++) {
        c[i] = a[i] + b[i];
    }
}

int main() {
    int *h_a = new int [10],
    int *b[10] = new int[10]
    int *c[10] = new int [10];

    int * d_a, * d_b, *d_c;

    for (int i = 0; i < 10; i++) {
        a[i] = i;
        b[i] = i *2;
        c[i] = 0;
    }

    if (cudaMalloc(&d_a, 10 * sizeof(int)) != cudaSucces) {

    }

    if (cudaMalloc(&d_b, 10* sizeof(int)) != cudaSucces) {
        
    }

    if (cudaMalloc(&d_c, 10* sizeof(int)) != cudaSucces) {
        
    }

    cudaMemcpy(d_a,  &a, 10*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b,  &b, 10*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_c,  &c, 10*sizeof(int), cudaMemcpyHostToDevice);

    AddIntsCuda<<<1, 10>>>(d_a, d_b, d_c);

    cudaMemcpy(&c,  d_c, sizeof(int), cudaMemcpyDeviceToHost);

    for (int i = 0; i < 10; i++) {
        cout << "New value of C is " << c[i] << endl;
    }

    delete[] h_a;
    delete[] h_b;

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}

// large amount of global memory cudaMalloc, free, memCpy, memset happen here
// persistent across kernel calls

// local memory used automatically when we run out of registers
// register spilling

// caches very fast

// constant memory all threads have access to the same const memory
// very fast, has its own cache 
// warp read many threads read the same value

// texture memory has its own cache
// designed for indexing

// shared memory is very fast, shared bw threads of each block
// bank conflicts slow it
// fast when all threads read from different banks or warp threads read the same value
// successive dword 4 bytes reside in different banks

// register scope is per thread