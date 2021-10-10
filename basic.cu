#include <iostream>

using namespace std;

__global__ void AddIntsCuda(int *a, int *b) {
    a[0] += b[0];
}

int main() {
    int a = 5, b = 9;
    int *d_a, *d_b;

    if (cudaMalloc(&d_a, sizeof(int)) != cudaSucces) {

    }

    if (cudaMalloc(&d_b, sizeof(int)) != cudaSucces) {
        
    }
    
    cudaMemcpy(d_a,  &a, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b,  &b, sizeof(int), cudaMemcpyHostToDevice);
    
    AddIntsCuda<<<1, 1>>>(d_a, d_b);

    cudaMemcpy(&a,  d_a, sizeof(int), cudaMemcpyDeviceToHost);

    cout << "New value is " << a << endl;
    cudaFree(d_a);
    cudaFree(d_b);

    return 0;

}