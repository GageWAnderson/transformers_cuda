#include <cuda_runtime.h>

// Kernel for element-wise addition
__global__ void add_tensors_kernel(const float* a, const float* b, float* c, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        c[idx] = a[idx] + b[idx];
    }
}

// Host function to call the kernel
void add_tensors(const float* a, const float* b, float* c, int size, cudaStream_t stream) {
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    add_tensors_kernel<<<blocks, threads, 0, stream>>>(a, b, c, size);
}
