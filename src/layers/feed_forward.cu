#include "../../include/layers/feed_forward.cuh"
#include <cuda_runtime.h>
#include <cublas_v2.h>

// Activation function (ReLU)
__global__ void relu_activation(float* data, int size) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < size) {
        data[idx] = fmaxf(0.0f, data[idx]);
    }
}

// Kernel to add bias
__global__ void add_bias(float* data, const float* bias, int seq_len, int dim) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int total_size = seq_len * dim;
    if (idx < total_size) {
        int bias_idx = idx % dim;
        data[idx] += bias[bias_idx];
    }
}

FeedForward::FeedForward(int hidden_dim, int intermediate_dim) {
    this->hidden_dim = hidden_dim;
    this->intermediate_dim = intermediate_dim;

    // Allocate weights and biases
    size_t size_W1 = hidden_dim * intermediate_dim * sizeof(float);
    size_t size_b1 = intermediate_dim * sizeof(float);
    size_t size_W2 = intermediate_dim * hidden_dim * sizeof(float);
    size_t size_b2 = hidden_dim * sizeof(float);

    cudaMalloc(&d_W1, size_W1);
    cudaMalloc(&d_b1, size_b1);
    cudaMalloc(&d_W2, size_W2);
    cudaMalloc(&d_b2, size_b2);

    // Initialize weights and biases (e.g., using random values)
    // For simplicity, we'll leave initialization to be implemented

    // TODO: Implement weight initialization
}

FeedForward::~FeedForward() {
    // Free allocated memory
    cudaFree(d_W1);
    cudaFree(d_b1);
    cudaFree(d_W2);
    cudaFree(d_b2);
}

void FeedForward::forward(float* output, const float* input, int seq_len, cudaStream_t stream) {
    // Implement the feed-forward network forward pass

    // Allocate intermediate memory
    float* d_intermediate = nullptr;
    size_t intermediate_size = seq_len * intermediate_dim * sizeof(float);
    cudaMalloc(&d_intermediate, intermediate_size);

    cublasHandle_t cublas_handle;
    cublasCreate(&cublas_handle);
    cublasSetStream(cublas_handle, stream);

    // Linear Layer 1: intermediate = input * W1 + b1
    const float alpha = 1.0f;
    const float beta = 0.0f;

    // Compute input * W1
    // input: [seq_len, hidden_dim]
    // W1: [hidden_dim, intermediate_dim]
    // d_intermediate: [seq_len, intermediate_dim]
    cublasSgemm(cublas_handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                intermediate_dim, seq_len, hidden_dim,
                &alpha,
                d_W1, intermediate_dim,
                input, hidden_dim,
                &beta,
                d_intermediate, intermediate_dim);

    // Add bias b1
    int threads = 256;
    int blocks = (seq_len * intermediate_dim + threads - 1) / threads;
    add_bias<<<blocks, threads, 0, stream>>>(d_intermediate, d_b1, seq_len, intermediate_dim);

    // Apply ReLU activation
    relu_activation<<<blocks, threads, 0, stream>>>(d_intermediate, seq_len * intermediate_dim);

    // Linear Layer 2: output = intermediate * W2 + b2
    // output: [seq_len, hidden_dim]
    cublasSgemm(cublas_handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                hidden_dim, seq_len, intermediate_dim,
                &alpha,
                d_W2, hidden_dim,
                d_intermediate, intermediate_dim,
                &beta,
                output, hidden_dim);

    // Add bias b2
    blocks = (seq_len * hidden_dim + threads - 1) / threads;
    add_bias<<<blocks, threads, 0, stream>>>(output, d_b2, seq_len, hidden_dim);

    // Cleanup
    cudaFree(d_intermediate);
    cublasDestroy(cublas_handle);
}
