#include "../../include/layers/feed_forward.cuh"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <curand.h>
#include <curand_kernel.h>

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

// Helper function to initialize weights with random values
void initialize_weights(float* d_weights, size_t size, cudaStream_t stream) {
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetStream(gen, stream);
    curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);
    curandGenerateUniform(gen, d_weights, size / sizeof(float));
    curandDestroyGenerator(gen);
}

// Helper function to initialize biases with zeros
void initialize_biases(float* d_biases, size_t size, cudaStream_t stream) {
    cudaMemsetAsync(d_biases, 0, size, stream);
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

    // Initialize weights and biases
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    initialize_weights(d_W1, size_W1, stream);
    initialize_biases(d_b1, size_b1, stream);
    initialize_weights(d_W2, size_W2, stream);
    initialize_biases(d_b2, size_b2, stream);
    cudaStreamDestroy(stream);
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
