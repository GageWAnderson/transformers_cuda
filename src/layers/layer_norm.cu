#include "layers/layer_norm.cuh"
#include "utils/utils.cuh"
#include "utils/debug.cuh"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cstdlib>
#include <stdio.h>
#include <cub/cub.cuh>

LayerNorm::LayerNorm(int hidden_dim, float* gamma_weights, float* beta_weights) 
    : hidden_dim(hidden_dim), gamma(gamma_weights), beta(beta_weights)
{
    // Verify we have a valid CUDA device
    int device_count;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (device_count == 0) {
        throw std::runtime_error("No CUDA devices available");
    }

    // Validate input parameters
    if (!gamma_weights || !beta_weights) {
        throw std::runtime_error("Null pointer passed for gamma or beta weights");
    }

    // Initialize cuBLAS
    cublasCreate(&cublas_handle);
}

LayerNorm::~LayerNorm() noexcept
{
    // No need to free gamma and beta as they're now just references
    cublasDestroy(cublas_handle);
}

__global__ void fused_layer_norm_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    int hidden_dim,
    int seq_len,
    float epsilon)
{
    extern __shared__ float shared_data[];
    
    int tid = threadIdx.x;
    int seq_idx = blockIdx.x;
    
    // Debug prints for first sequence only
    if (seq_idx == 0 && tid == 0) {
        printf("LayerNorm input (first 5): %f %f %f %f %f\n", 
               input[0], input[1], input[2], input[3], input[4]);
        printf("LayerNorm gamma (first 5): %f %f %f %f %f\n",
               gamma[0], gamma[1], gamma[2], gamma[3], gamma[4]);
        printf("LayerNorm beta (first 5): %f %f %f %f %f\n",
               beta[0], beta[1], beta[2], beta[3], beta[4]);
    }
    
    // Each block handles one sequence
    float sum = 0.0f;
    float sum_sq = 0.0f;
    
    // Cooperative loading and summation
    for (int i = tid; i < hidden_dim; i += blockDim.x) {
        float val = input[seq_idx * hidden_dim + i];
        sum += val;
        sum_sq += val * val;
    }
    
    // Block-level reduction
    __syncthreads();
    
    if (tid == 0) {
        float mean_val = sum / hidden_dim;
        float variance = (sum_sq / hidden_dim) - (mean_val * mean_val);
        shared_data[0] = mean_val;
        shared_data[1] = variance;
        if (seq_idx == 0) {
            printf("LayerNorm mean: %f, variance: %f\n", mean_val, variance);
        }
    }
    __syncthreads();
    
    // Normalize and apply gamma/beta
    float mean_val = shared_data[0];
    float variance = shared_data[1];
    
    for (int i = tid; i < hidden_dim; i += blockDim.x) {
        float val = input[seq_idx * hidden_dim + i];
        float norm_val = (val - mean_val) * rsqrtf(variance + epsilon);
        output[seq_idx * hidden_dim + i] = norm_val * gamma[i] + beta[i];
    }
    
    // Debug output values
    if (seq_idx == 0 && tid == 0) {
        printf("LayerNorm output (first 5): %f %f %f %f %f\n",
               output[0], output[1], output[2], output[3], output[4]);
    }
}

void LayerNorm::forward(float* output, const float* input, int seq_len, cudaStream_t stream) {
    // Check input parameters
    CUDA_CHECK(cudaPeekAtLastError()); // Check for any previous errors
    
    const int BLOCK_SIZE = 256;
    const int shared_mem_size = 2 * sizeof(float); // For mean and variance
    
    // Validate pointers and parameters
    if (output == nullptr || input == nullptr || gamma == nullptr || beta == nullptr) {
        throw std::runtime_error("Null pointer passed to layer norm forward");
    }
    
    if (seq_len <= 0 || hidden_dim <= 0) {
        throw std::runtime_error("Invalid dimensions in layer norm forward");
    }
    
    // Calculate total elements to process (seq_len is actually batch_size * seq_len)
    int total_sequences = seq_len;
    
    fused_layer_norm_kernel<<<total_sequences, BLOCK_SIZE, shared_mem_size, stream>>>(
        input, output, gamma, beta, hidden_dim, total_sequences, 1e-5f);
    
    CUDA_CHECK(cudaPeekAtLastError()); // Check for kernel launch errors
}

void LayerNorm::setGamma(float* gamma_weights) {
    CUDA_CHECK(cudaMemcpy(gamma, gamma_weights, hidden_dim * sizeof(float), cudaMemcpyDeviceToDevice));
}

void LayerNorm::setBeta(float* beta_weights) {
    CUDA_CHECK(cudaMemcpy(beta, beta_weights, hidden_dim * sizeof(float), cudaMemcpyDeviceToDevice));
}
