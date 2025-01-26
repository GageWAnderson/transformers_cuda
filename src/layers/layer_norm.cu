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

    // Add debug prints for gamma and beta weights
    float h_gamma[5], h_beta[5];
    cudaMemcpy(h_gamma, gamma, 5 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_beta, beta, 5 * sizeof(float), cudaMemcpyDeviceToHost);
    
    debugPrint("LayerNorm gamma (first 5): %f %f %f %f %f\n",
               h_gamma[0], h_gamma[1], h_gamma[2], h_gamma[3], h_gamma[4]);
    debugPrint("LayerNorm beta (first 5): %f %f %f %f %f\n",
               h_beta[0], h_beta[1], h_beta[2], h_beta[3], h_beta[4]);
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
}

void LayerNorm::forward(float* output, const float* input, int seq_len, cudaStream_t stream) {
    // Check input parameters
    CUDA_CHECK(cudaPeekAtLastError()); // Check for any previous errors
    
    const int BLOCK_SIZE = 256;
    const int shared_mem_size = 2 * sizeof(float); // For mean and variance

    
    if (seq_len <= 0 || hidden_dim <= 0) {
        throw std::runtime_error("Invalid dimensions in layer norm forward");
    }
    
    // Calculate total elements to process (seq_len is actually batch_size * seq_len)
    int total_sequences = seq_len;
    
    fused_layer_norm_kernel<<<total_sequences, BLOCK_SIZE, shared_mem_size, stream>>>(
        input, output, gamma, beta, hidden_dim, total_sequences, 1e-5f);
    
    CUDA_CHECK(cudaPeekAtLastError()); // Check for kernel launch errors
}
