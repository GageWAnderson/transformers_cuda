#include "layers/layer_norm.cuh"
#include "utils/utils.cuh"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cstdlib>
#include <stdio.h>
#include <cub/cub.cuh>

LayerNorm::LayerNorm(int hidden_dim) : hidden_dim(hidden_dim)
{
    // Allocate and initialize gamma and beta (scale and shift parameters)
    cudaMalloc(&gamma, hidden_dim * sizeof(float));
    cudaMalloc(&beta, hidden_dim * sizeof(float));

    // Initialize gamma to 1 and beta to 0
    float *h_gamma = (float *)malloc(hidden_dim * sizeof(float));
    float *h_beta = (float *)malloc(hidden_dim * sizeof(float));
    for (int i = 0; i < hidden_dim; ++i)
    {
        h_gamma[i] = 1.0f;
        h_beta[i] = 0.0f;
    }
    cudaMemcpy(gamma, h_gamma, hidden_dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(beta, h_beta, hidden_dim * sizeof(float), cudaMemcpyHostToDevice);

    free(h_gamma);
    free(h_beta);

    // Initialize cuBLAS
    cublasCreate(&cublas_handle);
    temp_storage = nullptr;
    temp_storage_bytes = 0;
}

LayerNorm::~LayerNorm()
{
    // Free resources
    cudaFree(gamma);
    cudaFree(beta);
    if (temp_storage) cudaFree(temp_storage);
    cublasDestroy(cublas_handle);
}

void LayerNorm::initialize_temp_storage(int max_seq_len) {
    if (temp_storage) cudaFree(temp_storage);
    
    // Calculate required temporary storage
    size_t temp_bytes = 0;
    cub::DeviceSegmentedReduce::Sum(nullptr, temp_bytes, (float*)nullptr,
                                   (float*)nullptr, max_seq_len, 
                                   (int*)nullptr, (int*)nullptr);
    
    cudaMalloc(&temp_storage, temp_bytes);
    temp_storage_bytes = temp_bytes;
}

__global__ void fused_layer_norm_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    const float* __restrict__ mean_input,
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
    
    // TODO: Understand warp-level programming to get more efficiencies
    // Warp-level reduction
    // sum = warpReduceSum(sum);
    // sum_sq = warpReduceSum(sum_sq);
    
    // Block-level reduction
    // if (tid < warpSize) {
    //     shared_data[tid] = sum;
    //     shared_data[tid + warpSize] = sum_sq;
    // }
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
    if (!temp_storage || seq_len > temp_storage_bytes) {
        initialize_temp_storage(seq_len);
    }

    const int BLOCK_SIZE = 256;
    const int shared_mem_size = 2 * sizeof(float) * 32; // For mean and variance
    
    fused_layer_norm_kernel<<<seq_len, BLOCK_SIZE, shared_mem_size, stream>>>(
        input, output, gamma, beta, nullptr, hidden_dim, seq_len, 1e-5f);
}

void LayerNorm::setGamma(float* gamma_weights) {
    cudaMemcpy(gamma, gamma_weights, hidden_dim * sizeof(float), cudaMemcpyDeviceToDevice);
}

void LayerNorm::setBeta(float* beta_weights) {
    cudaMemcpy(beta, beta_weights, hidden_dim * sizeof(float), cudaMemcpyDeviceToDevice);
}
