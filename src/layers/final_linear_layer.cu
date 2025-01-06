#include <vector>
#include <iostream>
#include "layers/final_linear_layer.cuh"
#include "utils/weight_init.cuh"
#include "utils/softmax.cuh"
#include "utils/utils.cuh"
#include <cuda_runtime.h>

__global__ void linearTransformKernel(const float* input, const float* weights, float* output,
                                    int vocab_size, int batch_seq_len, int hidden_dim) {
    // Calculate global thread indices
    int row = blockIdx.x * blockDim.x + threadIdx.x; // For vocab_size dimension
    int col = blockIdx.y * blockDim.y + threadIdx.y; // For batch_seq_len dimension
    
    if (row < vocab_size && col < batch_seq_len) {
        float sum = 0.0f;
        // Perform dot product between input and weights
        for (int k = 0; k < hidden_dim; k++) {
            sum += weights[row * hidden_dim + k] * input[col * hidden_dim + k];
        }
        output[col * vocab_size + row] = sum;
    }
}

FinalLinearLayer::FinalLinearLayer(const Config &config,
                                   cublasHandle_t &cublas_handle,
                                   cudnnHandle_t &cudnn_handle,
                                   curandGenerator_t &curand_gen)
    : config_(config), cublas_(cublas_handle), cudnn_(cudnn_handle), curand_gen_(curand_gen)
{
}

FinalLinearLayer::~FinalLinearLayer()
{
    freeWeights();
}

void FinalLinearLayer::initialize()
{
    allocateWeights();

    // Initialize weights with random values
    size_t weight_size = config_.hidden_dim * config_.vocab_size;
    initializeWeights(curand_gen_, d_linear_weights_, weight_size);
}

void FinalLinearLayer::allocateWeights()
{
    size_t weights_size = config_.hidden_dim * config_.vocab_size * sizeof(float);
    cudaMalloc(&d_linear_weights_, weights_size);
}

void FinalLinearLayer::freeWeights()
{
    if (d_linear_weights_)
    {
        cudaFree(d_linear_weights_);
        d_linear_weights_ = nullptr;
    }
}

void FinalLinearLayer::forward(float *d_input, float *d_logits, int seq_len)
{
    // Dimensions for the linear layer
    int vocab_size = config_.vocab_size;
    int batch_seq_len = config_.batch_size * seq_len;
    int hidden_dim = config_.hidden_dim;

    // Define block and grid dimensions
    dim3 blockDim(16, 16);  // 256 threads per block
    dim3 gridDim(
        (vocab_size + blockDim.x - 1) / blockDim.x,
        (batch_seq_len + blockDim.y - 1) / blockDim.y
    );

    // Launch custom linear transformation kernel
    linearTransformKernel<<<gridDim, blockDim>>>(
        d_input,
        d_linear_weights_,
        d_logits,
        vocab_size,
        batch_seq_len,
        hidden_dim
    );

    // Check for kernel launch errors
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << "CUDA error in linear transform kernel: " << cudaGetErrorString(error) << std::endl;
    }

    // Apply softmax to the logits
    applySoftmax(cudnn_, d_logits, d_logits, vocab_size, batch_seq_len);

    // Print the first 10 logits before softmax
    std::vector<float> h_logits_before(batch_seq_len * vocab_size);
    cudaMemcpy(h_logits_before.data(), d_logits, batch_seq_len * vocab_size * sizeof(float), cudaMemcpyDeviceToHost);
    std::cout << "Logits before softmax (first 10 elements): ";
    for (int i = 0; i < 10 && i < h_logits_before.size(); ++i)
    {
        std::cout << h_logits_before[i] << " ";
    }
    std::cout << "\n";

    // Print the first 10 logits after softmax
    std::vector<float> h_logits_after(batch_seq_len * vocab_size);
    cudaMemcpy(h_logits_after.data(), d_logits, batch_seq_len * vocab_size * sizeof(float), cudaMemcpyDeviceToHost);
    std::cout << "Logits after softmax (first 10 elements): ";
    for (int i = 0; i < 10 && i < h_logits_after.size(); ++i)
    {
        std::cout << h_logits_after[i] << " ";
    }
    std::cout << "\n";

    // Optionally, you can process or log d_logits here

    // Removed internal allocation and deallocation of d_logits
    // No cudaMalloc or cudaFree for d_logits in this function
}
