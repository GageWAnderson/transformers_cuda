#include <vector>
#include <iostream>
#include "layers/final_linear_layer.cuh"
#include "utils/softmax.cuh"
#include "utils/utils.cuh"
#include <cuda_runtime.h>

/**
 * @brief CUDA kernel for linear transformation
 * @param input Input tensor
 * @param weights Weight matrix
 * @param output Output tensor
 * @param vocab_size Size of vocabulary
 * @param batch_seq_len Combined batch and sequence length
 * @param hidden_dim Hidden dimension size
 *
 * Performs matrix multiplication between input and weights to produce logits.
 * Each thread computes one element of the output matrix.
 */
__global__ void linearTransformKernel(const float *input, const float *weights, float *output,
                                      int vocab_size, int batch_seq_len, int hidden_dim)
{
    // Calculate global thread indices
    int row = blockIdx.x * blockDim.x + threadIdx.x; // For vocab_size dimension
    int col = blockIdx.y * blockDim.y + threadIdx.y; // For batch_seq_len dimension

    if (row < vocab_size && col < batch_seq_len)
    {
        float sum = 0.0f;
        // Perform dot product between input and weights
        for (int k = 0; k < hidden_dim; k++)
        {
            sum += weights[row * hidden_dim + k] * input[col * hidden_dim + k];
        }
        output[col * vocab_size + row] = sum;
    }
}

/**
 * @brief Constructs final linear layer with configuration and weights
 * @param config Model configuration
 * @param cublas_handle cuBLAS handle
 * @param cudnn_handle cuDNN handle
 * @param weights GPT2Weights object containing model weights
 *
 * Initializes final linear layer and loads weights for projecting hidden states to vocabulary size.
 */
FinalLinearLayer::FinalLinearLayer(const Config &config,
                                   cublasHandle_t &cublas_handle,
                                   cudnnHandle_t &cudnn_handle,
                                   const GPT2Weights *weights)
    : config_(config), cublas_(cublas_handle), cudnn_(cudnn_handle)
{
    // Allocate memory for weights
    allocateWeights();

    if (weights)
    {
        // Calculate sizes
        size_t weights_size = config_.hidden_dim * config_.vocab_size * sizeof(float);
        size_t bias_size = config_.vocab_size * sizeof(float);

        // Copy weights to device
        cudaMemcpy(d_linear_weights_, weights->getFinalLayerNormWeight(),
                   weights_size, cudaMemcpyHostToDevice);

        // Copy bias if available
        float *bias = weights->getFinalLayerNormBias();
        if (bias)
        {
            cudaMemcpy(d_linear_bias_, bias, bias_size, cudaMemcpyHostToDevice);
        }
    }
}

/**
 * @brief Destructor for the FinalLinearLayer class
 *
 * Cleans up all allocated memory for layer components including
 * weights and layer normalization.
 */
FinalLinearLayer::~FinalLinearLayer()
{
    freeWeights();
}

/**
 * @brief Allocates memory for layer weights
 *
 * Allocates GPU memory for the weight matrix used in linear transformation.
 */
void FinalLinearLayer::allocateWeights()
{
    size_t weights_size = config_.hidden_dim * config_.vocab_size * sizeof(float);
    cudaMalloc(&d_linear_weights_, weights_size);
}

/**
 * @brief Frees allocated weight memory
 *
 * Releases GPU memory used for weights when layer is destroyed.
 */
void FinalLinearLayer::freeWeights()
{
    if (d_linear_weights_)
    {
        cudaFree(d_linear_weights_);
        d_linear_weights_ = nullptr;
    }

    if (d_linear_bias_)
    {
        cudaFree(d_linear_bias_);
        d_linear_bias_ = nullptr;
    }
}

/**
 * @brief Performs forward pass through final linear layer
 * @param d_input Input hidden states
 * @param d_logits Output logits
 * @param seq_len Sequence length
 *
 * Projects hidden states to vocabulary size using linear transformation,
 * then applies softmax to get probability distribution over vocabulary.
 */
void FinalLinearLayer::forward(float *d_input, float *d_logits, int seq_len)
{
    // Dimensions for the linear layer
    int vocab_size = config_.vocab_size;
    int batch_seq_len = config_.batch_size * seq_len;
    int hidden_dim = config_.hidden_dim;

    // Define block and grid dimensions
    dim3 blockDim(16, 16); // 256 threads per block
    dim3 gridDim(
        (vocab_size + blockDim.x - 1) / blockDim.x,
        (batch_seq_len + blockDim.y - 1) / blockDim.y);

    std::cout << "Dimensions - vocab_size: " << vocab_size
              << ", batch_seq_len: " << batch_seq_len
              << ", hidden_dim: " << hidden_dim << std::endl;
    std::cout << "Grid dims - x: " << gridDim.x << ", y: " << gridDim.y << std::endl;

    // Launch custom linear transformation kernel
    linearTransformKernel<<<gridDim, blockDim>>>(
        d_input,
        d_linear_weights_,
        d_logits,
        vocab_size,
        batch_seq_len,
        hidden_dim);

    // Check for kernel launch errors
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        std::cerr << "CUDA error in linear transform kernel: " << cudaGetErrorString(error) << std::endl;
    }

    // If bias is available, add it to the output
    if (d_linear_bias_)
    {
        // TODO: Add bias addition kernel call here
        // This would need to be implemented as a separate CUDA kernel
    }

    // Apply softmax to the logits
    applySoftmax(cudnn_, d_logits, d_logits, batch_seq_len, vocab_size);
}
