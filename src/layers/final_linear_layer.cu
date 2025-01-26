#include <vector>
#include <iostream>
#include "layers/final_linear_layer.cuh"
#include "utils/softmax.cuh"
#include "utils/utils.cuh"
#include <cuda_runtime.h>
#include <algorithm>
#include <numeric>

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
        // Use FP32 accumulator for better numerical stability
        #pragma unroll
        for (int k = 0; k < hidden_dim; k++)
        {
            // Ensure proper memory access pattern
            float input_val = input[col * hidden_dim + k];
            float weight_val = weights[row * hidden_dim + k];
            sum = fmaf(weight_val, input_val, sum);  // Use fmaf for better precision
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
    // We don't need to allocate weights here since we'll use token embeddings
    d_linear_weights_ = nullptr;
}

/**
 * @brief Destructor for the FinalLinearLayer class
 *
 * Cleans up all allocated memory for layer components including
 * weights and layer normalization.
 */
FinalLinearLayer::~FinalLinearLayer()
{
    // No need to free weights since they're managed by token embeddings
}

/**
 * @brief Allocates memory for layer weights
 *
 * Allocates GPU memory for the weight matrix used in linear transformation.
 */
void FinalLinearLayer::allocateWeights()
{
    // No allocation needed - weights will come from token embeddings
}

/**
 * @brief Frees allocated weight memory
 *
 * Releases GPU memory used for weights when layer is destroyed.
 */
void FinalLinearLayer::freeWeights()
{
    // No freeing needed - weights are managed by token embeddings
}

/**
 * @brief Performs forward pass through final linear layer using token embeddings
 * @param d_input Input hidden states
 * @param d_logits Output logits
 * @param seq_len Sequence length
 * @param d_token_embeddings Token embedding weights to use for linear projection
 */
void FinalLinearLayer::forward(float *d_input, float *d_logits, int seq_len, float *d_token_embeddings)
{
    // Use token embeddings as the weight matrix
    d_linear_weights_ = d_token_embeddings;

    // Dimensions for the linear layer
    int vocab_size = config_.vocab_size;
    int batch_seq_len = config_.batch_size * seq_len;
    int hidden_dim = config_.hidden_dim;

    // Define block and grid dimensions
    dim3 blockDim(16, 16);
    dim3 gridDim(
        (vocab_size + blockDim.x - 1) / blockDim.x,
        (batch_seq_len + blockDim.y - 1) / blockDim.y);

    // Launch linear transformation kernel using token embeddings as weights
    linearTransformKernel<<<gridDim, blockDim>>>(
        d_input,
        d_token_embeddings,  // Use token embeddings as weights
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

    // Debug: Print the output logits before softmax
    std::vector<float> h_logits(batch_seq_len * vocab_size);
    cudaMemcpy(h_logits.data(), d_logits, batch_seq_len * vocab_size * sizeof(float), cudaMemcpyDeviceToHost);
    debugPrint("Logits before softmax: ");
    for (int i = 0; i < std::min(10, static_cast<int>(h_logits.size())); ++i) {
        debugPrint("%f ", h_logits[i]);
    }
    debugPrint("\n");

    // Sort logits and get indexes of the top 5
    std::vector<int> indices(h_logits.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::partial_sort(indices.begin(), indices.begin() + 5, indices.end(),
                      [&h_logits](int a, int b) { return h_logits[a] > h_logits[b]; });

    // Debug: Print the top 5 logits and their indices
    debugPrint("Top 5 logits before softmax: ");
    for (int i = 0; i < 5; ++i) {
        debugPrint("Index: %d, Logit: %f ", indices[i], h_logits[indices[i]]);
    }
    debugPrint("\n");

    // Apply softmax to the logits
    applySoftmax(cudnn_, d_logits, d_logits, batch_seq_len, vocab_size);

    // Debug: Print the top logits after softmax
    cudaMemcpy(h_logits.data(), d_logits, batch_seq_len * vocab_size * sizeof(float), cudaMemcpyDeviceToHost);
    debugPrint("Logits after softmax: ");
    for (int i = 0; i < std::min(10, static_cast<int>(h_logits.size())); ++i) {
        debugPrint("%f ", h_logits[i]);
    }
    debugPrint("\n");

    // Verify the sum of the logits after softmax
    for (int i = 0; i < batch_seq_len; ++i) {
        float sum = 0.0f;
        for (int j = 0; j < vocab_size; ++j) {
            sum += h_logits[i * vocab_size + j];
        }
        debugPrint("Sum of logits for sequence %d: %f\n", i, sum);
    }

    // Sort logits after softmax and get indexes of the top 5
    std::partial_sort(indices.begin(), indices.begin() + 5, indices.end(),
                      [&h_logits](int a, int b) { return h_logits[a] > h_logits[b]; });

    // Debug: Print the top 5 logits and their indices after softmax
    debugPrint("Top 5 logits after softmax: ");
    for (int i = 0; i < 5; ++i) {
        debugPrint("Index: %d, Logit: %f ", indices[i], h_logits[indices[i]]);
    }
    debugPrint("\n");
}