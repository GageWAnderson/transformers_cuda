#include <cuda_runtime.h>
#include <vector>
#include "embeddings/token_embeddings.cuh"
#include "gpt2_weights.cuh"
#include "utils/debug.cuh"

// Kernel: fetch the embedding row for each token and write to output
__global__ void wteForwardKernel(const int *d_tokens,
                                 const float *d_token_embedding,
                                 float *d_output,
                                 int hidden_dim,
                                 int total_tokens)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < total_tokens * hidden_dim)
    {
        int token_idx = idx / hidden_dim; // which token in the batch
        int emb_col = idx % hidden_dim;   // which dimension of the embedding
        int token_id = d_tokens[token_idx];
        d_output[idx] = d_token_embedding[token_id * hidden_dim + emb_col];
    }
}

WTELayer::WTELayer(const GPT2Weights *weights)
    : weights_(weights)
{
    // Nothing else needed, as we use weights_->getTokenEmbedding() in forward.
}

void WTELayer::forward(const std::vector<int> &host_tokens,
                       float *d_output,
                       int batch_size,
                       int seq_len,
                       cudaStream_t stream)
{
    // Copy tokens to device
    int total_tokens = batch_size * seq_len;
    if (total_tokens <= 0)
        return;

    // Allocate GPU buffer for tokens
    int *d_tokens = nullptr;
    cudaMalloc(&d_tokens, total_tokens * sizeof(int));
    cudaMemcpyAsync(d_tokens,
                    host_tokens.data(),
                    total_tokens * sizeof(int),
                    cudaMemcpyHostToDevice,
                    stream);

    // Kernel to copy rows from the token embedding
    int hidden_dim = weights_->getDims().hidden_dim;
    float *d_token_embedding = weights_->getTokenEmbedding();

    int blockSize = 256;
    int gridSize = (total_tokens * hidden_dim + blockSize - 1) / blockSize;
    wteForwardKernel<<<gridSize, blockSize, 0, stream>>>(d_tokens,
                                                         d_token_embedding,
                                                         d_output,
                                                         hidden_dim,
                                                         total_tokens);

    // Free device token indices
    cudaFree(d_tokens);
}