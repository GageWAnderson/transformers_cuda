#include "embeddings/positional_encoding.cuh"
#include "gpt2_weights.cuh"
#include <cuda_runtime.h>

// Kernel to add the position embedding to each token embedding.
__global__ void wpeForwardKernel(float *d_input_embeddings,
                                 const float *d_position_embedding,
                                 int seq_len,
                                 int batch_size,
                                 int hidden_dim,
                                 int position_offset)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x; 
    int total_tokens = seq_len * batch_size; // e.g. batch_size * seq_len

    // Flattened size in elements across (batch_size * seq_len * hidden_dim)
    int total_size = total_tokens * hidden_dim;

    if (idx < total_size)
    {
        // token_index in [0..(batch_size * seq_len - 1)]
        int token_index = idx / hidden_dim;  
        int emb_col     = idx % hidden_dim;

        // Offset the position by position_offset.
        int position = position_offset + (token_index % seq_len);

        // Add the row from GPT2 position embedding
        d_input_embeddings[idx] += d_position_embedding[position * hidden_dim + emb_col];
    }
}

WPELayer::WPELayer(const GPT2Weights *weights)
    : weights_(weights)
{
    // Nothing else needed, as we just reference weights_->getPositionEmbedding()
}

void WPELayer::forward(float *d_input_embeddings,
                       int seq_len,
                       int batch_size,
                       cudaStream_t stream,
                       int position_offset)
{
    if (seq_len <= 0 || batch_size <= 0) return;

    int hidden_dim             = weights_->getDims().hidden_dim;
    const float *d_pos_embeds  = weights_->getPositionEmbedding();

    // For convenience:
    int total_tokens = batch_size * seq_len;
    int total_elements = total_tokens * hidden_dim;

    int blockSize = 256;
    int gridSize  = (total_elements + blockSize - 1) / blockSize;

    wpeForwardKernel<<<gridSize, blockSize, 0, stream>>>(
        d_input_embeddings,
        d_pos_embeds,
        seq_len,
        batch_size,
        hidden_dim,
        position_offset
    );
    cudaStreamSynchronize(stream);
}
