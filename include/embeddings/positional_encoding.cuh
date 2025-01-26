#pragma once

#include <cuda_runtime.h>

class GPT2Weights;

// A layer class for adding GPT-2 position embeddings
class WPELayer
{
public:
    // Constructor: store pointer to the GPT2Weights
    explicit WPELayer(const GPT2Weights *weights);

    // forward(): Add position embeddings for each token in the batch
    // d_input_embeddings shape: [batch_size * seq_len, hidden_dim]
    // seq_len: length of the sequence so far
    // batch_size: number of samples in the batch
    // stream: CUDA stream
    void forward(float *d_input_embeddings,
                 int seq_len,
                 int batch_size,
                 cudaStream_t stream);

private:
    const GPT2Weights *weights_;
};
