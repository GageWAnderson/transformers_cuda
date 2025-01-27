#pragma once

#include <cuda_runtime.h>

class GPT2Weights;

// A layer class for adding GPT-2 position embeddings
class WPELayer
{
public:
    // Constructor: store pointer to the GPT2Weights
    explicit WPELayer(const GPT2Weights *weights);

    void forward(float *d_input_embeddings,
                 int seq_len,
                 int batch_size,
                 cudaStream_t stream,
                 int position_offset);

private:
    const GPT2Weights *weights_;
};
