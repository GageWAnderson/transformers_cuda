#pragma once

#include <vector>

class GPT2Weights;

class WTELayer
{
public:
    explicit WTELayer(const GPT2Weights *weights);

    void forward(const std::vector<int> &host_tokens,
                 float *d_output,
                 int batch_size,
                 int seq_len,
                 cudaStream_t stream);

private:
    const GPT2Weights *weights_;
};