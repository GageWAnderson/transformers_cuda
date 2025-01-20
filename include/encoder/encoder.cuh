#ifndef ENCODER_H
#define ENCODER_H

#include "../config.cuh"
#include "../layers/multihead_attention.cuh"
#include "../layers/feed_forward.cuh"
#include "../layers/layer_norm.cuh"
#include "gpt2_weights.cuh"

class Encoder
{
private:
    // Encoder stack configuration
    int num_layers;
    int hidden_dim;
    int num_heads;
    int intermediate_dim;

    // Components for each layer
    MultiHeadAttention **self_attention_layers;
    FeedForward **feed_forward_layers;
    LayerNorm **layer_norm1_layers;
    LayerNorm **layer_norm2_layers;

public:
    // Updated constructor to take weights
    Encoder(const Config &config, const GPT2Weights* weights);
    ~Encoder();

    void forward(float *output, const float *input, int batch_size, int seq_len, cudaStream_t stream);
};

#endif // ENCODER_H
