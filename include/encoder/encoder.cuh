#ifndef ENCODER_H
#define ENCODER_H

#include "../config.cuh"
#include "../layers/multihead_attention.cuh"
#include "../layers/feed_forward.cuh"
#include "../layers/layer_norm.cuh"

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
    Encoder(const Config &config);
    ~Encoder();

    void forward(float *output, const float *input, int batch_size, int seq_len, cudaStream_t stream);
    void loadWeights(const GPT2Weights* weights);
};

#endif // ENCODER_H
