#ifndef DECODER_H
#define DECODER_H

#include "../config.cuh"
#include "../layers/multihead_attention.cuh"
#include "../layers/feed_forward.cuh"
#include "../layers/layer_norm.cuh"

class Decoder
{
private:
    // Decoder stack configuration
    int num_layers;
    int hidden_dim;
    int num_heads;
    int intermediate_dim;

    // Components for each layer
    MultiHeadAttention **self_attention_layers;
    MultiHeadAttention **encoder_attention_layers;
    FeedForward **feed_forward_layers;
    LayerNorm **layer_norm1_layers;
    LayerNorm **layer_norm2_layers;
    LayerNorm **layer_norm3_layers;

public:
    Decoder(const Config &config);
    ~Decoder();

    void forward(float *output,
                 const float *input,
                 const float *encoder_output,
                 int batch_size,
                 int seq_len,
                 cudaStream_t stream);
};

#endif // DECODER_H
