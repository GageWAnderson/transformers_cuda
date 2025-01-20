#ifndef DECODER_H
#define DECODER_H

#include "../config.cuh"
#include "../layers/multihead_attention.cuh"
#include "../layers/feed_forward.cuh"
#include "../layers/layer_norm.cuh"
#include "gpt2_weights.cuh"

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
    // Updated constructor to take weights
    Decoder(const Config &config, const GPT2Weights* weights);
    ~Decoder();

    /**
     * @brief Forward pass through the decoder
     * @param output Output tensor
     * @param input Input tensor
     * @param encoder_output Optional encoder output tensor (can be null for decoder-only models)
     * @param batch_size Batch size
     * @param seq_len Sequence length
     * @param stream CUDA stream
     */
    void forward(float *output,
                const float *input,
                const float *encoder_output,  // Optional for decoder-only models
                int batch_size,
                int seq_len,
                cudaStream_t stream);
};

#endif // DECODER_H
