#include "decoder/decoder.cuh"
#include "layers/layer_norm.cuh"
#include "utils/utils.cuh"
#include <cuda_runtime.h>
#include <cuda.h>

Decoder::Decoder(const Config &config)
{
    num_layers = config.num_layers;
    hidden_dim = config.hidden_dim;
    num_heads = config.num_heads;
    intermediate_dim = config.intermediate_dim;

    // Allocate arrays for each layer's components
    self_attention_layers = new MultiHeadAttention *[num_layers];
    encoder_attention_layers = new MultiHeadAttention *[num_layers];
    feed_forward_layers = new FeedForward *[num_layers];
    layer_norm1_layers = new LayerNorm *[num_layers];
    layer_norm2_layers = new LayerNorm *[num_layers];
    layer_norm3_layers = new LayerNorm *[num_layers];

    // Initialize components for each layer
    for (int i = 0; i < num_layers; ++i)
    {
        self_attention_layers[i] = new MultiHeadAttention(hidden_dim, num_heads);
        encoder_attention_layers[i] = new MultiHeadAttention(hidden_dim, num_heads);
        feed_forward_layers[i] = new FeedForward(hidden_dim, intermediate_dim);
        layer_norm1_layers[i] = new LayerNorm(hidden_dim);
        layer_norm2_layers[i] = new LayerNorm(hidden_dim);
        layer_norm3_layers[i] = new LayerNorm(hidden_dim);
    }
}

Decoder::~Decoder()
{
    // Delete components of each layer
    for (int i = 0; i < num_layers; ++i)
    {
        delete self_attention_layers[i];
        delete encoder_attention_layers[i];
        delete feed_forward_layers[i];
        delete layer_norm1_layers[i];
        delete layer_norm2_layers[i];
        delete layer_norm3_layers[i];
    }
    delete[] self_attention_layers;
    delete[] encoder_attention_layers;
    delete[] feed_forward_layers;
    delete[] layer_norm1_layers;
    delete[] layer_norm2_layers;
    delete[] layer_norm3_layers;
}

void Decoder::forward(float *output,
                      const float *input,
                      const float *encoder_output,
                      int batch_size,
                      int seq_len,
                      cudaStream_t stream)
{
    // TODO: Implement the forward pass
}
