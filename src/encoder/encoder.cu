#include "encoder/encoder.cuh"
#include "layers/layer_norm.cuh"
#include "utils/utils.cuh"
#include <cuda_runtime.h>
#include <cuda.h>

Encoder::Encoder(const Config &config)
{
    num_layers = config.num_layers;
    hidden_dim = config.hidden_dim;
    num_heads = config.num_heads;
    intermediate_dim = config.intermediate_dim;

    // Allocate arrays for each layer's components
    self_attention_layers = new MultiHeadAttention *[num_layers];
    feed_forward_layers = new FeedForward *[num_layers];
    layer_norm1_layers = new LayerNorm *[num_layers];
    layer_norm2_layers = new LayerNorm *[num_layers];

    // Initialize components for each layer
    for (int i = 0; i < num_layers; ++i)
    {
        self_attention_layers[i] = new MultiHeadAttention(hidden_dim, num_heads);
        feed_forward_layers[i] = new FeedForward(hidden_dim, intermediate_dim);
        layer_norm1_layers[i] = new LayerNorm(hidden_dim);
        layer_norm2_layers[i] = new LayerNorm(hidden_dim);
    }
}

Encoder::~Encoder()
{
    // Delete components of each layer
    for (int i = 0; i < num_layers; ++i)
    {
        delete self_attention_layers[i];
        delete feed_forward_layers[i];
        delete layer_norm1_layers[i];
        delete layer_norm2_layers[i];
    }
    delete[] self_attention_layers;
    delete[] feed_forward_layers;
    delete[] layer_norm1_layers;
    delete[] layer_norm2_layers;
}

void Encoder::forward(float *output, const float *input, int batch_size, int seq_len, cudaStream_t stream)
{
    // Allocate memory for intermediate outputs (update sizes to account for batch dimension)
    float *current_input = nullptr;
    float *current_output = nullptr;
    cudaMalloc(&current_input, batch_size * seq_len * hidden_dim * sizeof(float));
    cudaMalloc(&current_output, batch_size * seq_len * hidden_dim * sizeof(float));
    cudaMemcpy(current_input, input, batch_size * seq_len * hidden_dim * sizeof(float), cudaMemcpyDeviceToDevice);

    for (int i = 0; i < num_layers; ++i)
    {
        // Layer Norm 1
        layer_norm1_layers[i]->forward(current_output, current_input, seq_len, stream);

        // Self-Attention
        self_attention_layers[i]->forward(current_output, current_output, batch_size, seq_len, stream);

        // Add & Norm
        add_tensors(current_output, current_input, current_output, batch_size * seq_len * hidden_dim, stream);

        // Layer Norm 2
        layer_norm2_layers[i]->forward(current_output, current_output, seq_len, stream);

        // Feed Forward
        feed_forward_layers[i]->forward(current_output, current_output, seq_len, stream);

        // Add & Prepare for next layer
        add_tensors(current_output, current_input, current_output, batch_size * seq_len * hidden_dim, stream);

        // Swap pointers for next layer
        std::swap(current_input, current_output);
    }

    // Copy the final output
    cudaMemcpy(output, current_input, batch_size * seq_len * hidden_dim * sizeof(float), cudaMemcpyDeviceToDevice);

    // Free intermediate memory
    cudaFree(current_input);
    cudaFree(current_output);
}
