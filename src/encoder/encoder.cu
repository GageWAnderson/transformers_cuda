#include "encoder/encoder.cuh"
#include "layers/layer_norm.cuh"
#include "utils/utils.cuh"
#include <cuda_runtime.h>
#include <cuda.h>

/**
 * @brief Constructs an Encoder with the given configuration
 * @param config Configuration object containing model parameters
 *
 * Initializes a transformer encoder with the specified number of layers,
 * hidden dimensions, attention heads, and intermediate dimensions.
 * Allocates memory for all layer components including self-attention,
 * feed-forward networks, and layer normalization.
 */
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
        // Initialize weights and biases as null
        float *W_q_ptr = nullptr, *W_k_ptr = nullptr, *W_v_ptr = nullptr, *W_o_ptr = nullptr;
        float *b_q_ptr = nullptr, *b_k_ptr = nullptr, *b_v_ptr = nullptr, *b_o_ptr = nullptr;
        float *W1_ptr = nullptr, *b1_ptr = nullptr, *W2_ptr = nullptr, *b2_ptr = nullptr;

        self_attention_layers[i] = new MultiHeadAttention(hidden_dim, num_heads, W_q_ptr, W_k_ptr, W_v_ptr, W_o_ptr);
        feed_forward_layers[i] = new FeedForward(hidden_dim, intermediate_dim, W1_ptr, b1_ptr, W2_ptr, b2_ptr);
        layer_norm1_layers[i] = new LayerNorm(hidden_dim);
        layer_norm2_layers[i] = new LayerNorm(hidden_dim);
    }
}

/**
 * @brief Destructor for the Encoder class
 *
 * Cleans up all allocated memory for layer components including
 * self-attention layers, feed-forward networks, and layer normalization.
 */
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

/**
 * @brief Performs forward pass through the encoder
 * @param output Pointer to output tensor on device
 * @param input Pointer to input tensor on device
 * @param batch_size Number of sequences in batch
 * @param seq_len Length of input sequences
 * @param stream CUDA stream for asynchronous execution
 *
 * Processes input through multiple encoder layers with self-attention,
 * feed-forward networks, residual connections and layer normalization.
 * Uses multiple CUDA streams for parallel operations where possible.
 */
void Encoder::forward(float *output, const float *input, int batch_size, int seq_len, cudaStream_t stream)
{
    // Allocate memory for intermediate outputs (update sizes to account for batch dimension)
    float *current_input = nullptr;
    float *current_output = nullptr;
    float *residual = nullptr;
    cudaMalloc(&current_input, batch_size * seq_len * hidden_dim * sizeof(float));
    cudaMalloc(&current_output, batch_size * seq_len * hidden_dim * sizeof(float));
    cudaMalloc(&residual, batch_size * seq_len * hidden_dim * sizeof(float));
    cudaMemcpy(current_input, input, batch_size * seq_len * hidden_dim * sizeof(float), cudaMemcpyDeviceToDevice);

    for (int i = 0; i < num_layers; ++i)
    {
        // Store the current input as residual
        cudaMemcpy(residual, current_input, batch_size * seq_len * hidden_dim * sizeof(float), cudaMemcpyDeviceToDevice);

        // Create separate streams for parallel operations within the layer
        cudaStream_t norm_stream, attn_stream;
        cudaStreamCreate(&norm_stream);
        cudaStreamCreate(&attn_stream);

        // Layer Norm 1 and Self-Attention can start in parallel
        layer_norm1_layers[i]->forward(current_output, current_input, seq_len, norm_stream);

        // Wait for norm to complete before attention
        cudaStreamSynchronize(norm_stream);
        self_attention_layers[i]->forward(current_output, current_output, batch_size, seq_len, attn_stream);

        // Add & Norm
        cudaStreamSynchronize(attn_stream);
        add_tensors(current_output, residual, current_output, batch_size * seq_len * hidden_dim, stream);

        // Layer Norm 2
        layer_norm2_layers[i]->forward(current_output, current_output, seq_len, stream);

        // Feed Forward
        feed_forward_layers[i]->forward(current_output, current_output, seq_len, stream);

        // Add & Prepare for next layer
        add_tensors(current_output, residual, current_output, batch_size * seq_len * hidden_dim, stream);

        // Clean up streams
        cudaStreamDestroy(norm_stream);
        cudaStreamDestroy(attn_stream);

        // Swap pointers for next layer
        std::swap(current_input, current_output);
    }

    // Copy the final output
    cudaMemcpy(output, current_input, batch_size * seq_len * hidden_dim * sizeof(float), cudaMemcpyDeviceToDevice);

    // Free intermediate memory
    cudaFree(current_input);
    cudaFree(current_output);
    cudaFree(residual);
}
