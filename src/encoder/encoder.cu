#include "encoder/encoder.cuh"
#include "layers/layer_norm.cuh"
#include "utils/utils.cuh"
#include "gpt2_weights.cuh"
#include <cuda_runtime.h>
#include <cuda.h>

/**
 * @brief Constructs an Encoder with the given configuration and weights
 * @param config Configuration object containing model parameters
 * @param weights GPT2Weights object containing model weights
 *
 * Initializes a transformer encoder with the specified parameters and weights.
 * Allocates memory for all layer components and initializes them with the provided weights.
 */
Encoder::Encoder(const Config &config, const GPT2Weights* weights)
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

    // Initialize components for each layer with weights
    for (int i = 0; i < num_layers; ++i)
    {
        // Get layer weights
        const LayerWeights& layer = weights->getLayerWeights(i);
        
        // Initialize self attention with direct weight references
        self_attention_layers[i] = new MultiHeadAttention(
            hidden_dim, 
            num_heads,
            layer.attn_qkv_weight,     // Q weight
            layer.attn_qkv_weight + hidden_dim * hidden_dim,  // K weight offset
            layer.attn_qkv_weight + 2 * hidden_dim * hidden_dim,  // V weight offset
            layer.attn_proj_weight,    // Output projection weight
            layer.attn_qkv_bias,       // Q bias
            layer.attn_qkv_bias + hidden_dim,  // K bias offset
            layer.attn_qkv_bias + 2 * hidden_dim,  // V bias offset
            layer.attn_proj_bias       // Output projection bias
        );

        // Initialize feed forward with direct weight references
        feed_forward_layers[i] = new FeedForward(
            hidden_dim, 
            intermediate_dim,
            layer.ffn_fc1_weight,  // W1
            layer.ffn_fc1_bias,    // b1 
            layer.ffn_fc2_weight,  // W2
            layer.ffn_fc2_bias     // b2
        );

        // Initialize layer norms with weights directly in constructor
        layer_norm1_layers[i] = new LayerNorm(hidden_dim, 
                                             layer.attn_ln_weight,
                                             layer.attn_ln_bias);

        layer_norm2_layers[i] = new LayerNorm(hidden_dim,
                                             layer.ffn_ln_weight,
                                             layer.ffn_ln_bias);
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
