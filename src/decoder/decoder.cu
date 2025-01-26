#include "decoder/decoder.cuh"
#include "layers/layer_norm.cuh"
#include "utils/utils.cuh"
#include "utils/debug.cuh"
#include <cuda_runtime.h>
#include <cuda.h>

/**
 * @brief Constructs a Decoder with the given configuration and weights
 * @param config Configuration object containing model parameters
 * @param weights GPT2Weights object containing model weights
 *
 * Initializes a transformer decoder with the specified parameters and weights.
 * Allocates memory for all layer components and initializes them with the provided weights.
 */
Decoder::Decoder(const Config &config, const GPT2Weights* weights)
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

        // Initialize encoder-decoder attention similarly
        encoder_attention_layers[i] = new MultiHeadAttention(
            hidden_dim,
            num_heads,
            layer.attn_qkv_weight,
            layer.attn_qkv_weight + hidden_dim * hidden_dim,
            layer.attn_qkv_weight + 2 * hidden_dim * hidden_dim,
            layer.attn_proj_weight,
            layer.attn_qkv_bias,
            layer.attn_qkv_bias + hidden_dim,
            layer.attn_qkv_bias + 2 * hidden_dim,
            layer.attn_proj_bias
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
    }
    delete[] self_attention_layers;
    delete[] encoder_attention_layers;
    delete[] feed_forward_layers;
    delete[] layer_norm1_layers;
    delete[] layer_norm2_layers;
}

void Decoder::forward(float *output,
                     const float *input,
                     const float *encoder_output,
                     int batch_size,
                     int seq_len,
                     cudaStream_t stream)
{
    // Verify required input parameters
    if (!output || !input) {
        throw std::runtime_error("Null pointer passed for required decoder forward parameters");
    }
    
    // Validate dimensions first
    if (batch_size <= 0 || seq_len <= 0) {
        throw std::runtime_error("Invalid batch_size or seq_len");
    }

    // Check if CUDA is initialized and we have a valid device
    int deviceCount;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        throw std::runtime_error("No CUDA devices available");
    }

    // Get and verify current device
    int device;
    CUDA_CHECK(cudaGetDevice(&device));
    if (device >= deviceCount) {
        throw std::runtime_error("Invalid CUDA device");
    }

    // Ensure we have an active CUDA context
    CUDA_CHECK(cudaSetDevice(device));
    
    // Validate stream
    if (stream == nullptr) {
        stream = cudaStreamDefault;
    }
    
    // Calculate required memory size for current sequence length
    size_t tensor_size = batch_size * seq_len * hidden_dim * sizeof(float);
    
    // Allocate memory for intermediate outputs
    float *current_input = nullptr;
    float *current_output = nullptr;
    float *residual = nullptr;
    
    CUDA_CHECK(cudaMalloc(&current_input, tensor_size));
    if (current_input == nullptr) {
        throw std::runtime_error("Failed to allocate current_input buffer");
    }
    
    CUDA_CHECK(cudaMalloc(&current_output, tensor_size));
    if (current_output == nullptr) {
        cudaFree(current_input);
        throw std::runtime_error("Failed to allocate current_output buffer");
    }
    
    CUDA_CHECK(cudaMalloc(&residual, tensor_size));
    if (residual == nullptr) {
        cudaFree(current_input);
        cudaFree(current_output);
        throw std::runtime_error("Failed to allocate residual buffer");
    }

    // Copy input sequence to current_input
    CUDA_CHECK(cudaMemcpy(current_input, input, tensor_size, cudaMemcpyDeviceToDevice));

    cudaPointerAttributes attributes;
    CUDA_CHECK(cudaPointerGetAttributes(&attributes, input));
    if (attributes.type != cudaMemoryTypeDevice) {
        throw std::runtime_error("Input must be a device pointer");
    }

    debugPrint("Starting decoder forward pass with num_layers: %d\n", num_layers);
    for (int i = 0; i < num_layers; ++i)
    {
        debugPrint("Processing layer %d\n", i);

        // Store the current input as residual
        debugPrint("Storing current input as residual for layer %d\n", i);
        CUDA_CHECK(cudaMemcpy(residual, current_input, batch_size * seq_len * hidden_dim * sizeof(float), 
                             cudaMemcpyDeviceToDevice));

        // Masked Self-Attention
        debugPrint("Performing Masked Self-Attention for layer %d\n", i);
        self_attention_layers[i]->forward(current_output, current_input, batch_size, seq_len, stream, /*mask=*/true);

        // Add & Norm
        debugPrint("Performing Add & Norm after self-attention for layer %d\n", i);
        add_tensors(current_output, residual, current_output, batch_size * seq_len * hidden_dim, stream);

        // Prepare residual for next sublayer
        debugPrint("Preparing residual for next sublayer in layer %d\n", i);
        CUDA_CHECK(cudaMemcpy(residual, current_output, batch_size * seq_len * hidden_dim * sizeof(float), 
                             cudaMemcpyDeviceToDevice));

        // Only perform encoder-decoder attention if encoder_output is provided
        if (encoder_output) {
            debugPrint("Encoder output provided, performing encoder-decoder attention for layer %d\n", i);

            // Layer Norm 1
            debugPrint("Applying Layer Norm 1 for layer %d\n", i);
            layer_norm1_layers[i]->forward(current_output, current_output, batch_size * seq_len, stream);

            // Encoder-Decoder Attention
            debugPrint("Performing Encoder-Decoder Attention for layer %d\n", i);
            encoder_attention_layers[i]->forward(current_output, current_output, encoder_output, batch_size, seq_len, stream);

            // Add & Norm
            debugPrint("Performing Add & Norm after encoder-decoder attention for layer %d\n", i);
            add_tensors(current_output, residual, current_output, batch_size * seq_len * hidden_dim, stream);

            debugPrint("Updating residual after encoder-decoder attention for layer %d\n", i);
            CUDA_CHECK(cudaMemcpy(residual, current_output, batch_size * seq_len * hidden_dim * sizeof(float), 
                                cudaMemcpyDeviceToDevice));
        }

        // Feed Forward
        debugPrint("Performing Feed Forward for layer %d\n", i);
        feed_forward_layers[i]->forward(current_output, current_output, seq_len, stream);

        // Add & Norm
        debugPrint("Performing final Add & Norm for layer %d\n", i);
        add_tensors(current_output, residual, current_output, batch_size * seq_len * hidden_dim, stream);

        // Swap pointers for next layer
        debugPrint("Swapping pointers for next layer %d\n", i);
        std::swap(current_input, current_output);
    }

    // Copy the final output
    debugPrint("Copying final output to destination\n");
    CUDA_CHECK(cudaMemcpy(output, current_input, batch_size * seq_len * hidden_dim * sizeof(float), 
                         cudaMemcpyDeviceToDevice));

    // Free intermediate memory
    debugPrint("Freeing intermediate memory\n");
    CUDA_CHECK(cudaFree(current_input));
    CUDA_CHECK(cudaFree(current_output));
    CUDA_CHECK(cudaFree(residual));
    debugPrint("Decoder forward pass completed\n");
}
