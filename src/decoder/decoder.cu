#include "decoder/decoder.cuh"
#include "layers/layer_norm.cuh"
#include "utils/utils.cuh"
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
    layer_norm3_layers = new LayerNorm *[num_layers];

    // Initialize components for each layer with weights
    for (int i = 0; i < num_layers; ++i)
    {
        // Initialize self attention with weights
        self_attention_layers[i] = new MultiHeadAttention(hidden_dim, num_heads);
        self_attention_layers[i]->setQueryWeight(weights->getAttentionQKVWeight(i));
        self_attention_layers[i]->setQueryBias(weights->getAttentionQKVBias(i));
        self_attention_layers[i]->setOutputProjWeight(weights->getAttentionProjectionWeight(i));
        self_attention_layers[i]->setOutputProjBias(weights->getAttentionProjectionBias(i));

        // Initialize encoder-decoder attention with weights
        encoder_attention_layers[i] = new MultiHeadAttention(hidden_dim, num_heads);
        encoder_attention_layers[i]->setQueryWeight(weights->getAttentionQKVWeight(i));
        encoder_attention_layers[i]->setQueryBias(weights->getAttentionQKVBias(i));
        encoder_attention_layers[i]->setOutputProjWeight(weights->getAttentionProjectionWeight(i));
        encoder_attention_layers[i]->setOutputProjBias(weights->getAttentionProjectionBias(i));

        // Initialize feed forward with weights
        feed_forward_layers[i] = new FeedForward(hidden_dim, intermediate_dim);
        feed_forward_layers[i]->setWeight1(weights->getFFNFC1Weight(i));
        feed_forward_layers[i]->setBias1(weights->getFFNFC1Bias(i));
        feed_forward_layers[i]->setWeight2(weights->getFFNFC2Weight(i));
        feed_forward_layers[i]->setBias2(weights->getFFNFC2Bias(i));

        // Initialize layer norms with weights
        layer_norm1_layers[i] = new LayerNorm(hidden_dim);
        layer_norm1_layers[i]->setGamma(weights->getAttentionLayerNormWeight(i));
        layer_norm1_layers[i]->setBeta(weights->getAttentionLayerNormBias(i));
        
        layer_norm2_layers[i] = new LayerNorm(hidden_dim);
        layer_norm2_layers[i]->setGamma(weights->getFFNLayerNormWeight(i));
        layer_norm2_layers[i]->setBeta(weights->getFFNLayerNormBias(i));

        layer_norm3_layers[i] = new LayerNorm(hidden_dim);
        layer_norm3_layers[i]->setGamma(weights->getFFNLayerNormWeight(i));
        layer_norm3_layers[i]->setBeta(weights->getFFNLayerNormBias(i));
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
    // Allocate memory for intermediate outputs
    float *current_input = nullptr;
    float *current_output = nullptr;
    float *residual = nullptr;
    cudaMalloc(&current_input, batch_size * seq_len * hidden_dim * sizeof(float));
    cudaMalloc(&current_output, batch_size * seq_len * hidden_dim * sizeof(float));
    cudaMalloc(&residual, batch_size * seq_len * hidden_dim * sizeof(float));

    // Copy input to current_input
    cudaMemcpy(current_input, input, batch_size * seq_len * hidden_dim * sizeof(float), cudaMemcpyDeviceToDevice);

    for (int i = 0; i < num_layers; ++i)
    {
        // Store the current input as residual
        cudaMemcpy(residual, current_input, batch_size * seq_len * hidden_dim * sizeof(float), cudaMemcpyDeviceToDevice);

        // Layer Norm 1
        layer_norm1_layers[i]->forward(current_input, current_input, seq_len, stream);

        // Masked Self-Attention
        // Note: For masked self-attention, you need to apply a mask to prevent attending to future positions.
        self_attention_layers[i]->forward(current_output, current_input, batch_size, seq_len, stream, /*mask=*/true);

        // Add & Norm
        add_tensors(current_output, residual, current_output, batch_size * seq_len * hidden_dim, stream);

        // Prepare residual for next sublayer
        cudaMemcpy(residual, current_output, batch_size * seq_len * hidden_dim * sizeof(float), cudaMemcpyDeviceToDevice);

        // Layer Norm 2
        layer_norm2_layers[i]->forward(current_output, current_output, seq_len, stream);

        // Encoder-Decoder Attention
        // Query comes from the previous sublayer's output, Key and Value come from the encoder output
        encoder_attention_layers[i]->forward(current_output, current_output, encoder_output, batch_size, seq_len, stream);

        // Add & Norm
        add_tensors(current_output, residual, current_output, batch_size * seq_len * hidden_dim, stream);

        // Prepare residual for next sublayer
        cudaMemcpy(residual, current_output, batch_size * seq_len * hidden_dim * sizeof(float), cudaMemcpyDeviceToDevice);

        // Layer Norm 3
        layer_norm3_layers[i]->forward(current_output, current_output, seq_len, stream);

        // Feed Forward
        feed_forward_layers[i]->forward(current_output, current_output, seq_len, stream);

        // Add & Norm
        add_tensors(current_output, residual, current_output, batch_size * seq_len * hidden_dim, stream);

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
