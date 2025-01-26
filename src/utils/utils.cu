#include <cuda_runtime.h>
#include <thrust/extrema.h>

#include "utils/utils.cuh"
#include "gpt2_weights.cuh"
#include "utils/debug.cuh"
#include "config.cuh"

// Kernel for element-wise addition
__global__ void add_tensors_kernel(const float *a, const float *b, float *c, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        c[idx] = a[idx] + b[idx];
    }
}

// Host function to call the kernel
void add_tensors(const float *a, const float *b, float *c, int size, cudaStream_t stream)
{
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    add_tensors_kernel<<<blocks, threads, 0, stream>>>(a, b, c, size);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

// Add validation kernel and functions
__global__ void validate_tensor_kernel(const float *tensor, size_t size, bool *has_invalid)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        float val = tensor[idx];
        if (isnan(val) || isinf(val))
        {
            *has_invalid = true;
        }
    }
}

bool validate_tensor_values(const float *tensor, size_t size, const char *tensor_name, float min_val, float max_val)
{
    bool *d_has_invalid;
    bool h_has_invalid = false;

    CUDA_CHECK(cudaMalloc(&d_has_invalid, sizeof(bool)));
    CUDA_CHECK(cudaMemset(d_has_invalid, 0, sizeof(bool)));

    int threads = 256;
    int blocks = (size + threads - 1) / threads;

    // Check for NaN and inf
    validate_tensor_kernel<<<blocks, threads>>>(tensor, size, d_has_invalid);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(&h_has_invalid, d_has_invalid, sizeof(bool), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_has_invalid));

    if (h_has_invalid)
    {
        debugPrint("Error: Found NaN or inf values in tensor %s\n", tensor_name);
        return false;
    }

    return true;
}

void validate_weights(const GPT2Weights *weights, const Config &config)
{
    if (!weights)
    {
        throw std::runtime_error("Null weights pointer passed to validate_weights");
    }

    // Validate embeddings
    validate_tensor_values(weights->getTokenEmbedding(),
                           weights->getDims().vocab_size * weights->getDims().hidden_dim,
                           "token_embedding");
    validate_tensor_values(weights->getPositionEmbedding(),
                           config.max_seq_len * weights->getDims().hidden_dim,
                           "position_embedding");

    // Validate final layer norm
    validate_tensor_values(weights->getFinalLayerNormWeight(),
                           weights->getDims().hidden_dim,
                           "final_layer_norm_weight");
    validate_tensor_values(weights->getFinalLayerNormBias(),
                           weights->getDims().hidden_dim,
                           "final_layer_norm_bias");

    // Validate each layer's weights
    for (int i = 0; i < weights->getNumLayers(); i++)
    {
        std::string layer_prefix = "layer_" + std::to_string(i) + "_";

        // Attention weights
        validate_tensor_values(weights->getAttentionQKVWeight(i),
                               3 * weights->getDims().hidden_dim * weights->getDims().hidden_dim,
                               (layer_prefix + "attn_qkv_weight").c_str());
        validate_tensor_values(weights->getAttentionQKVBias(i),
                               3 * weights->getDims().hidden_dim,
                               (layer_prefix + "attn_qkv_bias").c_str());
        validate_tensor_values(weights->getAttentionProjectionWeight(i),
                               weights->getDims().hidden_dim * weights->getDims().hidden_dim,
                               (layer_prefix + "attn_proj_weight").c_str());
        validate_tensor_values(weights->getAttentionProjectionBias(i),
                               weights->getDims().hidden_dim,
                               (layer_prefix + "attn_proj_bias").c_str());

        // Layer norms
        validate_tensor_values(weights->getAttentionLayerNormWeight(i),
                               weights->getDims().hidden_dim,
                               (layer_prefix + "attn_ln_weight").c_str());
        validate_tensor_values(weights->getAttentionLayerNormBias(i),
                               weights->getDims().hidden_dim,
                               (layer_prefix + "attn_ln_bias").c_str());
        validate_tensor_values(weights->getFFNLayerNormWeight(i),
                               weights->getDims().hidden_dim,
                               (layer_prefix + "ffn_ln_weight").c_str());
        validate_tensor_values(weights->getFFNLayerNormBias(i),
                               weights->getDims().hidden_dim,
                               (layer_prefix + "ffn_ln_bias").c_str());

        // FFN weights
        validate_tensor_values(weights->getFFNFC1Weight(i),
                               weights->getDims().intermediate_dim * weights->getDims().hidden_dim,
                               (layer_prefix + "ffn_fc1_weight").c_str());
        validate_tensor_values(weights->getFFNFC1Bias(i),
                               weights->getDims().intermediate_dim,
                               (layer_prefix + "ffn_fc1_bias").c_str());
        validate_tensor_values(weights->getFFNFC2Weight(i),
                               weights->getDims().hidden_dim * weights->getDims().intermediate_dim,
                               (layer_prefix + "ffn_fc2_weight").c_str());
        validate_tensor_values(weights->getFFNFC2Bias(i),
                               weights->getDims().hidden_dim,
                               (layer_prefix + "ffn_fc2_bias").c_str());
    }
}
