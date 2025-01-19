#pragma once

#include <vector>
#include <string>
#include <unordered_map>
#include "model_dimensions.cuh"
#include "utils/load_weights.cuh"
#include "cuda_runtime.h"

struct LayerWeights
{
    // Attention weights
    float *attn_qkv_weight;  // Combined QKV weights [3 * hidden_dim, hidden_dim]
    float *attn_qkv_bias;    // Combined QKV bias [3 * hidden_dim]
    float *attn_proj_weight; // Output projection [hidden_dim, hidden_dim]
    float *attn_proj_bias;   // Output projection bias [hidden_dim]

    // Layer norm weights
    float *attn_ln_weight; // Attention LayerNorm weight [hidden_dim]
    float *attn_ln_bias;   // Attention LayerNorm bias [hidden_dim]
    float *ffn_ln_weight;  // FFN LayerNorm weight [hidden_dim]
    float *ffn_ln_bias;    // FFN LayerNorm bias [hidden_dim]

    // FFN weights
    float *ffn_fc1_weight; // First FFN layer [intermediate_dim, hidden_dim]
    float *ffn_fc1_bias;   // First FFN bias [intermediate_dim]
    float *ffn_fc2_weight; // Second FFN layer [hidden_dim, intermediate_dim]
    float *ffn_fc2_bias;   // Second FFN bias [hidden_dim]
};

class GPT2Weights
{
public:
    GPT2Weights(ModelDimensions &dims,
                const std::vector<std::pair<std::string, TensorInfo>> &tensor_infos);
    ~GPT2Weights();

    // Load weights from SafeTensors data
    bool loadWeights(const std::vector<std::pair<std::string, TensorInfo>> &tensor_infos,
                     const std::vector<uint8_t> &data);

    // Getters for various weights
    float *getTokenEmbedding() const { return token_embedding; }
    float *getPositionEmbedding() const { return position_embedding; }
    float *getFinalLayerNormWeight() const { return final_ln_weight; }
    float *getFinalLayerNormBias() const { return final_ln_bias; }
    const LayerWeights &getLayerWeights(int layer) const { return layers[layer]; }

private:
    ModelDimensions dims;
    std::vector<LayerWeights> layers;

    // Embeddings
    float *token_embedding;    // [vocab_size, hidden_dim]
    float *position_embedding; // [max_seq_len, hidden_dim]

    // Final layer norm
    float *final_ln_weight; // [hidden_dim]
    float *final_ln_bias;   // [hidden_dim]

    // Helper functions
    void allocateWeights();
    void freeWeights();
    bool copyWeightToDevice(const std::vector<uint8_t> &data,
                            size_t offset,
                            size_t size,
                            float *dest);
    bool loadTensor(const std::string &name,
                    const TensorInfo &info,
                    const std::vector<uint8_t> &data);
};