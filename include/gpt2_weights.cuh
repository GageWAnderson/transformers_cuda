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

    // MLP weights (aliases for FFN weights)
    float *mlp_fc_weight;   // Same as ffn_fc1_weight
    float *mlp_fc_bias;     // Same as ffn_fc1_bias
    float *mlp_proj_weight; // Same as ffn_fc2_weight
    float *mlp_proj_bias;   // Same as ffn_fc2_bias
};

class GPT2Weights
{
public:
    GPT2Weights(ModelDimensions &dims,
                const std::vector<std::pair<std::string, TensorInfo>> &tensor_infos,
                const std::vector<uint8_t> &data);
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

    // Layer weight getters
    float *getAttentionQKVWeight(int layer) const { return layers[layer].attn_qkv_weight; }
    float *getAttentionQKVBias(int layer) const { return layers[layer].attn_qkv_bias; }
    float *getAttentionProjectionWeight(int layer) const { return layers[layer].attn_proj_weight; }
    float *getAttentionProjectionBias(int layer) const { return layers[layer].attn_proj_bias; }

    float *getAttentionLayerNormWeight(int layer) const { return layers[layer].attn_ln_weight; }
    float *getAttentionLayerNormBias(int layer) const { return layers[layer].attn_ln_bias; }
    float *getFFNLayerNormWeight(int layer) const { return layers[layer].ffn_ln_weight; }
    float *getFFNLayerNormBias(int layer) const { return layers[layer].ffn_ln_bias; }

    float *getFFNFC1Weight(int layer) const { return layers[layer].ffn_fc1_weight; }
    float *getFFNFC1Bias(int layer) const { return layers[layer].ffn_fc1_bias; }
    float *getFFNFC2Weight(int layer) const { return layers[layer].ffn_fc2_weight; }
    float *getFFNFC2Bias(int layer) const { return layers[layer].ffn_fc2_bias; }

    // Layer weight setters
    void setAttentionQKVWeight(int layer, float *weight) { layers[layer].attn_qkv_weight = weight; }
    void setAttentionQKVBias(int layer, float *bias) { layers[layer].attn_qkv_bias = bias; }
    void setAttentionProjectionWeight(int layer, float *weight) { layers[layer].attn_proj_weight = weight; }
    void setAttentionProjectionBias(int layer, float *bias) { layers[layer].attn_proj_bias = bias; }

    void setAttentionLayerNormWeight(int layer, float *weight) { layers[layer].attn_ln_weight = weight; }
    void setAttentionLayerNormBias(int layer, float *bias) { layers[layer].attn_ln_bias = bias; }
    void setFFNLayerNormWeight(int layer, float *weight) { layers[layer].ffn_ln_weight = weight; }
    void setFFNLayerNormBias(int layer, float *bias) { layers[layer].ffn_ln_bias = bias; }

    void setFFNFC1Weight(int layer, float *weight) { layers[layer].ffn_fc1_weight = weight; }
    void setFFNFC1Bias(int layer, float *bias) { layers[layer].ffn_fc1_bias = bias; }
    void setFFNFC2Weight(int layer, float *weight) { layers[layer].ffn_fc2_weight = weight; }
    void setFFNFC2Bias(int layer, float *bias) { layers[layer].ffn_fc2_bias = bias; }

    // Embedding setters
    void setTokenEmbedding(float *embedding) { token_embedding = embedding; }
    void setPositionEmbedding(float *embedding) { position_embedding = embedding; }

    // Final layer norm setters
    void setFinalLayerNormWeight(float *weight) { final_ln_weight = weight; }
    void setFinalLayerNormBias(float *bias) { final_ln_bias = bias; }

    // Get number of layers
    int getNumLayers() const { return layers.size(); }

    // Get model dimensions
    const ModelDimensions &getDims() const { return dims; }

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
                            float *dest,
                            Dtype src_dtype);
    bool loadTensor(const std::string &name,
                    const TensorInfo &info,
                    const std::vector<uint8_t> &data);

    const std::vector<std::pair<std::string, TensorInfo>> tensor_infos;
};