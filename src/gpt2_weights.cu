#include "gpt2_weights.cuh"
#include "utils/debug.cuh"
#include "utils/load_weights.cuh"
#include <cstring>

int countLayers(const std::vector<std::pair<std::string, TensorInfo>> &tensor_infos)
{
    int max_layer = -1;
    for (const auto &[name, _] : tensor_infos)
    {
        // Look for patterns like "h.5." or "transformer.h.5."
        size_t pos = name.find("h.");
        if (pos == std::string::npos)
            continue;

        pos += 2; // Skip "h."
        size_t end = name.find(".", pos);
        if (end == std::string::npos)
            continue;

        try
        {
            int layer_num = std::stoi(name.substr(pos, end - pos));
            max_layer = std::max(max_layer, layer_num);
        }
        catch (...)
        {
            continue;
        }
    }
    return max_layer + 1; // Convert from 0-based to count
}

GPT2Weights::GPT2Weights(ModelDimensions &dims,
                         const std::vector<std::pair<std::string, TensorInfo>> &tensor_infos)
{
    // Count layers before allocation
    dims.num_layers = countLayers(tensor_infos);
    debugPrint("Detected %d layers in weights\n", dims.num_layers);

    // Set other dimensions based on tensor shapes
    for (const auto &[name, info] : tensor_infos) {
        if (name == "wte.weight" || name.find("transformer.wte.weight") != std::string::npos) {
            // Token embedding shape is [vocab_size, hidden_dim]
            dims.vocab_size = info.shape[0];
            dims.hidden_dim = info.shape[1];
            dims.embedding_dim = info.shape[1];  // Same as hidden_dim for GPT-2
        }
        else if (name == "h.0.attn.c_attn.weight" || name.find("transformer.h.0.attn.c_attn.weight") != std::string::npos) {
            // QKV weight shape is [hidden_dim, 3 * hidden_dim]
            // Number of attention heads can be inferred from hidden_dim (hidden_dim must be divisible by num_heads)
            dims.num_heads = dims.hidden_dim / 64;  // GPT-2 uses 64 as head dimension
        }
        else if (name == "h.0.mlp.c_fc.weight" || name.find("transformer.h.0.mlp.c_fc.weight") != std::string::npos) {
            // First FFN layer shape is [hidden_dim, intermediate_dim]
            dims.intermediate_dim = info.shape[1];
        }
    }

    debugPrint("Model dimensions from weights:\n");
    debugPrint("  Vocabulary size: %d\n", dims.vocab_size);
    debugPrint("  Hidden dimension: %d\n", dims.hidden_dim);
    debugPrint("  Number of heads: %d\n", dims.num_heads);
    debugPrint("  Intermediate dimension: %d\n", dims.intermediate_dim);

    dims.valid = dims.vocab_size > 0 && dims.hidden_dim > 0 && 
                dims.num_heads > 0 && dims.intermediate_dim > 0;

    if (!dims.valid) {
        debugPrint("Warning: Failed to determine all model dimensions from weights\n");
    }

    allocateWeights();
}

GPT2Weights::~GPT2Weights()
{
    freeWeights();
}

void GPT2Weights::allocateWeights()
{
    // Allocate embeddings
    cudaMalloc(&token_embedding, dims.vocab_size * dims.hidden_dim * sizeof(float));
    cudaMalloc(&position_embedding, 1024 * dims.hidden_dim * sizeof(float)); // GPT-2 uses 1024 positions

    // Allocate final layer norm
    cudaMalloc(&final_ln_weight, dims.hidden_dim * sizeof(float));
    cudaMalloc(&final_ln_bias, dims.hidden_dim * sizeof(float));

    // Allocate layer weights
    layers.resize(dims.num_layers);
    for (int i = 0; i < dims.num_layers; i++)
    {
        LayerWeights &layer = layers[i];

        // Attention weights
        cudaMalloc(&layer.attn_qkv_weight, 3 * dims.hidden_dim * dims.hidden_dim * sizeof(float));
        cudaMalloc(&layer.attn_qkv_bias, 3 * dims.hidden_dim * sizeof(float));
        cudaMalloc(&layer.attn_proj_weight, dims.hidden_dim * dims.hidden_dim * sizeof(float));
        cudaMalloc(&layer.attn_proj_bias, dims.hidden_dim * sizeof(float));

        // Layer norms
        cudaMalloc(&layer.attn_ln_weight, dims.hidden_dim * sizeof(float));
        cudaMalloc(&layer.attn_ln_bias, dims.hidden_dim * sizeof(float));
        cudaMalloc(&layer.ffn_ln_weight, dims.hidden_dim * sizeof(float));
        cudaMalloc(&layer.ffn_ln_bias, dims.hidden_dim * sizeof(float));

        // FFN weights
        cudaMalloc(&layer.ffn_fc1_weight, dims.intermediate_dim * dims.hidden_dim * sizeof(float));
        cudaMalloc(&layer.ffn_fc1_bias, dims.intermediate_dim * sizeof(float));
        cudaMalloc(&layer.ffn_fc2_weight, dims.hidden_dim * dims.intermediate_dim * sizeof(float));
        cudaMalloc(&layer.ffn_fc2_bias, dims.hidden_dim * sizeof(float));
    }
}

void GPT2Weights::freeWeights()
{
    // Free embeddings
    cudaFree(token_embedding);
    cudaFree(position_embedding);

    // Free final layer norm
    cudaFree(final_ln_weight);
    cudaFree(final_ln_bias);

    // Free layer weights
    for (auto &layer : layers)
    {
        cudaFree(layer.attn_qkv_weight);
        cudaFree(layer.attn_qkv_bias);
        cudaFree(layer.attn_proj_weight);
        cudaFree(layer.attn_proj_bias);
        cudaFree(layer.attn_ln_weight);
        cudaFree(layer.attn_ln_bias);
        cudaFree(layer.ffn_ln_weight);
        cudaFree(layer.ffn_ln_bias);
        cudaFree(layer.ffn_fc1_weight);
        cudaFree(layer.ffn_fc1_bias);
        cudaFree(layer.ffn_fc2_weight);
        cudaFree(layer.ffn_fc2_bias);
    }
}

bool GPT2Weights::copyWeightToDevice(const std::vector<uint8_t> &data,
                                     size_t offset,
                                     size_t size,
                                     float *dest)
{
    cudaError_t error = cudaMemcpy(dest,
                                   data.data() + offset,
                                   size,
                                   cudaMemcpyHostToDevice);
    return error == cudaSuccess;
}

bool GPT2Weights::loadTensor(const std::string &name,
                             const TensorInfo &info,
                             const std::vector<uint8_t> &data)
{
    size_t offset = info.data_offsets.first;
    size_t size = info.data_offsets.second - info.data_offsets.first;

    // Helper lambda to extract layer number from tensor name
    auto getLayerNum = [](const std::string &name) -> int
    {
        // First check for "h." prefix directly
        size_t pos = name.find("h.");
        if (pos == std::string::npos)
        {
            // If not found, check for "transformer.h." prefix
            pos = name.find("transformer.h.");
            if (pos == std::string::npos)
                return -1;
            pos += 14; // length of "transformer.h."
        }
        else
        {
            pos += 2; // length of "h."
        }

        // Find the next dot after the layer number
        size_t end = name.find(".", pos);
        if (end == std::string::npos)
            return -1;

        // Extract and convert the layer number
        return std::stoi(name.substr(pos, end - pos));
    };

    // Handle embeddings and final layer norm
    if (name.find("wte.weight") != std::string::npos || name.find("transformer.wte.weight") != std::string::npos)
    {
        return copyWeightToDevice(data, offset, size, token_embedding);
    }
    else if (name.find("wpe.weight") != std::string::npos || name.find("transformer.wpe.weight") != std::string::npos)
    {
        return copyWeightToDevice(data, offset, size, position_embedding);
    }
    else if (name.find("ln_f.weight") != std::string::npos)
    {
        return copyWeightToDevice(data, offset, size, final_ln_weight);
    }
    else if (name.find("ln_f.bias") != std::string::npos)
    {
        return copyWeightToDevice(data, offset, size, final_ln_bias);
    }

    // Handle layer-specific weights
    int layer_num = getLayerNum(name);
    debugPrint("Loading tensor: %s, layer number: %d, dims.num_layers: %d\n", name.c_str(), layer_num, dims.num_layers);
    if (layer_num >= 0 && layer_num < dims.num_layers)
    {
        LayerWeights &layer = layers[layer_num];

        // Extract the part after the layer number for easier matching
        size_t layer_prefix = name.find("h." + std::to_string(layer_num) + ".");
        if (layer_prefix == std::string::npos)
        {
            layer_prefix = name.find("transformer.h." + std::to_string(layer_num) + ".");
        }
        if (layer_prefix == std::string::npos)
        {
            debugPrint("Warning: Unexpected tensor name format %s\n", name.c_str());
            return false;
        }

        std::string weight_name = name.substr(layer_prefix + 2 + std::to_string(layer_num).length() + 1);
        debugPrint("Loading tensor: %s, weight name: %s\n", name.c_str(), weight_name.c_str());

        // Attention weights
        if (weight_name == "attn.c_attn.weight")
        {
            return copyWeightToDevice(data, offset, size, layer.attn_qkv_weight);
        }
        else if (weight_name == "attn.c_attn.bias")
        {
            return copyWeightToDevice(data, offset, size, layer.attn_qkv_bias);
        }
        else if (weight_name == "attn.c_proj.weight")
        {
            return copyWeightToDevice(data, offset, size, layer.attn_proj_weight);
        }
        else if (weight_name == "attn.c_proj.bias")
        {
            return copyWeightToDevice(data, offset, size, layer.attn_proj_bias);
        }

        // Layer norms
        else if (weight_name == "ln_1.weight")
        {
            return copyWeightToDevice(data, offset, size, layer.attn_ln_weight);
        }
        else if (weight_name == "ln_1.bias")
        {
            return copyWeightToDevice(data, offset, size, layer.attn_ln_bias);
        }
        else if (weight_name == "ln_2.weight")
        {
            return copyWeightToDevice(data, offset, size, layer.ffn_ln_weight);
        }
        else if (weight_name == "ln_2.bias")
        {
            return copyWeightToDevice(data, offset, size, layer.ffn_ln_bias);
        }

        // MLP/FFN weights
        else if (weight_name == "mlp.c_fc.weight")
        {
            return copyWeightToDevice(data, offset, size, layer.ffn_fc1_weight);
        }
        else if (weight_name == "mlp.c_fc.bias")
        {
            return copyWeightToDevice(data, offset, size, layer.ffn_fc1_bias);
        }
        else if (weight_name == "mlp.c_proj.weight")
        {
            return copyWeightToDevice(data, offset, size, layer.ffn_fc2_weight);
        }
        else if (weight_name == "mlp.c_proj.bias")
        {
            return copyWeightToDevice(data, offset, size, layer.ffn_fc2_bias);
        }
    }

    debugPrint("Warning: Unhandled tensor %s\n", name.c_str());
    return false;
}

bool GPT2Weights::loadWeights(const std::vector<std::pair<std::string, TensorInfo>> &tensor_infos,
                              const std::vector<uint8_t> &data)
{
    bool success = true;
    for (const auto &[name, info] : tensor_infos)
    {
        success &= loadTensor(name, info, data);
    }
    return success;
}