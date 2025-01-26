#include "gpt2_weights.cuh"
#include "utils/debug.cuh"
#include "utils/load_weights.cuh"
#include "utils/utils.cuh"
#include <cstring>
#include <exception>
#include <cuda_fp16.h>

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
    debugPrint("countLayers found max_layer = %d, returning count = %d\n", max_layer, max_layer + 1);
    return max_layer + 1; // Convert from 0-based to count
}

GPT2Weights::GPT2Weights(ModelDimensions &dims,
                         const std::vector<std::pair<std::string, TensorInfo>> &tensor_infos,
                         const std::vector<uint8_t> &data)
    : tensor_infos(tensor_infos), // Store tensor_infos as member variable
      dims(dims)                  // Store dims as member variable to prevent reference issues
{
    // Add CUDA device initialization check
    int deviceCount;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0)
    {
        throw std::runtime_error("No CUDA devices available");
    }

    // Select first available device
    CUDA_CHECK(cudaSetDevice(0));

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    debugPrint("Using CUDA Device %s with %zu MB of memory\n",
               prop.name,
               prop.totalGlobalMem / (1024 * 1024));

    try
    {
        // Count layers before allocation
        this->dims.num_layers = countLayers(tensor_infos);

        // Set other dimensions based on tensor shapes
        int max_head_number = -1;
        for (const auto &[name, info] : tensor_infos)
        {
            if (name == "wte.weight" || name.find("transformer.wte.weight") != std::string::npos)
            {
                // Token embedding shape is [vocab_size, hidden_dim]
                dims.vocab_size = info.shape[0]; // TODO: Vocab size should be a power of 2
                dims.hidden_dim = info.shape[1];
                dims.embedding_dim = info.shape[1];
                dims.intermediate_dim = dims.hidden_dim * 4; // Intermediate dim is 4x hidden dim for GPT-2
            }
            else if (name.find("h.") != std::string::npos)
            {
                // Extract the layer number after "h."
                size_t pos = name.find("h.") + 2;
                size_t end = name.find(".", pos);
                if (end != std::string::npos)
                {
                    try
                    {
                        int head_number = std::stoi(name.substr(pos, end - pos));
                        max_head_number = std::max(max_head_number, head_number);
                    }
                    catch (...)
                    {
                        continue;
                    }
                }
            }
        }

        // Set the number of heads based on the highest head number found
        dims.num_heads = max_head_number + 1; // Convert from 0-based to count

        debugPrint("Model dimensions from weights:\n");
        debugPrint("  Vocabulary size: %d\n", dims.vocab_size);
        debugPrint("  Hidden dimension: %d\n", dims.hidden_dim);
        debugPrint("  Number of heads: %d\n", dims.num_heads);
        debugPrint("  Intermediate dimension: %d\n", dims.intermediate_dim);
        debugPrint("  Number of layers: %d\n", this->dims.num_layers);
        dims.valid = dims.vocab_size > 0 && dims.hidden_dim > 0 &&
                     dims.num_heads > 0 && dims.intermediate_dim > 0;

        if (!dims.valid)
        {
            debugPrint("Warning: Failed to determine all model dimensions from weights\n");
        }
        debugPrint("Allocating weights for %d layers\n", this->dims.num_layers);
        allocateWeights();

        if (!loadWeights(tensor_infos, data))
        {
            throw std::runtime_error("Failed to load weights");
        }
    }
    catch (const std::exception &e)
    {
        debugPrint("Error: %s\n", e.what());
        freeWeights(); // Free allocated memory before terminating
        throw;
    }
}

GPT2Weights::~GPT2Weights()
{
    freeWeights();
}

void GPT2Weights::allocateWeights()
{
    try
    {
        debugPrint("Allocating weights for %d layers\n", this->dims.num_layers);
        // Helper lambda to find tensor info by name
        auto findTensorInfo = [](const std::string &name,
                                 const std::vector<std::pair<std::string, TensorInfo>> &tensor_infos)
            -> const TensorInfo *
        {
            for (const auto &[tensor_name, info] : tensor_infos)
            {
                if (tensor_name.find(name) != std::string::npos)
                {
                    return &info;
                }
            }
            return nullptr;
        };

        // Track allocations to free them if an error occurs
        std::vector<void *> allocated_ptrs;
        auto safeCudaMalloc = [&allocated_ptrs](void **ptr, size_t size)
        {
            CUDA_CHECK(cudaMalloc(ptr, size));
            allocated_ptrs.push_back(*ptr);
        };

        // Allocate embeddings
        const TensorInfo *wte_info = findTensorInfo("wte.weight", tensor_infos);
        if (!wte_info)
        {
            throw std::runtime_error("Could not find token embedding weights");
        }
        size_t wte_size = wte_info->shape[0] * wte_info->shape[1] * sizeof(float);
        debugPrint("Allocating token embedding: %zu bytes\n", wte_size);
        safeCudaMalloc((void **)&token_embedding, wte_size);
        if (token_embedding == nullptr)
        {
            throw std::runtime_error("Failed to allocate token embedding");
        }

        const TensorInfo *wpe_info = findTensorInfo("wpe.weight", tensor_infos);
        if (!wpe_info)
        {
            throw std::runtime_error("Could not find position embedding weights");
        }
        size_t wpe_size = wpe_info->shape[0] * wpe_info->shape[1] * sizeof(float);
        debugPrint("Allocating position embedding: %zu bytes\n", wpe_size);
        safeCudaMalloc((void **)&position_embedding, wpe_size);

        // Allocate final layer norm
        const TensorInfo *ln_f_weight = findTensorInfo("ln_f.weight", tensor_infos);
        if (!ln_f_weight)
        {
            throw std::runtime_error("Could not find final layer norm weights");
        }
        size_t ln_size = ln_f_weight->shape[0] * sizeof(float);
        debugPrint("Allocating final layer norm: %zu bytes\n", ln_size);
        safeCudaMalloc((void **)&final_ln_weight, ln_size);
        safeCudaMalloc((void **)&final_ln_bias, ln_size);

        // Allocate layer weights
        try
        {
            layers.resize(this->dims.num_layers);
        }
        catch (const std::bad_alloc &e)
        {
            throw std::runtime_error("Failed to allocate memory for layer weights: " + std::string(e.what()));
        }
        for (int i = 0; i < this->dims.num_layers; i++)
        {
            LayerWeights &layer = layers[i];
            std::string layer_prefix = "h." + std::to_string(i) + ".";

            // Find both weight and bias tensors for QKV
            const TensorInfo *qkv_weight = findTensorInfo(layer_prefix + "attn.c_attn.weight", tensor_infos);
            const TensorInfo *qkv_bias = findTensorInfo(layer_prefix + "attn.c_attn.bias", tensor_infos);
            if (!qkv_weight || !qkv_bias)
            {
                throw std::runtime_error("Could not find QKV weights/bias for layer " + std::to_string(i));
            }

            size_t qkv_weight_size = qkv_weight->shape[0] * qkv_weight->shape[1] * sizeof(float);
            size_t qkv_bias_size = qkv_bias->shape[0] * sizeof(float); // Use bias tensor shape

            debugPrint("Layer %d QKV weight size: %zu bytes\n", i, qkv_weight_size);
            safeCudaMalloc((void **)&layer.attn_qkv_weight, qkv_weight_size);
            safeCudaMalloc((void **)&layer.attn_qkv_bias, qkv_bias_size);

            const TensorInfo *proj_weight = findTensorInfo(layer_prefix + "attn.c_proj.weight", tensor_infos);
            if (!proj_weight)
            {
                throw std::runtime_error("Could not find projection weights for layer " + std::to_string(i));
            }
            size_t proj_weight_size = proj_weight->shape[0] * proj_weight->shape[1] * sizeof(float);
            size_t proj_bias_size = proj_weight->shape[0] * sizeof(float);
            safeCudaMalloc((void **)&layer.attn_proj_weight, proj_weight_size);
            safeCudaMalloc((void **)&layer.attn_proj_bias, proj_bias_size);

            // Layer norms
            safeCudaMalloc((void **)&layer.attn_ln_weight, ln_size);
            safeCudaMalloc((void **)&layer.attn_ln_bias, ln_size);
            safeCudaMalloc((void **)&layer.ffn_ln_weight, ln_size);
            safeCudaMalloc((void **)&layer.ffn_ln_bias, ln_size);

            // Find MLP weights and biases
            const TensorInfo *mlp_fc_weight = findTensorInfo(layer_prefix + "mlp.c_fc.weight", tensor_infos);
            const TensorInfo *mlp_fc_bias = findTensorInfo(layer_prefix + "mlp.c_fc.bias", tensor_infos);
            const TensorInfo *mlp_proj_weight = findTensorInfo(layer_prefix + "mlp.c_proj.weight", tensor_infos);
            const TensorInfo *mlp_proj_bias = findTensorInfo(layer_prefix + "mlp.c_proj.bias", tensor_infos);

            if (!mlp_fc_weight || !mlp_fc_bias || !mlp_proj_weight || !mlp_proj_bias)
            {
                throw std::runtime_error("Could not find MLP weights/biases for layer " + std::to_string(i));
            }

            // Calculate sizes using the actual tensor shapes
            size_t mlp_fc_weight_size = mlp_fc_weight->shape[0] * mlp_fc_weight->shape[1] * sizeof(float);
            size_t mlp_fc_bias_size = mlp_fc_bias->shape[0] * sizeof(float);
            size_t mlp_proj_weight_size = mlp_proj_weight->shape[0] * mlp_proj_weight->shape[1] * sizeof(float);
            size_t mlp_proj_bias_size = mlp_proj_bias->shape[0] * sizeof(float);

            // Allocate MLP weights and biases using correct sizes
            layer.mlp_fc_weight = layer.ffn_fc1_weight;
            layer.mlp_fc_bias = layer.ffn_fc1_bias;
            layer.mlp_proj_weight = layer.ffn_fc2_weight;
            layer.mlp_proj_bias = layer.ffn_fc2_bias;

            CUDA_CHECK(cudaMalloc(&layer.ffn_fc1_weight, mlp_fc_weight_size));
            CUDA_CHECK(cudaMalloc(&layer.ffn_fc1_bias, mlp_fc_bias_size));
            CUDA_CHECK(cudaMalloc(&layer.ffn_fc2_weight, mlp_proj_weight_size));
            CUDA_CHECK(cudaMalloc(&layer.ffn_fc2_bias, mlp_proj_bias_size));
        }
    }
    catch (const std::exception &e)
    {
        debugPrint("Error during weight allocation: %s\n", e.what());
        freeWeights(); // Clean up any weights that were successfully allocated
        throw;
    }
}

void GPT2Weights::freeWeights()
{
    // Free embeddings
    CUDA_CHECK(cudaFree(token_embedding));
    CUDA_CHECK(cudaFree(position_embedding));

    // Free final layer norm
    CUDA_CHECK(cudaFree(final_ln_weight));
    CUDA_CHECK(cudaFree(final_ln_bias));

    // Free layer weights
    for (auto &layer : layers)
    {
        CUDA_CHECK(cudaFree(layer.attn_qkv_weight));
        CUDA_CHECK(cudaFree(layer.attn_qkv_bias));
        CUDA_CHECK(cudaFree(layer.attn_proj_weight));
        CUDA_CHECK(cudaFree(layer.attn_proj_bias));
        CUDA_CHECK(cudaFree(layer.attn_ln_weight));
        CUDA_CHECK(cudaFree(layer.attn_ln_bias));
        CUDA_CHECK(cudaFree(layer.ffn_ln_weight));
        CUDA_CHECK(cudaFree(layer.ffn_ln_bias));
        CUDA_CHECK(cudaFree(layer.ffn_fc1_weight));
        CUDA_CHECK(cudaFree(layer.ffn_fc1_bias));
        CUDA_CHECK(cudaFree(layer.ffn_fc2_weight));
        CUDA_CHECK(cudaFree(layer.ffn_fc2_bias));
    }
}

bool GPT2Weights::copyWeightToDevice(const std::vector<uint8_t> &data,
                                     size_t offset,
                                     size_t size,
                                     float *dest,
                                     Dtype src_dtype)
{
    debugPrint("copyWeightToDevice called with offset = %zu, size = %zu, dtype = %d\n",
               offset, size, static_cast<int>(src_dtype));

    const void *src_ptr = data.data() + offset - 1;

    switch (src_dtype)
    {
    case Dtype::F32:
    {
        debugPrint("Copying F32 data to device\n");

        // Create temporary host buffer
        std::vector<float> host_buffer(size / sizeof(float));
        std::memcpy(host_buffer.data(), src_ptr, size);

        // Copy weights to device
        CUDA_CHECK(cudaMemcpy(dest, host_buffer.data(), size, cudaMemcpyHostToDevice));
        break;
    }
    default:
        debugPrint("Unsupported dtype in copyWeightToDevice\n");
        return false;
    }

    debugPrint("Successfully copied data to device\n");
    return true;
}

bool GPT2Weights::loadTensor(const std::string &name,
                             const TensorInfo &info,
                             const std::vector<uint8_t> &data)
{
    size_t offset = info.data_offsets.first;
    size_t size = info.data_offsets.second - info.data_offsets.first;

    // Add debug print for tensor loading
    debugPrint("Loading tensor %s: offset=%zu, size=%zu, dtype=%d\n",
               name.c_str(), offset, size, static_cast<int>(info.dtype));

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
        return copyWeightToDevice(data, offset, size, token_embedding, info.dtype);
    }
    else if (name.find("wpe.weight") != std::string::npos || name.find("transformer.wpe.weight") != std::string::npos)
    {
        return copyWeightToDevice(data, offset, size, position_embedding, info.dtype);
    }
    else if (name.find("ln_f.weight") != std::string::npos)
    {
        return copyWeightToDevice(data, offset, size, final_ln_weight, info.dtype);
    }
    else if (name.find("ln_f.bias") != std::string::npos)
    {
        return copyWeightToDevice(data, offset, size, final_ln_bias, info.dtype);
    }

    // Handle layer-specific weights
    int layer_num = getLayerNum(name);
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
        debugPrint("Loading tensor: %s, weight name: %s, offset: %zu, size: %zu, data type: %d\n", name.c_str(), weight_name.c_str(), offset, size, static_cast<int>(info.dtype));

        // Attention weights
        if (weight_name == "attn.c_attn.weight")
        {
            return copyWeightToDevice(data, offset, size, layer.attn_qkv_weight, info.dtype);
        }
        else if (weight_name == "attn.c_attn.bias")
        {
            return copyWeightToDevice(data, offset, size, layer.attn_qkv_bias, info.dtype);
        }
        else if (weight_name == "attn.c_proj.weight")
        {
            return copyWeightToDevice(data, offset, size, layer.attn_proj_weight, info.dtype);
        }
        else if (weight_name == "attn.c_proj.bias")
        {
            return copyWeightToDevice(data, offset, size, layer.attn_proj_bias, info.dtype);
        }
        // Add handling for attn.bias
        else if (weight_name == "attn.bias")
        {
            // This tensor is typically used as an attention mask and doesn't need to be loaded
            // as it's usually all zeros except for the upper triangle being set to -inf
            return true;
        }

        // Layer norms
        else if (weight_name == "ln_1.weight")
        {
            return copyWeightToDevice(data, offset, size, layer.attn_ln_weight, info.dtype);
        }
        else if (weight_name == "ln_1.bias")
        {
            return copyWeightToDevice(data, offset, size, layer.attn_ln_bias, info.dtype);
        }
        else if (weight_name == "ln_2.weight")
        {
            return copyWeightToDevice(data, offset, size, layer.ffn_ln_weight, info.dtype);
        }
        else if (weight_name == "ln_2.bias")
        {
            return copyWeightToDevice(data, offset, size, layer.ffn_ln_bias, info.dtype);
        }

        // MLP/FFN weights
        else if (weight_name == "mlp.c_fc.weight")
        {
            layer.mlp_fc_weight = layer.ffn_fc1_weight;
            return copyWeightToDevice(data, offset, size, layer.ffn_fc1_weight, info.dtype);
        }
        else if (weight_name == "mlp.c_fc.bias")
        {
            layer.mlp_fc_bias = layer.ffn_fc1_bias;
            return copyWeightToDevice(data, offset, size, layer.ffn_fc1_bias, info.dtype);
        }
        else if (weight_name == "mlp.c_proj.weight")
        {
            layer.mlp_proj_weight = layer.ffn_fc2_weight;
            return copyWeightToDevice(data, offset, size, layer.ffn_fc2_weight, info.dtype);
        }
        else if (weight_name == "mlp.c_proj.bias")
        {
            layer.mlp_proj_bias = layer.ffn_fc2_bias;
            return copyWeightToDevice(data, offset, size, layer.ffn_fc2_bias, info.dtype);
        }
    }

    debugPrint("Warning: Unhandled tensor %s\n", name.c_str());
    return false;
}

bool GPT2Weights::loadWeights(const std::vector<std::pair<std::string, TensorInfo>> &tensor_infos,
                              const std::vector<uint8_t> &data)
{
    bool success = true;
    size_t total_size = 0;

    // Get header size (first 8 bytes contain header length)
    uint64_t header_length = *reinterpret_cast<const uint64_t *>(data.data());
    size_t header_total_size = 8 + header_length; // 8 bytes for length + header content

    for (const auto &[name, info] : tensor_infos)
    {
        size_t size = info.data_offsets.second - info.data_offsets.first;
        total_size += size;
        if (!loadTensor(name, info, data))
        {
            debugPrint("Failed to load tensor: %s\n", name.c_str());
            success = false;
        }
    }

    // Check if the total size matches the data buffer size minus header
    if (total_size != (data.size() - header_total_size))
    {
        debugPrint("Error: Total size of all tensors %zu does not match data buffer size minus header %zu\n",
                   total_size, (data.size() - header_total_size));
        return false;
    }

    return success;
}