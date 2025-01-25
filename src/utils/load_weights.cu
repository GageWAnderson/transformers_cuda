#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <unordered_map>
#include <stdexcept>
#include <sstream>
#include <nlohmann/json.hpp>
#include "utils/debug.cuh"
#include "model_dimensions.cuh"
#include "gpt2_weights.cuh"
#include "config.cuh"
#include "utils/load_weights.cuh"
#include "utils/utils.cuh"

using json = nlohmann::json;

size_t dtype_size(Dtype dtype)
{
    switch (dtype)
    {
    case Dtype::BOOL:
        return 1;
    case Dtype::U8:
        return 1;
    case Dtype::I8:
        return 1;
    case Dtype::F8_E5M2:
        return 1;
    case Dtype::F8_E4M3:
        return 1;
    case Dtype::I16:
        return 2;
    case Dtype::U16:
        return 2;
    case Dtype::F16:
        return 2;
    case Dtype::BF16:
        return 2;
    case Dtype::I32:
        return 4;
    case Dtype::U32:
        return 4;
    case Dtype::F32:
        return 4;
    case Dtype::F64:
        return 8;
    case Dtype::I64:
        return 8;
    case Dtype::U64:
        return 8;
    default:
        throw std::invalid_argument("Unknown dtype");
    }
}

// Metadata implementation
Metadata::Metadata(std::optional<std::unordered_map<std::string, std::string>> metadata_param,
                   std::vector<std::pair<std::string, TensorInfo>> tensors)
{
    this->metadata = metadata_param;
    for (size_t i = 0; i < tensors.size(); ++i)
    {
        this->index_map[tensors[i].first] = i;
        this->tensors.push_back(tensors[i].second);
    }
    validate();
}

Metadata::Metadata() : metadata(std::nullopt) {}

void Metadata::validate()
{
    for (size_t i = 0; i < tensors.size(); ++i)
    {
        const auto &info = tensors[i];
        size_t s = info.data_offsets.first;
        size_t e = info.data_offsets.second;
        if (e < s)
        {
            std::ostringstream oss;
            oss << "Invalid offset for tensor at index " << i << ": start=" << s << ", end=" << e;
            throw SafeTensorError(oss.str());
        }
        size_t nelements = 1;
        for (size_t dim : info.shape)
        {
            nelements *= dim;
        }
        size_t nbytes = nelements * dtype_size(info.dtype);
        if ((e - s) != nbytes)
        {
            std::ostringstream oss;
            oss << "Tensor invalid info at index " << i << ": start=" << s << ", end=" << e << ", expected bytes=" << nbytes << ", actual bytes=" << (e - s);
            throw SafeTensorError(oss.str());
        }
    }
}

// TensorView implementation
TensorView::TensorView(Dtype dtype, std::vector<size_t> shape, const std::vector<uint8_t> &data)
    : dtype_value(dtype), shape_value(shape), data_value(data)
{
    size_t n = data.size();
    size_t n_elements = 1;
    for (size_t dim : shape_value)
    {
        n_elements *= dim;
    }
    if (n != n_elements * dtype_size(dtype))
    {
        throw SafeTensorError("Invalid tensor view");
    }
}

Dtype TensorView::dtype() const { return dtype_value; }
const std::vector<size_t> &TensorView::shape() const { return shape_value; }
const std::vector<uint8_t> &TensorView::data() const { return data_value; }
size_t TensorView::data_len() const { return data_value.size(); }

// SafeTensors implementation
SafeTensors::SafeTensors(const std::vector<uint8_t> &buffer) : data(buffer)
{
    auto [n, metadata_result] = read_metadata(buffer);
    this->metadata = metadata_result;
}

// SafeTensors member function implementations
std::vector<std::pair<std::string, TensorInfo>> SafeTensors::tensors() const
{
    std::vector<std::pair<std::string, TensorInfo>> result;
    for (const auto &[name, index] : metadata.index_map)
    {
        result.emplace_back(name, metadata.tensors[index]);
    }
    return result;
}

std::vector<std::pair<std::string, TensorInfo>> SafeTensors::iter() const
{
    return tensors();
}

TensorInfo SafeTensors::tensor(const std::string &tensor_name) const
{
    auto it = metadata.index_map.find(tensor_name);
    if (it == metadata.index_map.end())
    {
        throw SafeTensorError("Tensor not found: " + tensor_name);
    }
    return metadata.tensors[it->second];
}

std::vector<std::string> SafeTensors::names() const
{
    std::vector<std::string> result;
    result.reserve(metadata.index_map.size());
    for (const auto &[name, _] : metadata.index_map)
    {
        result.push_back(name);
    }
    return result;
}

size_t SafeTensors::len() const
{
    return metadata.tensors.size();
}

bool SafeTensors::is_empty() const
{
    return metadata.tensors.empty();
}

std::pair<size_t, Metadata> SafeTensors::read_metadata(const std::vector<uint8_t> &buffer)
{
    // Read the header length (64-bit little endian)
    if (buffer.size() < 8)
    {
        throw SafeTensorError("Buffer too small to contain header length");
    }
    uint64_t header_length = *reinterpret_cast<const uint64_t *>(buffer.data());

    // Read the JSON header
    if (buffer.size() < 8 + header_length)
    {
        throw SafeTensorError("Buffer too small to contain header");
    }
    std::string header_str(reinterpret_cast<const char *>(buffer.data() + 8), header_length);

    // Parse the JSON
    json header = json::parse(header_str);

    // Extract metadata if present
    std::optional<std::unordered_map<std::string, std::string>> metadata;
    if (header.contains("__metadata__"))
    {
        metadata = header["__metadata__"].get<std::unordered_map<std::string, std::string>>();
        header.erase("__metadata__");
    }

    // Parse tensor information
    std::vector<std::pair<std::string, TensorInfo>> tensors;
    for (const auto &[name, info] : header.items())
    {
        TensorInfo tensor_info;

        std::string dtype_str = info["dtype"].get<std::string>();
        if (dtype_str == "F32")
            tensor_info.dtype = Dtype::F32;
        else if (dtype_str == "F16")
            tensor_info.dtype = Dtype::F16;

        // Parse shape
        tensor_info.shape = info["shape"].get<std::vector<size_t>>();

        // Parse data offsets
        tensor_info.data_offsets.first = info["data_offsets"][0].get<size_t>();
        tensor_info.data_offsets.second = info["data_offsets"][1].get<size_t>();

        tensors.emplace_back(name, tensor_info);
    }

    return {8 + header_length, Metadata(metadata, tensors)};
}

GPT2Weights *loadGPT2ModelWeights(const std::string &weights_file)
{
    ModelDimensions dims{0, 0, 0, 0, 0, 0, false};
    GPT2Weights *weights = nullptr;

    try
    {
        // Read the file into a buffer
        std::ifstream file(weights_file, std::ios::binary | std::ios::ate);
        if (!file.is_open())
        {
            debugPrint("Failed to open weights file: %s\n", weights_file.c_str());
            return nullptr;
        }

        std::streamsize size = file.tellg();
        file.seekg(0, std::ios::beg);

        std::vector<uint8_t> buffer(size);
        if (!file.read(reinterpret_cast<char *>(buffer.data()), size))
        {
            debugPrint("Failed to read weights file\n");
            return nullptr;
        }

        // Parse the SafeTensors file
        SafeTensors safe_tensors(buffer);
        debugPrint("Successfully loaded weights file. Found %zu tensors\n", safe_tensors.len());

        auto tensor_infos = safe_tensors.tensors();

        // Create GPT2Weights object which will also populate dims
        weights = new GPT2Weights(dims, tensor_infos, safe_tensors.get_data());

        return weights;
    }
    catch (const std::exception &e)
    {
        debugPrint("Error loading weights: %s\n", e.what());
        if (weights)
        {
            delete weights;
        }
        return nullptr;
    }
}