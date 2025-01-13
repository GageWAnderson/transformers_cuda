#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <unordered_map>
#include <stdexcept>
#include <sstream>
#include <nlohmann/json.hpp>
#include "utils/debug.cuh"

using json = nlohmann::json;

const size_t MAX_HEADER_SIZE = 100000000;
enum class Dtype
{
    BOOL,
    U8,
    I8,
    F8_E5M2,
    F8_E4M3,
    I16,
    U16,
    F16,
    BF16,
    I32,
    U32,
    F32,
    F64,
    I64,
    U64
};

static Dtype valueOf(const std::string &dtype_str)
{
    if (dtype_str == "BOOL")
        return Dtype::BOOL;
    if (dtype_str == "U8")
        return Dtype::U8;
    if (dtype_str == "I8")
        return Dtype::I8;
    if (dtype_str == "F8_E5M2")
        return Dtype::F8_E5M2;
    if (dtype_str == "F8_E4M3")
        return Dtype::F8_E4M3;
    if (dtype_str == "I16")
        return Dtype::I16;
    if (dtype_str == "U16")
        return Dtype::U16;
    if (dtype_str == "F16")
        return Dtype::F16;
    if (dtype_str == "BF16")
        return Dtype::BF16;
    if (dtype_str == "I32")
        return Dtype::I32;
    if (dtype_str == "U32")
        return Dtype::U32;
    if (dtype_str == "F32")
        return Dtype::F32;
    if (dtype_str == "F64")
        return Dtype::F64;
    if (dtype_str == "I64")
        return Dtype::I64;
    if (dtype_str == "U64")
        return Dtype::U64;
    throw std::invalid_argument("Unknown dtype: " + dtype_str);
}

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

struct SafeTensorError : public std::runtime_error
{
    using std::runtime_error::runtime_error;
};

struct TensorInfo
{
    Dtype dtype;
    std::vector<size_t> shape;
    std::pair<size_t, size_t> data_offsets;
};

struct Metadata
{
    bool has_metadata;
    std::unordered_map<std::string, std::string> metadata;
    std::vector<TensorInfo> tensors;
    std::unordered_map<std::string, size_t> index_map;

    Metadata() = default;

    Metadata(bool has_metadata, std::unordered_map<std::string, std::string> metadata, std::vector<std::pair<std::string, TensorInfo>> tensors)
    {
        this->has_metadata = has_metadata;
        this->metadata = metadata;
        for (size_t i = 0; i < tensors.size(); ++i)
        {
            this->index_map[tensors[i].first] = i;
            this->tensors.push_back(tensors[i].second);
        }
        validate();
    }

    void validate()
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
};

struct PreparedData
{
    uint64_t n;
    std::vector<uint8_t> header_bytes;
    size_t offset;
};

template <typename S, typename V>
PreparedData prepare(std::vector<std::pair<S, V>> &data,
                     const std::unordered_map<std::string, std::string> *data_info = nullptr)
{
    std::sort(data.begin(), data.end(), [](const auto &left, const auto &right)
              { return std::tie(right.second.dtype(), left.first) < std::tie(left.second.dtype(), right.first); });

    std::vector<V> tensors;
    std::vector<std::pair<std::string, TensorInfo>> hmetadata;
    size_t offset = 0;

    for (auto &[name, tensor] : data)
    {
        size_t n = tensor.data_len();
        TensorInfo tensor_info = {tensor.dtype(), tensor.shape(), {offset, offset + n}};
        offset += n;
        hmetadata.push_back({name, tensor_info});
        tensors.push_back(tensor);
    }

    Metadata metadata(data_info != nullptr,
                      data_info ? *data_info : std::unordered_map<std::string, std::string>(),
                      hmetadata);
    std::string json_str = json(metadata).dump();
    std::vector<uint8_t> metadata_buf(json_str.begin(), json_str.end());
    size_t extra = (8 - metadata_buf.size() % 8) % 8;
    metadata_buf.insert(metadata_buf.end(), extra, ' ');

    uint64_t n = metadata_buf.size();

    return {n, metadata_buf, offset};
}

template <typename S, typename V>
std::vector<uint8_t> serialize(std::vector<std::pair<S, V>> &data, const std::unordered_map<std::string, std::string> &data_info = std::unordered_map<std::string, std::string>())
{
    PreparedData prepared_data = prepare(data, &data_info);
    uint64_t n = prepared_data.n;
    std::vector<uint8_t> header_bytes = prepared_data.header_bytes;
    size_t offset = prepared_data.offset;
    size_t expected_size = 8 + header_bytes.size() + offset;
    std::vector<uint8_t> buffer;
    buffer.reserve(expected_size);
    buffer.insert(buffer.end(), reinterpret_cast<uint8_t *>(&n), reinterpret_cast<uint8_t *>(&n) + 8);
    buffer.insert(buffer.end(), header_bytes.begin(), header_bytes.end());
    for (auto &[name, tensor] : data)
    {
        auto tensor_data = tensor.data();
        buffer.insert(buffer.end(), tensor_data.begin(), tensor_data.end());
    }
    return buffer;
}

template <typename S, typename V>
void serialize_to_file(std::vector<std::pair<S, V>> &data, const std::unordered_map<std::string, std::string> &data_info, const std::string &filename)
{
    PreparedData prepared_data = prepare(data, &data_info);
    uint64_t n = prepared_data.n;
    std::vector<uint8_t> header_bytes = prepared_data.header_bytes;
    size_t offset = prepared_data.offset;
    std::ofstream file(filename, std::ios::binary);
    if (!file)
    {
        throw SafeTensorError("Failed to open file");
    }
    file.write(reinterpret_cast<const char *>(&n), 8);
    file.write(reinterpret_cast<const char *>(header_bytes.data()), header_bytes.size());
    for (auto &[name, tensor] : data)
    {
        auto tensor_data = tensor.data();
        file.write(reinterpret_cast<const char *>(tensor_data.data()), tensor_data.size());
    }
    file.close();
}

void from_json(const json &j, Metadata &m)
{
    // Handle __metadata__ field
    m.has_metadata = j.contains("__metadata__");
    if (m.has_metadata)
    {
        const auto &meta = j.at("__metadata__");
        for (auto it = meta.begin(); it != meta.end(); ++it)
        {
            m.metadata[it.key()] = it.value().get<std::string>();
        }
    }

    std::vector<std::pair<std::string, TensorInfo>> tensors;

    // Process all fields except __metadata__
    for (auto it = j.begin(); it != j.end(); ++it)
    {
        if (it.key() == "__metadata__")
            continue;

        const auto &tensor = it.value();
        if (!tensor.contains("dtype") || !tensor.contains("shape") || !tensor.contains("data_offsets"))
        {
            throw SafeTensorError("Tensor missing required fields (dtype, shape, or data_offsets) for " + it.key());
        }

        TensorInfo info;
        info.dtype = valueOf(tensor.at("dtype").get<std::string>());

        // Handle shape values that might be strings
        auto shape_array = tensor.at("shape");
        info.shape.clear();
        for (const auto &dim : shape_array)
        {
            if (dim.is_string())
            {
                info.shape.push_back(std::stoull(dim.get<std::string>()));
            }
            else
            {
                info.shape.push_back(dim.get<size_t>());
            }
        }

        // Handle data_offsets that might be strings
        auto offsets = tensor.at("data_offsets");
        if (offsets.is_array() && offsets.size() == 2)
        {
            size_t start = offsets[0].is_string() ? std::stoull(offsets[0].get<std::string>()) : offsets[0].get<size_t>();
            size_t end = offsets[1].is_string() ? std::stoull(offsets[1].get<std::string>()) : offsets[1].get<size_t>();
            info.data_offsets = {start, end};
        }
        else
        {
            throw SafeTensorError("Invalid data_offsets format for " + it.key());
        }

        tensors.push_back({it.key(), info});
    }

    m = Metadata(m.has_metadata, m.metadata, tensors);
}

struct SafeTensors
{
    Metadata metadata;
    const std::vector<uint8_t> &data;

    SafeTensors(const std::vector<uint8_t> &buffer) : data(buffer)
    {
        auto [n, metadata] = read_metadata(buffer);
        this->metadata = metadata;
    }

    static bool utf8_validation(const std::string &str)
    {
        const unsigned char *bytes = (const unsigned char *)str.c_str();
        size_t len = str.length();

        for (size_t i = 0; i < len; i++)
        {
            if (bytes[i] <= 0x7F)
            {
                continue;
            }
            else if (bytes[i] >= 0xC0 && bytes[i] <= 0xDF)
            {
                if (i + 1 >= len || (bytes[i + 1] & 0xC0) != 0x80)
                    return false;
                i += 1;
            }
            else if (bytes[i] >= 0xE0 && bytes[i] <= 0xEF)
            {
                if (i + 2 >= len || (bytes[i + 1] & 0xC0) != 0x80 ||
                    (bytes[i + 2] & 0xC0) != 0x80)
                    return false;
                i += 2;
            }
            else if (bytes[i] >= 0xF0 && bytes[i] <= 0xF7)
            {
                if (i + 3 >= len || (bytes[i + 1] & 0xC0) != 0x80 ||
                    (bytes[i + 2] & 0xC0) != 0x80 ||
                    (bytes[i + 3] & 0xC0) != 0x80)
                    return false;
                i += 3;
            }
            else
            {
                return false;
            }
        }
        return true;
    }

    static std::pair<size_t, Metadata> read_metadata(const std::vector<uint8_t> &buffer)
    {
        if (buffer.size() < 8)
        {
            debugPrint("Buffer size too small: %zu\n", buffer.size());
            throw SafeTensorError("Header too small");
        }
        uint64_t n = buffer[0] |
                     (uint64_t)buffer[1] << 8 |
                     (uint64_t)buffer[2] << 16 |
                     (uint64_t)buffer[3] << 24 |
                     (uint64_t)buffer[4] << 32 |
                     (uint64_t)buffer[5] << 40 |
                     (uint64_t)buffer[6] << 48 |
                     (uint64_t)buffer[7] << 56;

        if (n > MAX_HEADER_SIZE)
        {
            debugPrint("Header too large: n=%llu, buffer size=%zu\n", n, buffer.size());
            throw SafeTensorError("Header too large");
        }
        size_t stop = n + 8;
        if (stop > buffer.size())
        {
            debugPrint("Invalid header length: n=%llu, stop=%zu, buffer size=%zu\n", n, stop, buffer.size());
            throw SafeTensorError("Invalid header length");
        }

        std::string header_str(buffer.begin() + 8, buffer.begin() + stop);
        if (!utf8_validation(header_str))
        {
            debugPrint("Invalid UTF-8 in header: n=%llu, buffer size=%zu\n", n, buffer.size());
            throw SafeTensorError("Invalid header: not valid UTF-8");
        }

        auto metadata_json = json::parse(header_str);
        Metadata metadata;
        from_json(metadata_json, metadata);
        metadata.validate();
        if (8 + n > buffer.size())
        {
            debugPrint("Metadata incomplete buffer: n=%llu, buffer size=%zu\n", n, buffer.size());
            throw SafeTensorError("Metadata incomplete buffer");
        }
        // Print the loaded JSON metadata
        debugPrint("Loaded JSON metadata: %s\n", metadata_json.dump(4).c_str());
        return {n, metadata};
    }

    std::vector<std::pair<std::string, TensorInfo>> tensors() const
    {
        std::vector<std::pair<std::string, TensorInfo>> tensors;
        for (const auto &[name, index] : metadata.index_map)
        {
            const auto &info = metadata.tensors[index];
            tensors.push_back({name, info});
        }
        return tensors;
    }

    std::vector<std::pair<std::string, TensorInfo>> iter() const
    {
        return tensors();
    }

    TensorInfo tensor(const std::string &tensor_name) const
    {
        auto it = metadata.index_map.find(tensor_name);
        if (it == metadata.index_map.end())
        {
            throw SafeTensorError("Tensor not found: " + tensor_name);
        }
        return metadata.tensors[it->second];
    }

    std::vector<std::string> names() const
    {
        std::vector<std::string> names;
        for (const auto &[name, _] : metadata.index_map)
        {
            names.push_back(name);
        }
        return names;
    }

    size_t len() const
    {
        return metadata.tensors.size();
    }

    bool is_empty() const
    {
        return metadata.tensors.empty();
    }
};

struct TensorView
{
    Dtype dtype_;
    std::vector<size_t> shape_;
    const std::vector<uint8_t> &data_;

    TensorView(Dtype dtype, std::vector<size_t> shape, const std::vector<uint8_t> &data)
        : dtype_(dtype), shape_(shape), data_(data)
    {
        size_t n = data.size();
        size_t n_elements = 1;
        for (size_t dim : shape_)
        {
            n_elements *= dim;
        }
        if (n != n_elements * dtype_size(dtype))
        {
            throw SafeTensorError("Invalid tensor view");
        }
    }

    Dtype dtype() const
    {
        return dtype_;
    }

    const std::vector<size_t> &shape() const
    {
        return shape_;
    }

    const std::vector<uint8_t> &data() const
    {
        return data_;
    }

    size_t data_len() const
    {
        return data_.size();
    }
};

bool loadModelWeights(const std::string &weights_file)
{
    try
    {
        // Read the file into a buffer
        std::ifstream file(weights_file, std::ios::binary | std::ios::ate);
        if (!file.is_open())
        {
            debugPrint("Failed to open weights file: %s\n", weights_file.c_str());
            return false;
        }

        std::streamsize size = file.tellg();
        file.seekg(0, std::ios::beg);

        std::vector<uint8_t> buffer(size);
        if (!file.read(reinterpret_cast<char *>(buffer.data()), size))
        {
            debugPrint("Failed to read weights file\n");
            return false;
        }

        // Parse the SafeTensors file
        SafeTensors safe_tensors(buffer);

        debugPrint("Successfully loaded weights file. Found %zu tensors\n", safe_tensors.len());

        // Print names of available tensors for debugging
        auto tensor_names = safe_tensors.names();
        debugPrint("Available tensors:\n");
        for (const auto &name : tensor_names)
        {
            auto info = safe_tensors.tensor(name);
            debugPrint("  %s: ", name.c_str());
            for (size_t dim : info.shape)
            {
                debugPrint("%zu ", dim);
            }
            debugPrint("\n");
        }

        return true;
    }
    catch (const SafeTensorError &e)
    {
        debugPrint("Error loading weights: %s\n", e.what());
        return false;
    }
    catch (const std::exception &e)
    {
        debugPrint("Unexpected error loading weights: %s\n", e.what());
        return false;
    }
}