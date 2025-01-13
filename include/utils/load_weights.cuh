#pragma once

#include <vector>
#include <string>
#include <unordered_map>
#include <optional>
#include <stdexcept>

// Forward declare json to avoid requiring nlohmann/json.hpp in header
namespace nlohmann {
    class json;
}

enum class Dtype {
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

size_t dtype_size(Dtype dtype);

struct SafeTensorError : public std::runtime_error {
    using std::runtime_error::runtime_error;
};

struct TensorInfo {
    Dtype dtype;
    std::vector<size_t> shape;
    std::pair<size_t, size_t> data_offsets;
};

struct Metadata {
    std::optional<std::unordered_map<std::string, std::string>> metadata;
    std::vector<TensorInfo> tensors;
    std::unordered_map<std::string, size_t> index_map;

    Metadata(std::optional<std::unordered_map<std::string, std::string>> metadata, 
            std::vector<std::pair<std::string, TensorInfo>> tensors);
    void validate();
};

struct PreparedData {
    uint64_t n;
    std::vector<uint8_t> header_bytes;
    size_t offset;
};

struct TensorView {
    Dtype dtype;
    std::vector<size_t> shape;
    const std::vector<uint8_t>& data;

    TensorView(Dtype dtype, std::vector<size_t> shape, const std::vector<uint8_t>& data);
    
    Dtype dtype() const;
    const std::vector<size_t>& shape() const;
    const std::vector<uint8_t>& data() const;
    size_t data_len() const;
};

class SafeTensors {
public:
    SafeTensors(const std::vector<uint8_t>& buffer);
    
    std::vector<std::pair<std::string, TensorInfo>> tensors() const;
    std::vector<std::pair<std::string, TensorInfo>> iter() const;
    TensorInfo tensor(const std::string& tensor_name) const;
    std::vector<std::string> names() const;
    size_t len() const;
    bool is_empty() const;

private:
    Metadata metadata;
    const std::vector<uint8_t>& data;
    static std::pair<size_t, Metadata> read_metadata(const std::vector<uint8_t>& buffer);
};

// Template function declarations
template <typename S, typename V>
PreparedData prepare(std::vector<std::pair<S, V>>& data, 
                    const std::optional<std::unordered_map<std::string, std::string>>& data_info);

template <typename S, typename V>
std::vector<uint8_t> serialize(std::vector<std::pair<S, V>>& data,
                              const std::optional<std::unordered_map<std::string, std::string>>& data_info);

template <typename S, typename V>
void serialize_to_file(std::vector<std::pair<S, V>>& data,
                      const std::optional<std::unordered_map<std::string, std::string>>& data_info,
                      const std::string& filename);
