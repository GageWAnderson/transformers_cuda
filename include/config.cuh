#ifndef CONFIG_H
#define CONFIG_H

#include <string>
#include "model_dimensions.cuh"

enum class ModelArchitecture {
    GPT2,
    UNKNOWN
};

struct Config
{
    int num_layers;
    int hidden_dim;
    int num_heads;
    int intermediate_dim;
    int vocab_size;
    int embedding_dim;
    int max_seq_len;
    int batch_size;
    int max_generation_length;
    int start_token_id;
    int stop_token_id;
    float temperature;
    std::string vocab_file;
    ModelArchitecture model_arch;

    // Constructor to set default values
    Config();

    public:
    void updateFromWeights(const ModelDimensions &dims);
    ModelArchitecture parseModelArchitecture(const std::string& arch_str);

    // Function to load configurations from a file
    bool loadFromFile(const std::string &filename);

    // Helper to check if architecture is supported
    bool isArchitectureSupported() const {
        return model_arch == ModelArchitecture::GPT2;
    }
};

#endif // CONFIG_H
