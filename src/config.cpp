#include "config.cuh"
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>

Config::Config() : num_layers(6), hidden_dim(512), num_heads(8),
                   intermediate_dim(2048), vocab_size(30522), embedding_dim(512), max_seq_len(512),
                   batch_size(1), max_generation_length(50), start_token_id(2), stop_token_id(3),
                   temperature(0.8),
                   model_arch(ModelArchitecture::GPT2)
{
    // Default values are set here
}

ModelArchitecture Config::parseModelArchitecture(const std::string &arch_str)
{
    // Convert to lowercase for case-insensitive comparison
    std::string lower_arch = arch_str;
    std::transform(lower_arch.begin(), lower_arch.end(), lower_arch.begin(), ::tolower);

    if (lower_arch == "gpt2")
    {
        return ModelArchitecture::GPT2;
    }

    // If not recognized, return UNKNOWN
    return ModelArchitecture::UNKNOWN;
}

bool Config::loadFromFile(const std::string &filename)
{
    std::ifstream file(filename);
    if (!file.is_open())
    {
        std::cerr << "Failed to open config file: " << filename << std::endl;
        return false;
    }

    std::string line, section;
    while (std::getline(file, line))
    {
        // Trim whitespace
        line.erase(0, line.find_first_not_of(" \t\r\n"));
        line.erase(line.find_last_not_of(" \t\r\n") + 1);

        // Skip empty lines and comments
        if (line.empty() || line[0] == ';' || line[0] == '#')
            continue;

        // Handle section headers
        if (line.front() == '[' && line.back() == ']')
        {
            section = line.substr(1, line.size() - 2);
            continue;
        }

        // Parse key-value pairs
        size_t delimiter_pos = line.find('=');
        if (delimiter_pos == std::string::npos)
            continue;

        std::string key = line.substr(0, delimiter_pos);
        std::string value = line.substr(delimiter_pos + 1);

        // Remove any whitespace around the key and value
        key.erase(0, key.find_first_not_of(" \t\r\n"));
        key.erase(key.find_last_not_of(" \t\r\n") + 1);
        value.erase(0, value.find_first_not_of(" \t\r\n"));
        value.erase(value.find_last_not_of(" \t\r\n") + 1);

        // Assign values based on the key
        if (section == "Transformer")
        {
            if (key == "num_layers")
            {
                num_layers = std::stoi(value);
            }
            else if (key == "hidden_dim")
            {
                hidden_dim = std::stoi(value);
            }
            else if (key == "num_heads")
            {
                num_heads = std::stoi(value);
            }
            else if (key == "intermediate_dim")
            {
                intermediate_dim = std::stoi(value);
            }
            else if (key == "vocab_size")
            {
                vocab_size = std::stoi(value);
            }
            else if (key == "embedding_dim")
            {
                embedding_dim = std::stoi(value);
            }
            else if (key == "vocab_file")
            {
                vocab_file = value;
            }
            else if (key == "max_seq_len")
            {
                max_seq_len = std::stoi(value);
            }
            else if (key == "batch_size")
            {
                batch_size = std::stoi(value);
            }
            else if (key == "max_generation_length")
            {
                max_generation_length = std::stoi(value);
            }
            else if (key == "start_token_id")
            {
                start_token_id = std::stoi(value);
            }
            else if (key == "stop_token_id")
            {
                stop_token_id = std::stoi(value);
            }
            else if (key == "model_arch")
            {
                model_arch = parseModelArchitecture(value);
                if (model_arch == ModelArchitecture::UNKNOWN)
                {
                    std::cerr << "Warning: Unknown model architecture '" << value
                              << "'. Defaulting to GPT2." << std::endl;
                    model_arch = ModelArchitecture::GPT2;
                }
            }
            else if (key == "temperature")
            {
                temperature = std::stof(value);
            }
        }
    }

    file.close();

    // After loading, validate the architecture
    if (!isArchitectureSupported())
    {
        std::cerr << "Error: Model architecture '" << static_cast<int>(model_arch)
                  << "' is not supported." << std::endl;
        return false;
    }

    return true;
}

void Config::updateFromWeights(const ModelDimensions &dims)
{
    if (!dims.valid)
    {
        return;
    }
    num_layers = dims.num_layers;
    hidden_dim = dims.hidden_dim;
    num_heads = dims.num_heads;
    intermediate_dim = dims.intermediate_dim;
    vocab_size = dims.vocab_size;
    embedding_dim = dims.embedding_dim;
}
