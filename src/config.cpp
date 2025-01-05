#include "config.cuh"
#include <fstream>
#include <sstream>
#include <iostream>

Config::Config() : num_layers(6), hidden_dim(512), num_heads(8),
                   intermediate_dim(2048), vocab_size(30522), embedding_dim(512), max_seq_len(512) {
    // Default values are set here
}

bool Config::loadFromFile(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open config file: " << filename << std::endl;
        return false;
    }

    std::string line, section;
    while (std::getline(file, line)) {
        // Trim whitespace
        line.erase(0, line.find_first_not_of(" \t\r\n"));
        line.erase(line.find_last_not_of(" \t\r\n") + 1);

        // Skip empty lines and comments
        if (line.empty() || line[0] == ';' || line[0] == '#')
            continue;

        // Handle section headers
        if (line.front() == '[' && line.back() == ']') {
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
        if (section == "Transformer") {
            if (key == "num_layers") {
                num_layers = std::stoi(value);
            } else if (key == "hidden_dim") {
                hidden_dim = std::stoi(value);
            } else if (key == "num_heads") {
                num_heads = std::stoi(value);
            } else if (key == "intermediate_dim") {
                intermediate_dim = std::stoi(value);
            } else if (key == "vocab_size") {
                vocab_size = std::stoi(value);
            } else if (key == "embedding_dim") {
                embedding_dim = std::stoi(value);
            } else if (key == "vocab_file") {
                vocab_file = value;
            } else if (key == "max_seq_len") {
                max_seq_len = std::stoi(value);
            }
        }
    }

    file.close();
    return true;
}
