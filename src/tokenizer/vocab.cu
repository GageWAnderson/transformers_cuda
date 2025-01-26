#include "tokenizer/vocab.cuh"
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <nlohmann/json.hpp>

void loadVocabularyJson(const std::string& filename, std::vector<std::string>& vocab)
{
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open JSON vocabulary file: " << filename << std::endl;
        return;
    }

    try {
        nlohmann::json jsonData;
        file >> jsonData;

        // Clear and resize the vocab vector
        vocab.clear();
        vocab.resize(10000); // Reasonable initial size for GPT-2 vocab

        size_t max_token_id = 0;
        for (auto& [key, value] : jsonData.items()) {
            int token_id = value.get<int>();
            std::string token = key;

            // Ensure vector is large enough
            if (token_id >= vocab.size()) {
                vocab.resize(token_id + 1);
            }

            // Store token at its ID position
            vocab[token_id] = token;
            max_token_id = std::max(max_token_id, static_cast<size_t>(token_id));
        }

        // Resize to actual vocabulary size
        vocab.resize(max_token_id + 1);
    } catch (const nlohmann::json::exception& e) {
        std::cerr << "JSON parsing error in vocabulary file: " << e.what() << std::endl;
    }

    file.close();
}

void loadVocabulary(const std::string& filename, std::vector<std::string>& vocab)
{
    // Get file extension
    std::string ext = filename.substr(filename.find_last_of(".") + 1);
    
    if (ext == "json") {
        loadVocabularyJson(filename, vocab);
        return;
    }

    // Existing text file loading logic
    std::ifstream file(filename);
    if (!file.is_open())
    {
        std::cerr << "Failed to open vocabulary file: " << filename << std::endl;
        return;
    }

    std::string token;
    while (std::getline(file, token))
    {
        // Remove any trailing carriage return character (for Windows line endings)
        if (!token.empty() && token.back() == '\r')
        {
            token.pop_back();
        }
        vocab.push_back(token);
    }

    file.close();
}
