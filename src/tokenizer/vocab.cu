#include "tokenizer/vocab.cuh"
#include <fstream>
#include <iostream>

void loadVocabulary(const std::string& filename, std::vector<std::string>& vocab)
{
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
