#ifndef CONFIG_H
#define CONFIG_H

#include <string>

struct Config
{
    int num_layers;
    int hidden_dim;
    int num_heads;
    int intermediate_dim;
    int vocab_size;
    int embedding_dim;
    int max_seq_len;
    std::string vocab_file;

    // Constructor to set default values
    Config();

    // Function to load configurations from a file
    bool loadFromFile(const std::string &filename);
};

#endif // CONFIG_H
