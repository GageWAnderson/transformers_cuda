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
    int batch_size;
    int max_generation_length;
    int start_token_id;
    int stop_token_id;
    std::string vocab_file;

    // Constructor to set default values
    Config();

    // Function to load configurations from a file
    bool loadFromFile(const std::string &filename);
};

#endif // CONFIG_H
