#pragma once

struct ModelDimensions {
    int num_layers;
    int hidden_dim; 
    int num_heads;
    int intermediate_dim;
    int vocab_size;
    int embedding_dim;
    bool valid;
}; 