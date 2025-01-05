#ifndef MULTIHEAD_ATTENTION_H
#define MULTIHEAD_ATTENTION_H

class MultiHeadAttention {
private:
    int hidden_dim;
    int num_heads;
    // Weights and biases

public:
    MultiHeadAttention(int hidden_dim, int num_heads);
    ~MultiHeadAttention();

    void forward(float* output, const float* input, int seq_len, cudaStream_t stream);
};

#endif // MULTIHEAD_ATTENTION_H 