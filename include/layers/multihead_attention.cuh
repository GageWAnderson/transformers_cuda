#ifndef MULTIHEAD_ATTENTION_H
#define MULTIHEAD_ATTENTION_H

#include <cuda.h>
#include <cublas_v2.h>
#include <curand.h>
#include <cudnn.h>

class MultiHeadAttention {
private:
    int hidden_dim;
    int num_heads;
    int head_dim;

    // Weights for linear transformations
    float* W_q; // Query weights
    float* W_k; // Key weights
    float* W_v; // Value weights
    float* W_o; // Output projection weights

    // Biases (optional)
    float* b_q;
    float* b_k;
    float* b_v;
    float* b_o;

    // cuBLAS handle
    cublasHandle_t cublas_handle;

public:
    MultiHeadAttention(int hidden_dim, int num_heads);
    ~MultiHeadAttention();

    // Updated forward method to optionally take encoder input
    void forward(float* output,
                 const float* query_input,
                 const float* key_value_input,
                 int batch_size,
                 int seq_len,
                 cudaStream_t stream,
                 bool mask = false);

    // Overloaded method for self-attention (key_value_input not provided)
    void forward(float* output,
                 const float* input,
                 int batch_size,
                 int seq_len,
                 cudaStream_t stream,
                 bool mask = false);
};

#endif // MULTIHEAD_ATTENTION_H 