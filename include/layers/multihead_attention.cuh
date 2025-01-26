#ifndef MULTIHEAD_ATTENTION_H
#define MULTIHEAD_ATTENTION_H

#include <cuda.h>
#include <cublas_v2.h>
#include <curand.h>
#include <cudnn.h>

class MultiHeadAttention
{
private:
    int hidden_dim;
    int num_heads;
    int head_dim;

    // Weights for linear transformations
    float *W_q; // Query weights
    float *W_k; // Key weights
    float *W_v; // Value weights
    float *W_o; // Output projection weights

    // Biases (optional)
    float *b_q;
    float *b_k;
    float *b_v;
    float *b_o;

    // cuBLAS handle
    cublasHandle_t cublas_handle;

public:
    // Constructor that takes all weights and biases
    MultiHeadAttention(int hidden_dim, int num_heads,
                       float *W_q = nullptr, float *W_k = nullptr,
                       float *W_v = nullptr, float *W_o = nullptr,
                       float *b_q = nullptr, float *b_k = nullptr,
                       float *b_v = nullptr, float *b_o = nullptr);

    ~MultiHeadAttention();

    // Updated forward method to optionally take encoder input
    void forward(float *output,
                 const float *query_input,
                 const float *key_value_input,
                 int batch_size,
                 int seq_len,
                 cudaStream_t stream,
                 bool mask = false);

    // Overloaded method for self-attention (key_value_input not provided)
    void forward(float *output,
                 const float *input,
                 int batch_size,
                 int seq_len,
                 cudaStream_t stream,
                 bool mask = false);

    void reset(cudaStream_t stream)
    {
        // Set stream for handles
        cublasSetStream(cublas_handle, stream);
    }

    void setQueryWeight(float *weight) { W_q = weight; }
    void setQueryBias(float *bias) { b_q = bias; }
    void setOutputProjWeight(float *weight) { W_o = weight; }
    void setOutputProjBias(float *bias) { b_o = bias; }

    // Add setters for key and value weights/biases
    void setKeyWeight(float *weight) { W_k = weight; }
    void setKeyBias(float *bias) { b_k = bias; }
    void setValueWeight(float *weight) { W_v = weight; }
    void setValueBias(float *bias) { b_v = bias; }
};

#endif // MULTIHEAD_ATTENTION_H