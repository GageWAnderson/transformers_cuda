#include "layers/multihead_attention.cuh"
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cassert>
#include <cudnn.h>
#include "utils/softmax.cuh"

MultiHeadAttention::MultiHeadAttention(int hidden_dim, int num_heads)
{
    this->hidden_dim = hidden_dim;
    this->num_heads = num_heads;
    this->head_dim = hidden_dim / num_heads;

    // Initialize cuBLAS handle
    cublasCreate(&cublas_handle);

    // Allocate memory for weights and biases
    size_t weight_size = hidden_dim * hidden_dim * sizeof(float); // Assuming square matrices for simplicity
    cudaMalloc((void **)&W_q, weight_size);
    cudaMalloc((void **)&W_k, weight_size);
    cudaMalloc((void **)&W_v, weight_size);
    cudaMalloc((void **)&W_o, weight_size);

    // Optionally allocate biases here if you use them

    // Initialize weights with random values
    curandGenerator_t curand_gen;
    curandCreateGenerator(&curand_gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(curand_gen, 1234ULL);

    curandGenerateUniform(curand_gen, W_q, hidden_dim * hidden_dim);
    curandGenerateUniform(curand_gen, W_k, hidden_dim * hidden_dim);
    curandGenerateUniform(curand_gen, W_v, hidden_dim * hidden_dim);
    curandGenerateUniform(curand_gen, W_o, hidden_dim * hidden_dim);

    // Destroy cuRAND generator
    curandDestroyGenerator(curand_gen);
}

MultiHeadAttention::~MultiHeadAttention()
{
    // Free weights and biases
    cudaFree(W_q);
    cudaFree(W_k);
    cudaFree(W_v);
    cudaFree(W_o);

    // Destroy cuBLAS handle
    cublasDestroy(cublas_handle);
}

void MultiHeadAttention::forward(float *output, const float *input, int batch_size, int seq_len, cudaStream_t stream)
{
    // Set the cuBLAS stream
    cublasSetStream(cublas_handle, stream);

    // Dimensions
    int embed_dim = hidden_dim;
    int head_dim = this->head_dim; // Corrected to use the member variable

    // Allocate memory for Q, K, V, and attention scores
    float *Q;
    float *K;
    float *V;
    cudaMalloc((void **)&Q, batch_size * seq_len * embed_dim * sizeof(float));
    cudaMalloc((void **)&K, batch_size * seq_len * embed_dim * sizeof(float));
    cudaMalloc((void **)&V, batch_size * seq_len * embed_dim * sizeof(float));

    // Linear projections: Q = input * W_q
    const float alpha = 1.0f;
    const float beta = 0.0f;
    cublasSgemm(
        cublas_handle,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        embed_dim,
        batch_size * seq_len,
        embed_dim,
        &alpha,
        W_q,
        embed_dim,
        input,
        embed_dim,
        &beta,
        Q,
        embed_dim);

    // Repeat for K and V
    cublasSgemm(
        cublas_handle,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        embed_dim,
        batch_size * seq_len,
        embed_dim,
        &alpha,
        W_k,
        embed_dim,
        input,
        embed_dim,
        &beta,
        K,
        embed_dim);

    cublasSgemm(
        cublas_handle,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        embed_dim,
        batch_size * seq_len,
        embed_dim,
        &alpha,
        W_v,
        embed_dim,
        input,
        embed_dim,
        &beta,
        V,
        embed_dim);

    // Reshape Q, K, V if necessary
    // Skipping for simplicity

    // Compute scaled dot-product attention scores
    float *attention_scores;
    cudaMalloc((void **)&attention_scores, batch_size * num_heads * seq_len * seq_len * sizeof(float));

    const float scale = 1.0f / sqrtf((float)head_dim);

    // Compute attention scores using cublasSgemmStridedBatched
    cublasSgemmStridedBatched(
        cublas_handle,
        CUBLAS_OP_T,
        CUBLAS_OP_N,
        seq_len,
        seq_len,
        head_dim,
        &scale,
        K,
        head_dim,
        seq_len * head_dim,
        Q,
        head_dim,
        seq_len * head_dim,
        &beta,
        attention_scores,
        seq_len,
        seq_len * seq_len,
        batch_size * num_heads);

    // Apply softmax to attention_scores
    // Initialize cuDNN handle
    cudnnHandle_t cudnn;
    cudnnCreate(&cudnn);
    cudnnSetStream(cudnn, stream);

    // Apply softmax using the provided function
    int total_elements = batch_size * num_heads * seq_len * seq_len;
    applySoftmax(cudnn, attention_scores, attention_scores, total_elements);

    // Destroy cuDNN handle
    cudnnDestroy(cudnn);

    // Compute attention output: attention_output = attention_scores * V
    float *attention_output;
    cudaMalloc((void **)&attention_output, batch_size * seq_len * embed_dim * sizeof(float));

    cublasSgemmStridedBatched(
        cublas_handle,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        head_dim,
        seq_len,
        seq_len,
        &alpha,
        V,
        head_dim,
        seq_len * head_dim,
        attention_scores,
        seq_len,
        seq_len * seq_len,
        &beta,
        attention_output,
        head_dim,
        seq_len * head_dim,
        batch_size * num_heads);

    // Concatenate heads and project the output: output = attention_output * W_o
    cublasSgemm(
        cublas_handle,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        embed_dim,
        batch_size * seq_len,
        embed_dim,
        &alpha,
        W_o,
        embed_dim,
        attention_output,
        embed_dim,
        &beta,
        output,
        embed_dim);

    // Free allocated memory
    cudaFree(Q);
    cudaFree(K);
    cudaFree(V);
    cudaFree(attention_scores);
    cudaFree(attention_output);
}