#include "layers/multihead_attention.cuh"
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cassert>
#include <cudnn.h>
#include "utils/softmax.cuh"
#include <cublas_v2.h>
#include <math.h>

void computeAttentionScores(const float *Q, const float *K, float *attention_scores,
                            int batch_size, int num_heads, int seq_len, int head_dim,
                            float scale, cublasHandle_t cublas_handle, cudaStream_t stream)
{
    // Set the cuBLAS stream
    cublasSetStream(cublas_handle, stream);

    int batch_count = batch_size * num_heads;
    int m = seq_len;  // Rows of the output matrix (attention_scores)
    int n = seq_len;  // Columns of the output matrix
    int k = head_dim; // Inner dimension

    const float alpha = scale;
    const float beta = 0.0f;

    // Leading dimensions
    int lda = k;
    int ldc = n;

    // Strides between matrices in the batch
    long long int strideA = m * k; // Q
    long long int strideB = n * k; // K^T
    long long int strideC = m * n; // attention_scores

    // Perform batched matrix multiplication: attention_scores = Q * K^T
    cublasStatus_t status = cublasSgemmStridedBatched(
        cublas_handle,
        CUBLAS_OP_T,      // Transpose K
        CUBLAS_OP_N,      // Q is not transposed
        n,                // Number of columns of the output matrix
        m,                // Number of rows of the output matrix
        k,                // Shared dimension
        &alpha,           // Alpha scaling factor
        K,                // Pointer to K
        lda,              // Leading dimension of K
        strideB,          // Stride between K matrices
        Q,                // Pointer to Q
        lda,              // Leading dimension of Q
        strideA,          // Stride between Q matrices
        &beta,            // Beta scaling factor
        attention_scores, // Pointer to output
        ldc,              // Leading dimension of output
        strideC,          // Stride between output matrices
        batch_count       // Number of matrices to compute
    );

    if (status != CUBLAS_STATUS_SUCCESS)
    {
        // Handle error (e.g., throw an exception or print an error message)
    }
}

// Apply a mask to the attention scores
__global__ void maskKernel(float *attention_scores, int seq_len, float mask_value)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < seq_len * seq_len)
    {
        // Replace masked positions with mask_value (e.g., -inf)
        // Implement your masking logic here
    }
}

void applyMask(float *attention_scores, int batch_size, int num_heads, int seq_len, cudaStream_t stream)
{
    int total_elements = batch_size * num_heads * seq_len * seq_len;

    // Define mask value (e.g., a very large negative number)
    float mask_value = -1e9;

    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;

    // Launch kernel to apply the mask
    maskKernel<<<blocks, threads, 0, stream>>>(attention_scores, seq_len, mask_value);

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        // Handle error
    }
}

// Apply softmax to attention scores
void applySoftmaxToAttentionScores(float *attention_scores,
                                   int batch_size, int num_heads, int seq_len,
                                   cudnnHandle_t cudnn_handle, cudaStream_t stream)
{
    // Set the cuDNN stream
    cudnnSetStream(cudnn_handle, stream);

    // Create tensor descriptor
    cudnnTensorDescriptor_t tensor_desc;
    cudnnCreateTensorDescriptor(&tensor_desc);
    cudnnSetTensor4dDescriptor(
        tensor_desc,
        CUDNN_TENSOR_NCHW,
        CUDNN_DATA_FLOAT,
        batch_size * num_heads,
        1,
        seq_len,
        seq_len);

    const float alpha = 1.0f;
    const float beta = 0.0f;

    // Apply softmax
    cudnnStatus_t status = cudnnSoftmaxForward(
        cudnn_handle,
        CUDNN_SOFTMAX_ACCURATE,
        CUDNN_SOFTMAX_MODE_CHANNEL, // Apply softmax across the last dimension
        &alpha,
        tensor_desc,
        attention_scores,
        &beta,
        tensor_desc,
        attention_scores);

    if (status != CUDNN_STATUS_SUCCESS)
    {
        // Handle error
    }

    // Destroy tensor descriptor
    cudnnDestroyTensorDescriptor(tensor_desc);
}

void computeAttentionOutput(const float *attention_scores, const float *V, float *attention_output,
                            int batch_size, int num_heads, int seq_len, int head_dim,
                            cublasHandle_t cublas_handle, cudaStream_t stream)
{
    // Set the cuBLAS stream
    cublasSetStream(cublas_handle, stream);

    int batch_count = batch_size * num_heads;
    int m = seq_len;  // Rows of the output matrix (attention_output)
    int n = head_dim; // Columns of the output matrix
    int k = seq_len;  // Inner dimension

    const float alpha = 1.0f;
    const float beta = 0.0f;

    // Leading dimensions
    int lda = k;
    int ldb = n;
    int ldc = n;

    // Strides between matrices in the batch
    long long int strideA = m * k; // attention_scores
    long long int strideB = k * n; // V
    long long int strideC = m * n; // attention_output

    // Perform batched matrix multiplication: attention_output = attention_scores * V
    cublasStatus_t status = cublasSgemmStridedBatched(
        cublas_handle,
        CUBLAS_OP_N,      // attention_scores is not transposed
        CUBLAS_OP_N,      // V is not transposed
        n,                // Number of columns of the output matrix
        m,                // Number of rows of the output matrix
        k,                // Shared dimension
        &alpha,           // Alpha scaling factor
        V,                // Pointer to V
        ldb,              // Leading dimension of V
        strideB,          // Stride between V matrices
        attention_scores, // Pointer to attention_scores
        lda,              // Leading dimension of attention_scores
        strideA,          // Stride between attention_scores matrices
        &beta,            // Beta scaling factor
        attention_output, // Pointer to output
        ldc,              // Leading dimension of output
        strideC,          // Stride between output matrices
        batch_count       // Number of matrices to compute
    );

    if (status != CUBLAS_STATUS_SUCCESS)
    {
        // Handle error
    }
}

// New constructor: accepts pre-loaded weights
MultiHeadAttention::MultiHeadAttention(int hidden_dim, int num_heads,
                                       float *W_q_ptr, float *W_k_ptr,
                                       float *W_v_ptr, float *W_o_ptr)
{
    this->hidden_dim = hidden_dim;
    this->num_heads = num_heads;
    this->head_dim = hidden_dim / num_heads;

    // Create cuBLAS handle
    cublasCreate(&cublas_handle);

    // Use pre-loaded GPU pointers
    this->W_q = W_q_ptr;
    this->W_k = W_k_ptr;
    this->W_v = W_v_ptr;
    this->W_o = W_o_ptr;
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

// Overloaded forward method for self-attention
void MultiHeadAttention::forward(float *output,
                                 const float *input,
                                 int batch_size,
                                 int seq_len,
                                 cudaStream_t stream,
                                 bool mask)
{
    // For self-attention, Q, K, V all come from input
    forward(output, input, input, batch_size, seq_len, stream, mask);
}

// Updated forward method that accepts separate query and key/value inputs
void MultiHeadAttention::forward(float *output,
                                 const float *query_input,
                                 const float *key_value_input,
                                 int batch_size,
                                 int seq_len,
                                 cudaStream_t stream,
                                 bool mask)
{
    // Set the cuBLAS stream
    cublasSetStream(cublas_handle, stream);

    // Create cuDNN handle
    cudnnHandle_t cudnn_handle;
    cudnnCreate(&cudnn_handle);
    cudnnSetStream(cudnn_handle, stream);

    // Dimensions
    int embed_dim = hidden_dim;
    int head_dim = this->head_dim;

    // Allocate memory for Q, K, V
    float *Q, *K, *V;
    size_t size = batch_size * seq_len * embed_dim * sizeof(float);
    cudaMalloc((void **)&Q, size);
    cudaMalloc((void **)&K, size);
    cudaMalloc((void **)&V, size);

    const float alpha = 1.0f;
    const float beta = 0.0f;

    // Compute Q = query_input * W_q
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
        query_input,
        embed_dim,
        &beta,
        Q,
        embed_dim);

    // Compute K = key_value_input * W_k
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
        key_value_input,
        embed_dim,
        &beta,
        K,
        embed_dim);

    // Compute V = key_value_input * W_v
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
        key_value_input,
        embed_dim,
        &beta,
        V,
        embed_dim);

    // Reshape and transpose Q, K, V for multi-head attention
    // [Optional] Implement functions to reshape Q, K, V into [Batch * NumHeads, SeqLen, HeadDim]

    // Compute attention scores
    float *attention_scores;
    size_t attention_size = batch_size * num_heads * seq_len * seq_len * sizeof(float);
    cudaMalloc((void **)&attention_scores, attention_size);

    // Compute scaled dot-product Q * K^T
    const float scale = 1.0f / sqrtf((float)head_dim);

    // Implement batched matrix multiplication for attention scores
    // For simplicity, assuming functions to handle this
    computeAttentionScores(Q, K, attention_scores, batch_size, num_heads, seq_len, head_dim, scale, cublas_handle, stream);

    // Apply mask if required
    if (mask)
    {
        applyMask(attention_scores, batch_size, num_heads, seq_len, stream);
    }

    // Apply softmax to attention scores
    applySoftmaxToAttentionScores(attention_scores, batch_size, num_heads, seq_len, cudnn_handle, stream);

    // Compute attention output
    float *attention_output;
    cudaMalloc((void **)&attention_output, size);

    // Implement batched matrix multiplication for attention output
    computeAttentionOutput(attention_scores, V, attention_output, batch_size, num_heads, seq_len, head_dim, cublas_handle, stream);

    // Concatenate heads and project the output
    // For simplicity, we skip the concatenation step and assume functions handle the reshaping

    // Output projection: output = attention_output * W_o
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

    // Destroy cuDNN handle
    cudnnDestroy(cudnn_handle);
}