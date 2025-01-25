#include "layers/multihead_attention.cuh"
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cassert>
#include <cudnn.h>
#include "utils/softmax.cuh"
#include <cublas_v2.h>
#include <math.h>

#include "utils/utils.cuh"
#include "utils/debug.cuh"

__global__ void scaleKernel(float *attention_scores, int n, float scale)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        attention_scores[idx] *= scale;
    }
}

__global__ void clipValuesKernel(float *data, int n, float max_val, float min_val)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        data[idx] = fmaxf(fminf(data[idx], max_val), min_val);
    }
}

void computeAttentionScores(const float *Q, const float *K, float *attention_scores,
                            int batch_size, int num_heads, int seq_len, int head_dim,
                            float scale, cublasHandle_t cublas_handle, cudaStream_t stream)
{
    // Add debug prints for dimensions and scale
    debugPrint("Computing attention scores with scale=%f, head_dim=%d\n", scale, head_dim);

    // Verify inputs are valid
    float h_check[5];
    cudaMemcpy(h_check, Q, 5 * sizeof(float), cudaMemcpyDeviceToHost);
    debugPrint("Q values before matmul: %f %f %f %f %f\n",
               h_check[0], h_check[1], h_check[2], h_check[3], h_check[4]);

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

    // Scale the attention scores after multiplication
    int total_elements = batch_count * m * n;
    scaleKernel<<<(total_elements + 255) / 256, 256, 0, stream>>>(
        attention_scores, total_elements, scale);

    // Verify outputs
    cudaMemcpy(h_check, attention_scores, 5 * sizeof(float), cudaMemcpyDeviceToHost);
    debugPrint("Attention scores after scaling: %f %f %f %f %f\n",
               h_check[0], h_check[1], h_check[2], h_check[3], h_check[4]);
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
    // Add numerical stability check before softmax
    float h_check[5];
    cudaMemcpy(h_check, attention_scores, 5 * sizeof(float), cudaMemcpyDeviceToHost);
    debugPrint("Pre-softmax scores: %f %f %f %f %f\n",
               h_check[0], h_check[1], h_check[2], h_check[3], h_check[4]);

    // Clip extremely large values to prevent overflow
    int total_elements = batch_size * num_heads * seq_len * seq_len;
    clipValuesKernel<<<(total_elements + 255) / 256, 256, 0, stream>>>(
        attention_scores, total_elements, 1e4f, -1e4f);

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

    // Verify output
    cudaMemcpy(h_check, attention_scores, 5 * sizeof(float), cudaMemcpyDeviceToHost);
    debugPrint("Post-softmax scores: %f %f %f %f %f\n",
               h_check[0], h_check[1], h_check[2], h_check[3], h_check[4]);
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

// Update constructor to take references instead of copying
MultiHeadAttention::MultiHeadAttention(int hidden_dim, int num_heads,
                                       float *W_q_ptr, float *W_k_ptr,
                                       float *W_v_ptr, float *W_o_ptr,
                                       float *b_q_ptr, float *b_k_ptr,
                                       float *b_v_ptr, float *b_o_ptr)
{
    this->hidden_dim = hidden_dim;
    this->num_heads = num_heads;
    this->head_dim = hidden_dim / num_heads;

    // Store references to weights instead of copying
    this->W_q = W_q_ptr;
    this->W_k = W_k_ptr;
    this->W_v = W_v_ptr;
    this->W_o = W_o_ptr;

    // Store references to biases
    this->b_q = b_q_ptr;
    this->b_k = b_k_ptr;
    this->b_v = b_v_ptr;
    this->b_o = b_o_ptr;

    // Create cuBLAS handle
    cublasCreate(&cublas_handle);

    // Log weights and biases
    float h_Wq[5], h_Wk[5], h_Wv[5], h_Wo[5];
    cudaMemcpy(h_Wq, W_q, 5 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_Wk, W_k, 5 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_Wv, W_v, 5 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_Wo, W_o, 5 * sizeof(float), cudaMemcpyDeviceToHost);
    debugPrint("MultiHeadAttention W_q (first 5): %f %f %f %f %f\n",
               h_Wq[0], h_Wq[1], h_Wq[2], h_Wq[3], h_Wq[4]);
    debugPrint("MultiHeadAttention W_k (first 5): %f %f %f %f %f\n",
               h_Wk[0], h_Wk[1], h_Wk[2], h_Wk[3], h_Wk[4]);
    debugPrint("MultiHeadAttention W_v (first 5): %f %f %f %f %f\n",
               h_Wv[0], h_Wv[1], h_Wv[2], h_Wv[3], h_Wv[4]);
    debugPrint("MultiHeadAttention W_o (first 5): %f %f %f %f %f\n",
               h_Wo[0], h_Wo[1], h_Wo[2], h_Wo[3], h_Wo[4]);

    float h_bq[5], h_bk[5], h_bv[5], h_bo[5];
    cudaMemcpy(h_bq, b_q, 5 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_bk, b_k, 5 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_bv, b_v, 5 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_bo, b_o, 5 * sizeof(float), cudaMemcpyDeviceToHost);
    debugPrint("MultiHeadAttention b_q (first 5): %f %f %f %f %f\n",
               h_bq[0], h_bq[1], h_bq[2], h_bq[3], h_bq[4]);
    debugPrint("MultiHeadAttention b_k (first 5): %f %f %f %f %f\n",
               h_bk[0], h_bk[1], h_bk[2], h_bk[3], h_bk[4]);
    debugPrint("MultiHeadAttention b_v (first 5): %f %f %f %f %f\n",
               h_bv[0], h_bv[1], h_bv[2], h_bv[3], h_bv[4]);
    debugPrint("MultiHeadAttention b_o (first 5): %f %f %f %f %f\n",
               h_bo[0], h_bo[1], h_bo[2], h_bo[3], h_bo[4]);

    // Verify weights are valid before using them
    for (int i = 0; i < 5; i++)
    {
        if (std::isnan(h_Wq[i]) || std::isinf(h_Wq[i]))
        {
            debugPrint("Warning: Invalid weight detected in W_q[%d]: %f\n", i, h_Wq[i]);
            // Initialize to small random value if invalid
            float random_val = (float)(rand()) / RAND_MAX * 0.02f - 0.01f; // Random between -0.01 and 0.01
            cudaMemcpy(&W_q[i], &random_val, sizeof(float), cudaMemcpyHostToDevice);
        }
    }
}

MultiHeadAttention::~MultiHeadAttention()
{
    // Only destroy cuBLAS handle since we don't own the weights anymore
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
    // Validate inputs
    if (!query_input || !key_value_input || !output)
    {
        debugPrint("Error: Null input/output pointers\n");
        return;
    }

    // Validate dimensions
    if (batch_size <= 0 || seq_len <= 0)
    {
        debugPrint("Error: Invalid batch_size=%d or seq_len=%d\n", batch_size, seq_len);
        return;
    }

    // Validate weights
    if (!W_q || !W_k || !W_v || !W_o)
    {
        debugPrint("Error: Uninitialized weights\n");
        return;
    }

    // Add checks for NaN in weights
    float s_check[5];
    cudaMemcpy(s_check, W_q, 5 * sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < 5; i++)
    {
        if (std::isnan(s_check[i]))
        {
            debugPrint("Warning: NaN detected in W_q[%d]\n", i);
        }
    }

    // Debug print inputs
    float h_query[5], h_key[5];
    cudaMemcpy(h_query, query_input, 5 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_key, key_value_input, 5 * sizeof(float), cudaMemcpyDeviceToHost);
    debugPrint("MultiHeadAttention query input (first 5): %f %f %f %f %f\n",
               h_query[0], h_query[1], h_query[2], h_query[3], h_query[4]);
    debugPrint("MultiHeadAttention key input (first 5): %f %f %f %f %f\n",
               h_key[0], h_key[1], h_key[2], h_key[3], h_key[4]);

    // Debug print weights
    float h_Wq[5], h_bq[5];
    cudaMemcpy(h_Wq, W_q, 5 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_bq, b_q, 5 * sizeof(float), cudaMemcpyDeviceToHost);
    debugPrint("MultiHeadAttention Wq (first 5): %f %f %f %f %f\n",
               h_Wq[0], h_Wq[1], h_Wq[2], h_Wq[3], h_Wq[4]);
    debugPrint("MultiHeadAttention bq (first 5): %f %f %f %f %f\n",
               h_bq[0], h_bq[1], h_bq[2], h_bq[3], h_bq[4]);

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
        embed_dim,            // m: rows of output
        batch_size * seq_len, // n: cols of output
        embed_dim,            // k: inner dimension
        &alpha,
        W_q,         // [embed_dim x embed_dim]
        embed_dim,   // leading dimension of W_q
        query_input, // [embed_dim x (batch_size * seq_len)]
        embed_dim,   // leading dimension of query_input
        &beta,
        Q,          // [embed_dim x (batch_size * seq_len)]
        embed_dim); // leading dimension of Q

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

    // Add error checking after matrix multiplications
    float h_check[5];
    cudaMemcpy(h_check, Q, 5 * sizeof(float), cudaMemcpyDeviceToHost);
    debugPrint("Q after projection (first 5): %f %f %f %f %f\n",
               h_check[0], h_check[1], h_check[2], h_check[3], h_check[4]);

    // After computing attention scores
    float h_scores[5];
    cudaMemcpy(h_scores, attention_scores, 5 * sizeof(float), cudaMemcpyDeviceToHost);
    debugPrint("MultiHeadAttention scores (first 5): %f %f %f %f %f\n",
               h_scores[0], h_scores[1], h_scores[2], h_scores[3], h_scores[4]);

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

    // Debug final output
    float h_output[5];
    cudaMemcpy(h_output, output, 5 * sizeof(float), cudaMemcpyDeviceToHost);
    debugPrint("MultiHeadAttention output (first 5): %f %f %f %f %f\n",
               h_output[0], h_output[1], h_output[2], h_output[3], h_output[4]);

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