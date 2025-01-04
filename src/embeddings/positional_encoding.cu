#include "embeddings/positional_encoding.cuh"
#include <cuda_runtime.h>
#include <cmath>

// Kernel to compute positional encodings
__global__ void computePositionalEncodingKernel(float* d_pos_encoding, int max_seq_len, int embedding_dim) {
    int pos = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    if (pos < max_seq_len && i < embedding_dim) {
        float angle_rate = 1.0f / powf(10000.0f, (2.0f * (i / 2)) / embedding_dim);
        float angle = pos * angle_rate;

        if (i % 2 == 0) {
            d_pos_encoding[pos * embedding_dim + i] = sinf(angle);
        } else {
            d_pos_encoding[pos * embedding_dim + i] = cosf(angle);
        }
    }
}

void createPositionalEncoding(int max_seq_len, int embedding_dim, float **d_positional_encoding) {
    size_t size = max_seq_len * embedding_dim * sizeof(float);

    // Allocate memory on the device
    cudaMalloc((void**)d_positional_encoding, size);

    // Define grid and block dimensions
    dim3 blockDim(16, 16);
    dim3 gridDim((max_seq_len + blockDim.x - 1) / blockDim.x,
                 (embedding_dim + blockDim.y - 1) / blockDim.y);

    // Launch kernel to compute positional encodings
    computePositionalEncodingKernel<<<gridDim, blockDim>>>(*d_positional_encoding, max_seq_len, embedding_dim);

    // Synchronize to ensure completion
    cudaDeviceSynchronize();
}

// Kernel to sum embeddings with positional encodings
__global__ void sumEmbeddingsKernel(float *d_input_embeddings, float *d_positional_encoding, int seq_len, int embedding_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int dim = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx < seq_len && dim < embedding_dim) {
        d_input_embeddings[idx * embedding_dim + dim] += d_positional_encoding[idx * embedding_dim + dim];
    }
}

void sumEmbeddingsAndPositionalEncoding(float *d_input_embeddings, float *d_positional_encoding, int seq_len, int embedding_dim) {
    // Define grid and block dimensions
    dim3 blockDim(16, 16);
    dim3 gridDim((seq_len + blockDim.x - 1) / blockDim.x,
                 (embedding_dim + blockDim.y - 1) / blockDim.y);

    // Launch kernel to sum embeddings
    sumEmbeddingsKernel<<<gridDim, blockDim>>>(d_input_embeddings, d_positional_encoding, seq_len, embedding_dim);

    // Synchronize to ensure completion
    cudaDeviceSynchronize();
}
