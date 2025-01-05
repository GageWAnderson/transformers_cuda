#include "embeddings/token_embeddings.cuh"
#include "utils/utils.cuh"
#include <curand_kernel.h>
#include <cuda_runtime.h>

// Kernel to initialize embeddings with random values
__global__ void initializeEmbeddings(float *embeddings, int total_elements, unsigned int seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_elements) {
        // Initialize cuRAND
        curandState_t state;
        curand_init(seed, idx, 0, &state);
        // Initialize embeddings with normal distribution (mean=0, stddev=0.02)
        embeddings[idx] = curand_normal(&state) * 0.02f;
    }
}

// Function to create and initialize token embeddings
void createTokenEmbeddings(const Config &config, float **d_token_embeddings) {
    int vocab_size = config.vocab_size;
    int embedding_dim = config.embedding_dim;
    int total_elements = vocab_size * embedding_dim;

    // Allocate memory on device for embeddings
    checkCUDA(cudaMalloc((void **)d_token_embeddings, total_elements * sizeof(float)));

    // Initialize embeddings with random values
    int threadsPerBlock = 256;
    int blocksPerGrid = (total_elements + threadsPerBlock - 1) / threadsPerBlock;
    unsigned int seed = 1234;
    initializeEmbeddings<<<blocksPerGrid, threadsPerBlock>>>(*d_token_embeddings, total_elements, seed);
    checkCUDA(cudaDeviceSynchronize());
}

// Kernel to gather input embeddings
__global__ void gatherEmbeddingsKernel(const int *d_token_ids, float *d_token_embeddings, float *d_input_embeddings, int seq_len, int embedding_dim, int vocab_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int dim = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx < seq_len && dim < embedding_dim) {
        int token_id = d_token_ids[idx];
        if (token_id < vocab_size) {
            d_input_embeddings[idx * embedding_dim + dim] = d_token_embeddings[token_id * embedding_dim + dim];
        } else {
            // Handle unknown token ID if necessary
            d_input_embeddings[idx * embedding_dim + dim] = 0.0f;
        }
    }
}

void getInputEmbeddings(const std::vector<int> &token_ids, float *d_token_embeddings, float **d_input_embeddings, const Config &config) {
    int seq_len = token_ids.size();
    int embedding_dim = config.embedding_dim;
    int vocab_size = config.vocab_size;

    // Allocate memory for token IDs on the device
    int *d_token_ids = nullptr;
    cudaMalloc((void**)&d_token_ids, seq_len * sizeof(int));
    cudaMemcpy(d_token_ids, token_ids.data(), seq_len * sizeof(int), cudaMemcpyHostToDevice);

    // Allocate memory for input embeddings on the device
    size_t embeddings_size = seq_len * embedding_dim * sizeof(float);
    cudaMalloc((void**)d_input_embeddings, embeddings_size);

    // Define grid and block dimensions
    dim3 blockDim(16, 16);
    dim3 gridDim((seq_len + blockDim.x - 1) / blockDim.x,
                 (embedding_dim + blockDim.y - 1) / blockDim.y);

    // Launch kernel to gather embeddings
    gatherEmbeddingsKernel<<<gridDim, blockDim>>>(d_token_ids, d_token_embeddings, *d_input_embeddings, seq_len, embedding_dim, vocab_size);

    // Synchronize to ensure completion
    cudaDeviceSynchronize();

    // Free device memory for token IDs
    cudaFree(d_token_ids);
}

// Function to get the embedding for a specific token ID
void getTokenEmbedding(int token_id, float *d_token_embeddings, float *d_output_embedding, const Config &config)
{
    size_t embedding_size = config.embedding_dim * sizeof(float);
    size_t offset = static_cast<size_t>(token_id) * config.embedding_dim;

    // Copy the embedding vector for the token ID from device to device memory
    cudaMemcpy(d_output_embedding, d_token_embeddings + offset, embedding_size, cudaMemcpyDeviceToDevice);
}
