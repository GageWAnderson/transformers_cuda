#include "layers/final_linear_layer.cuh"
#include "utils/weight_init.cuh"
#include "utils/softmax.cuh"
#include "utils/utils.cuh"

FinalLinearLayer::FinalLinearLayer(const Config &config,
                                   cublasHandle_t &cublas_handle,
                                   cudnnHandle_t &cudnn_handle,
                                   curandGenerator_t &curand_gen)
    : config_(config), cublas_(cublas_handle), cudnn_(cudnn_handle), curand_gen_(curand_gen)
{
}

FinalLinearLayer::~FinalLinearLayer()
{
    freeWeights();
}

void FinalLinearLayer::initialize()
{
    allocateWeights();

    // Initialize weights with random values
    size_t weight_size = config_.hidden_dim * config_.vocab_size;
    initializeWeights(curand_gen_, d_linear_weights_, weight_size);
}

void FinalLinearLayer::allocateWeights()
{
    size_t weights_size = config_.hidden_dim * config_.vocab_size * sizeof(float);
    cudaMalloc(&d_linear_weights_, weights_size);
}

void FinalLinearLayer::freeWeights()
{
    if (d_linear_weights_)
    {
        cudaFree(d_linear_weights_);
        d_linear_weights_ = nullptr;
    }
}

void FinalLinearLayer::forward(float *d_input)
{
    // Dimensions for the linear layer
    int m = config_.batch_size * config_.max_seq_len; // Rows of input
    int k = config_.hidden_dim;                       // Shared dimension
    int n = config_.vocab_size;                       // Output dimension

    // Allocate memory for the output of the linear layer (logits)
    float *d_logits = nullptr;
    size_t logits_size = m * n * sizeof(float);
    cudaMalloc(&d_logits, logits_size);

    float alpha = 1.0f;
    float beta = 0.0f;

    // Perform the linear transformation
    cublasSgemm(cublas_,
                CUBLAS_OP_N, // No transpose
                CUBLAS_OP_N, // No transpose
                n,           // Columns of d_linear_weights_ (output size)
                m,           // Rows of d_input (batch_size * seq_len)
                k,           // Shared dimension (hidden_dim)
                &alpha,
                d_linear_weights_, n, // Weights matrix [n x k]
                d_input, k,           // Input matrix [k x m]
                &beta,
                d_logits, n);         // Output matrix [n x m]

    // Apply softmax to the logits
    applySoftmax(cudnn_, d_logits, d_logits, m, n);

    // Now, d_logits contains the probabilities for each token

    // For testing purposes, retrieve and print the probabilities of the first token
    size_t token_probs_size = n * sizeof(float);
    float *h_probs = new float[n];
    cudaMemcpy(h_probs, d_logits, token_probs_size, cudaMemcpyDeviceToHost);

    std::cout << "Output probabilities for the first token:" << std::endl;
    for (int i = 0; i < n; ++i)
    {
        std::cout << h_probs[i] << " ";
    }
    std::cout << std::endl;

    delete[] h_probs;

    // Cleanup
    cudaFree(d_logits);
}
