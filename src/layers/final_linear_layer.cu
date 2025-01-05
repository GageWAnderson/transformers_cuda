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

void FinalLinearLayer::forward(float *d_input, float *d_logits)
{
    // Dimensions for the linear layer
    int m = config_.batch_size * config_.max_seq_len; // Rows of input
    int k = config_.hidden_dim;                       // Shared dimension
    int n = config_.vocab_size;                       // Output dimension

    float alpha = 1.0f;
    float beta = 0.0f;

    // Perform the linear transformation
    cublasSgemm(cublas_,
                CUBLAS_OP_N, // No transpose
                CUBLAS_OP_N, // No transpose
                n,           // Number of rows of d_linear_weights_ (output size)
                m,           // Number of columns of d_input (batch_size * seq_len)
                k,           // Shared dimension (hidden_dim)
                &alpha,
                d_linear_weights_, n, // Weights matrix [n x k]
                d_input, k,           // Input matrix [k x m]
                &beta,
                d_logits, n);         // Output matrix [n x m]

    // Apply softmax to the logits
    applySoftmax(cudnn_, d_logits, d_logits, m, n);

    // Optionally, you can process or log d_logits here

    // Removed internal allocation and deallocation of d_logits
    // No cudaMalloc or cudaFree for d_logits in this function
}
