#ifndef FEED_FORWARD_H
#define FEED_FORWARD_H

#include <cuda_runtime.h>
#include <cublas_v2.h>

class FeedForward
{
private:
    int hidden_dim;
    int intermediate_dim;

    // Weight and bias references
    const float *d_W1; // Weight matrix for the first linear layer
    const float *d_b1; // Bias vector for the first linear layer
    const float *d_W2; // Weight matrix for the second linear layer
    const float *d_b2; // Bias vector for the second linear layer

public:
    FeedForward(int hidden_dim, int intermediate_dim,
                const float *W1, const float *b1,
                const float *W2, const float *b2);
    ~FeedForward();

    void forward(float *output, const float *input, int seq_len, cudaStream_t stream);
    void reset(cudaStream_t stream, cublasHandle_t cublas_handle)
    {
        cublasSetStream(cublas_handle, stream);
    }
};

#endif // FEED_FORWARD_H
