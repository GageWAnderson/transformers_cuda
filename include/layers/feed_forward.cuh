#ifndef FEED_FORWARD_H
#define FEED_FORWARD_H

#include <cuda_runtime.h>

class FeedForward
{
private:
    int hidden_dim;
    int intermediate_dim;

    // Weights and biases
    float *d_W1; // Weight matrix for the first linear layer
    float *d_b1; // Bias vector for the first linear layer
    float *d_W2; // Weight matrix for the second linear layer
    float *d_b2; // Bias vector for the second linear layer

public:
    FeedForward(int hidden_dim, int intermediate_dim, float *W1_ptr, float *b1_ptr,
                float *W2_ptr, float *b2_ptr);
    ~FeedForward();

    void forward(float *output, const float *input, int seq_len, cudaStream_t stream);

    void setWeight1(float* weight) { d_W1 = weight; }
    void setBias1(float* bias) { d_b1 = bias; }
    void setWeight2(float* weight) { d_W2 = weight; }
    void setBias2(float* bias) { d_b2 = bias; }
};

#endif // FEED_FORWARD_H
