#ifndef LAYER_NORM_CUH
#define LAYER_NORM_CUH

#include <cudnn.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

class LayerNorm
{
public:
    LayerNorm(int hidden_dim, float* gamma_weights, float* beta_weights);
    ~LayerNorm();

    void forward(float *output, const float *input, int seq_len, cudaStream_t stream);

private:
    int hidden_dim;
    cudnnHandle_t cudnn_handle;
    cudnnTensorDescriptor_t x_desc;
    cudnnTensorDescriptor_t gamma_beta_desc;
    float *gamma;
    float *beta;
    cublasHandle_t cublas_handle;
};

#endif // LAYER_NORM_CUH