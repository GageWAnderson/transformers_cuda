#ifndef FINAL_LINEAR_LAYER_CUH
#define FINAL_LINEAR_LAYER_CUH

#include "cuda_runtime.h"
#include "cublas_v2.h"
#include "cudnn.h"
#include "curand.h"
#include "config.cuh"
#include "gpt2_weights.cuh"

class FinalLinearLayer
{
public:
    FinalLinearLayer(const Config &config,
                     cublasHandle_t &cublas_handle,
                     cudnnHandle_t &cudnn_handle,
                     const GPT2Weights* weights);

    ~FinalLinearLayer();

    void forward(float *d_input, float *d_logits, int seq_len);

private:
    const Config &config_;
    cublasHandle_t &cublas_;
    cudnnHandle_t &cudnn_;

    float *d_linear_weights_ = nullptr;
    float *d_linear_bias_ = nullptr;

    void allocateWeights();
    void freeWeights();
};

#endif // FINAL_LINEAR_LAYER_CUH
