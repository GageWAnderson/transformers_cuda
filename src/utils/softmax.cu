#include "utils/softmax.cuh"
#include "utils/utils.cuh"

void applySoftmax(cudnnHandle_t &cudnn, float *d_input, float *d_output, int batch_size, int num_classes)
{
    // Create tensor descriptor
    cudnnTensorDescriptor_t tensorDesc;
    cudnnCreateTensorDescriptor(&tensorDesc);
    cudnnSetTensor4dDescriptor(tensorDesc,
                               CUDNN_TENSOR_NCHW,
                               CUDNN_DATA_FLOAT,
                               batch_size,   // N
                               num_classes,  // C
                               1,            // H
                               1);           // W

    // Apply softmax
    const float alpha = 1.0f;
    const float beta = 0.0f;

    cudnnSoftmaxForward(cudnn,
                        CUDNN_SOFTMAX_ACCURATE,
                        CUDNN_SOFTMAX_MODE_CHANNEL,
                        &alpha,
                        tensorDesc,
                        d_input,
                        &beta,
                        tensorDesc,
                        d_output);

    // Destroy tensor descriptor
    cudnnDestroyTensorDescriptor(tensorDesc);
}
