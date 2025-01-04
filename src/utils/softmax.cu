#include "utils/softmax.cuh"
#include "utils/utils.cuh"

void applySoftmax(cudnnHandle_t &cudnn, float *d_input, float *d_output, int test_size)
{
    // Create tensor descriptor
    cudnnTensorDescriptor_t input_descriptor;
    checkCUDNN(cudnnCreateTensorDescriptor(&input_descriptor));
    checkCUDNN(cudnnSetTensor4dDescriptor(
        input_descriptor,
        CUDNN_TENSOR_NCHW,
        CUDNN_DATA_FLOAT,
        1, 1, 1, test_size));

    // Apply softmax
    float alpha = 1.0f, beta = 0.0f;
    checkCUDNN(cudnnSoftmaxForward(
        cudnn,
        CUDNN_SOFTMAX_ACCURATE,
        CUDNN_SOFTMAX_MODE_INSTANCE,
        &alpha,
        input_descriptor,
        d_input,
        &beta,
        input_descriptor,
        d_output));

    // Cleanup
    cudnnDestroyTensorDescriptor(input_descriptor);
}
