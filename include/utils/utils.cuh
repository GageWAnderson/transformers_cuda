#ifndef UTILS_CUH
#define UTILS_CUH

#include <iostream>
#include "cuda_runtime.h"
#include "cudnn.h"
#include "utils/debug.cuh"

// Function to check CUDA errors
#define checkCUDA(expression)                               \
    {                                                       \
        cudaError_t status = (expression);                  \
        if (status != cudaSuccess)                          \
        {                                                   \
            std::cerr << "CUDA Error on line " << __LINE__  \
                      << ": " << cudaGetErrorString(status) \
                      << std::endl;                         \
            exit(1);                                        \
        }                                                   \
    }

// Function to check cuDNN errors
#define checkCUDNN(expression)                               \
    {                                                        \
        cudnnStatus_t status = (expression);                 \
        if (status != CUDNN_STATUS_SUCCESS)                  \
        {                                                    \
            std::cerr << "cuDNN Error on line " << __LINE__  \
                      << ": " << cudnnGetErrorString(status) \
                      << std::endl;                          \
            exit(1);                                         \
        }                                                    \
    }

#define CUDA_CHECK(call)                                                                                      \
    do                                                                                                        \
    {                                                                                                         \
        cudaError_t error = call;                                                                             \
        if (error != cudaSuccess)                                                                             \
        {                                                                                                     \
            debugPrint("CUDA error %d: %s at %s:%d\n", error, cudaGetErrorString(error), __FILE__, __LINE__); \
            throw std::runtime_error("CUDA error");                                                           \
        }                                                                                                     \
    } while (0)

// Add the declaration of add_tensors function
void add_tensors(const float* a, const float* b, float* c, int size, cudaStream_t stream);

#endif // UTILS_CUH 