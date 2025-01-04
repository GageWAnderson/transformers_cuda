#ifndef UTILS_CUH
#define UTILS_CUH

#include <iostream>
#include "cuda_runtime.h"
#include "cudnn.h"

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

#endif // UTILS_CUH 