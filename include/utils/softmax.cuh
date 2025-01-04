#ifndef SOFTMAX_CUH
#define SOFTMAX_CUH

#include "cudnn.h"

// Helper function to handle softmax logic
void applySoftmax(cudnnHandle_t &cudnn, float *d_input, float *d_output, int test_size);

#endif // SOFTMAX_CUH
