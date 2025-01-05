#ifndef WEIGHT_INIT_CUH
#define WEIGHT_INIT_CUH

#include "curand.h"

// Function to initialize weights with a normal distribution
void initializeWeights(curandGenerator_t &gen, float *d_weights, size_t size);

#endif // WEIGHT_INIT_CUH 