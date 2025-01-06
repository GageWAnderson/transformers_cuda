#include "utils/weight_init.cuh"

void initializeWeights(curandGenerator_t &gen, float *d_weights, size_t size)
{
    // Initialize weights with a normal distribution (mean=0, stddev=0.02)
    curandGenerateNormal(gen, d_weights, size, 0.0f, 0.0002f);
}
