#ifndef TOKEN_EMBEDDINGS_CUH
#define TOKEN_EMBEDDINGS_CUH

#include <vector>
#include "config.cuh"

void createTokenEmbeddings(const Config &config, float **d_token_embeddings);
void getInputEmbeddings(const std::vector<int> &token_ids, float *d_token_embeddings, float **d_input_embeddings, const Config &config);

#endif // TOKEN_EMBEDDINGS_CUH 