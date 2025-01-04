#ifndef POSITIONAL_ENCODING_CUH
#define POSITIONAL_ENCODING_CUH

void createPositionalEncoding(int max_seq_len, int embedding_dim, float **d_positional_encoding);
void sumEmbeddingsAndPositionalEncoding(float *d_input_embeddings, float *d_positional_encoding, int seq_len, int embedding_dim);

#endif // POSITIONAL_ENCODING_CUH
