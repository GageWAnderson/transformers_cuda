#ifndef TOKENIZER_CUH
#define TOKENIZER_CUH

#include <string>
#include <vector>

// Function to tokenize input text into token IDs
std::vector<int> tokenize(const std::string& input, const std::vector<std::string>& vocabulary);

// Function to decode token IDs back into text
std::string decodeTokens(const std::vector<int>& token_ids, const std::vector<std::string>& vocabulary);

#endif // TOKENIZER_CUH 