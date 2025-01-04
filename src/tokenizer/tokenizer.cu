#include "tokenizer.cuh"
#include <sstream>
#include <algorithm>
#include <iostream>

// TODO: Investigate the best vocab for the tokenizer
// This is normally learned with the model, see if this is possible to train here

// Helper function to split input text into tokens
std::vector<std::string> tokenizeInput(const std::string& input) {
    std::vector<std::string> tokens;
    std::istringstream stream(input);
    std::string token;

    while (stream >> token) {
        tokens.push_back(token);
    }

    return tokens;
}

std::vector<int> tokenize(const std::string& input, const std::vector<std::string>& vocabulary) {
    std::vector<int> token_ids;
    std::string current_text = input;
    
    while (!current_text.empty()) {
        // Find the longest matching token in vocabulary
        size_t longest_match = 0;
        int best_token_id = -1;
        
        for (size_t i = 0; i < vocabulary.size(); i++) {
            const std::string& token = vocabulary[i];
            if (current_text.compare(0, token.length(), token) == 0 && token.length() > longest_match) {
                longest_match = token.length();
                best_token_id = i;
            }
        }
        
        if (best_token_id >= 0) {
            token_ids.push_back(best_token_id);
            current_text = current_text.substr(longest_match);
        } else {
            // Handle unknown characters - typically split into bytes or use special unknown token
            std::cerr << "Warning: Unknown character at: " << current_text[0] << '\n';
            current_text = current_text.substr(1);
        }
    }

    return token_ids;
}
