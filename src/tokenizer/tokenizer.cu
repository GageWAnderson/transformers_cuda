#include "tokenizer/tokenizer.cuh"
#include <sstream>
#include <algorithm>
#include <iostream>

// TODO: Investigate the best vocab for the tokenizer
// This is normally learned with the model, see if this is possible to train here

/**
 * @brief Splits input text into individual tokens
 * @param input String to be tokenized
 * @return Vector of string tokens
 * 
 * Helper function that splits input text on whitespace into individual tokens.
 */
std::vector<std::string> tokenizeInput(const std::string& input) {
    std::vector<std::string> tokens;
    std::istringstream stream(input);
    std::string token;

    while (stream >> token) {
        tokens.push_back(token);
    }

    return tokens;
}

/**
 * @brief Converts input text into token IDs using vocabulary
 * @param input Input text to tokenize
 * @param vocabulary Vector of vocabulary tokens
 * @return Vector of token IDs
 * 
 * Tokenizes input text by finding longest matching tokens in vocabulary.
 * Uses greedy longest-match-first approach. Handles unknown tokens by warning
 * and skipping.
 */
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

/**
 * @brief Converts token IDs back into text
 * @param token_ids Vector of token IDs to decode
 * @param vocabulary Vector of vocabulary tokens
 * @return Decoded text string
 * 
 * Converts sequence of token IDs back into text by looking up each ID
 * in vocabulary. Handles unknown tokens with <UNK> placeholder.
 */
std::string decodeTokens(const std::vector<int>& token_ids, const std::vector<std::string>& vocabulary) {
    std::string decoded_text;
    for (int token_id : token_ids) {
        if (token_id >= 0 && token_id < static_cast<int>(vocabulary.size())) {
            decoded_text += vocabulary[token_id] + " ";
        } else {
            decoded_text += "<UNK> ";
        }
    }
    // Remove the trailing space if necessary
    if (!decoded_text.empty()) {
        decoded_text.pop_back();
    }
    return decoded_text;
}
