#include <iostream>
#include <algorithm>
#include <string>
#include <vector>
#include "config.cuh"
#include "cuda_runtime.h"
#include "cudnn.h"
#include "utils/utils.cuh"
#include "utils/softmax.cuh"
#include "embeddings/token_embeddings.cuh"
#include "embeddings/positional_encoding.cuh"
#include "tokenizer/vocab.cuh"
#include "tokenizer/tokenizer.cuh"
#include "decoder/decoder.cuh"
#include "cublas_v2.h"
#include "curand.h"
#include "layers/final_linear_layer.cuh"
#include "utils/debug.cuh"
#include "utils/load_weights.cuh"
#include "gpt2_weights.cuh"
#include <fstream>

void printUsage();

/**
 * @brief Prints the generated tokens, rendering \u0120 as a space
 * @param tokens Vector of token IDs
 * @param vocabulary Model vocabulary
 *
 * Iterates through the generated tokens and prints them, replacing \u0120 with a space.
 */
void printGeneratedTokens(const std::vector<int> &tokens, const std::vector<std::string> &vocabulary)
{
    for (int token_id : tokens)
    {
        std::string token = vocabulary[token_id];
        size_t pos = 0;
        while ((pos = token.find("\u0120", pos)) != std::string::npos)
        {
            token.replace(pos, 1, " ");
        }
        std::cout << token;
    }
    std::cout << std::endl; // New line after generation
}

/**
 * @brief Loads configuration from file
 * @param config Reference to Config object
 * @return true if loaded successfully, false otherwise
 *
 * Attempts to load configuration from config.ini file.
 * Falls back to default values if load fails.
 */
bool loadConfiguration(Config &config)
{
    if (!config.loadFromFile("config/config.ini"))
    {
        debugPrint("Proceeding with default configuration values.\n");
        return false;
    }
    return true;
}

// Helper function to parse command-line arguments
/**
 * @brief Parses command line arguments
 * @param argc Argument count
 * @param argv Argument values
 * @param config Reference to Config object
 * @return true if parsed successfully, false if help requested or error
 *
 * Processes command line arguments to override config values.
 */
bool parseArguments(int argc, char *argv[], Config &config, std::string &weights_file)
{
    for (int i = 1; i < argc; ++i)
    {
        std::string arg = argv[i];
        if (arg.find("--weights=") == 0)
        {
            weights_file = arg.substr(10);
        }
        else if (arg == "--help" || arg == "-h")
        {
            printUsage();
            return false;
        }
        else
        {
            std::cerr << "Unknown argument: " << arg << std::endl;
            printUsage();
            return false;
        }
    }

    return true;
}

/**
 * @brief Loads and displays vocabulary
 * @param vocab_file Path to vocabulary file
 * @param vocabulary Vector to store vocabulary
 *
 * Loads vocabulary from file and prints size information.
 */
void loadAndDisplayVocabulary(const std::string &vocab_file, std::vector<std::string> &vocabulary)
{
    loadVocabulary(vocab_file, vocabulary);
    debugPrint("Loaded vocabulary with %zu tokens.\n", vocabulary.size());
}

// Helper function to initialize cuDNN
/**
 * @brief Initializes cuDNN library
 * @return Initialized cuDNN handle
 *
 * Creates and returns cuDNN handle for use with neural network operations.
 */
cudnnHandle_t initializeCUDNN()
{
    cudnnHandle_t cudnn;
    checkCUDNN(cudnnCreate(&cudnn));
    return cudnn;
}

// Helper function to run CLI server loop
/**
 * @brief Runs interactive CLI server
 * @param vocabulary Model vocabulary
 * @param wte_layer Token embedding layer
 * @param wpe_layer Positional encoding layer
 * @param config Model configuration
 * @param decoder Decoder object
 * @param final_linear_layer FinalLinearLayer object
 * @param cudnn cuDNN handle
 * @param cublas cuBLAS handle
 *
 * Runs interactive command line interface for model inference.
 */
void runCLIServer(
    const std::vector<std::string> &vocabulary,
    WTELayer &wte_layer,
    WPELayer &wpe_layer,
    const GPT2Weights *weights,
    const Config &config,
    Decoder &decoder,
    FinalLinearLayer &final_linear_layer,
    cudnnHandle_t cudnn,
    cublasHandle_t cublas)
{
    std::cout << "Transformer CLI server is running. Type 'exit' to quit.\n";

    // Allocate memory for the full sequence capacity upfront
    float *d_sequence_embeddings = nullptr;
    size_t max_sequence_size = config.max_generation_length * config.batch_size * config.hidden_dim * sizeof(float);
    cudaMalloc(&d_sequence_embeddings, max_sequence_size);

    // Allocate memory for decoder output (same size as sequence for safety)
    float *d_decoder_output = nullptr;
    cudaMalloc(&d_decoder_output, max_sequence_size);

    // Create CUDA stream
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    std::string input;
    while (true)
    {
        std::cout << "\n> ";
        std::getline(std::cin, input);

        if (input == "exit")
        {
            break;
        }

        // Reset sequence embeddings buffer for new input
        cudaMemset(d_sequence_embeddings, 0, max_sequence_size);
        cudaMemset(d_decoder_output, 0, max_sequence_size);

        // Reset generation variables for new input
        std::vector<int> generated_tokens;
        int current_token_id = config.start_token_id;
        int generation_step = 0;
        int current_seq_len = 1;

        debugPrint("\nGenerating tokens for input: %s\n", input.c_str());

        while (generation_step < config.max_generation_length)
        {
            // Get the embedding for the current token
            std::vector<int> current_tokens(1, current_token_id);

            wte_layer.forward(
                current_tokens,
                d_sequence_embeddings + (current_seq_len - 1) * config.hidden_dim,
                config.batch_size,
                1,
                stream);

            wpe_layer.forward(
                d_sequence_embeddings + (current_seq_len - 1) * config.hidden_dim,
                1, // we only want to embed the newly appended token
                config.batch_size,
                stream);

            // Run decoder with current sequence length
            decoder.forward(
                d_decoder_output,
                d_sequence_embeddings,
                nullptr,
                config.batch_size,
                current_seq_len,
                stream);

            // Get logits for last token position only
            float *d_last_hidden = d_decoder_output + (current_seq_len - 1) * config.hidden_dim;

            // Allocate memory for logits of last token
            float *d_logits = nullptr;
            size_t logits_size = config.batch_size * config.vocab_size * sizeof(float);
            cudaMalloc(&d_logits, logits_size);

            // Run final linear layer for last token
            final_linear_layer.forward(d_last_hidden, d_logits, 1, weights->getTokenEmbedding());

            // Copy logits to host
            std::vector<float> h_logits(config.batch_size * config.vocab_size);
            cudaMemcpy(h_logits.data(), d_logits, logits_size, cudaMemcpyDeviceToHost);

            // Select next token
            auto max_iter = std::max_element(h_logits.begin(), h_logits.end());
            int next_token_id = std::distance(h_logits.begin(), max_iter);

            // Append the token
            generated_tokens.push_back(next_token_id);

            // Check for stop token
            if (next_token_id == config.stop_token_id)
            {
                break;
            }

            // Update for next iteration
            current_token_id = next_token_id;
            current_seq_len++;
            generation_step++;

            cudaFree(d_logits);
        }

        // Print the generated tokens all at once at the end of the generation
        printGeneratedTokens(generated_tokens, vocabulary);
    }

    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);
    cudaFree(d_sequence_embeddings);
    cudaFree(d_decoder_output);
}

int main(int argc, char *argv[])
{
    Config config;

    // Load configurations from the config file
    loadConfiguration(config);

    // Parse command-line arguments and load weights
    std::string weights_file;
    if (!parseArguments(argc, argv, config, weights_file))
    {
        return 1;
    }

    // Load the vocabulary using the path from the config
    std::vector<std::string> vocabulary;
    loadAndDisplayVocabulary(config.vocab_file, vocabulary);

    // Initialize cuDNN
    cudnnHandle_t cudnn = initializeCUDNN();

    // Create cuBLAS handle
    cublasHandle_t cublas;
    cublasCreate(&cublas);

    // Create cuRAND generator
    curandGenerator_t curand_gen;
    curandCreateGenerator(&curand_gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(curand_gen, 1234ULL);

    // If weights file was specified, check architecture and try to load it
    GPT2Weights *weights = nullptr;
    if (!weights_file.empty())
    {
        if (config.model_arch == ModelArchitecture::GPT2)
        {
            weights = loadGPT2ModelWeights(weights_file);
            if (!weights)
            {
                std::cerr << "Failed to load weights from: " << weights_file << std::endl;
                return 1;
            }

            // Add validation here
            try
            {
                validate_weights(weights, config);
                debugPrint("Successfully loaded and validated GPT-2 model weights\n");
            }
            catch (const std::exception &e)
            {
                std::cerr << "Weight validation failed: " << e.what() << std::endl;
                delete weights;
                return 1;
            }
        }
        else
        {
            std::cerr << "Error: Model architecture is not supported. Cannot load weights." << std::endl;
            return 1;
        }
    }

    // Create WTE layer for token embeddings
    WTELayer wte_layer(weights);

    // Create WPE layer for position embeddings
    WPELayer wpe_layer(weights);

    debugPrint("Weights loaded successfully, loading decoder\n");
    // Initialize Decoder with weights
    Decoder decoder(config, weights);

    // Create and initialize the FinalLinearLayer with weights
    debugPrint("Initializing FinalLinearLayer\n");
    FinalLinearLayer final_linear_layer(config, cublas, cudnn, weights);

    // Run the CLI server with all necessary components
    runCLIServer(vocabulary,
                 wte_layer,
                 wpe_layer,
                 weights,
                 config,
                 decoder,
                 final_linear_layer,
                 cudnn,
                 cublas);

    // Destroy cuBLAS handle
    cublasDestroy(cublas);

    // Destroy cuRAND generator
    curandDestroyGenerator(curand_gen);

    // Destroy cuDNN handle
    cudnnDestroy(cudnn);

    return 0;
}

void printUsage()
{
    std::cout << "Usage: transformer [options]\n";
    std::cout << "Options:\n";
    std::cout << "  --weights=FILE    Load model weights from SafeTensors file\n";
    std::cout << "  --help, -h        Show this help message\n";
}
