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
#include "encoder/encoder.cuh"
#include "decoder/decoder.cuh"
#include "cublas_v2.h"
#include "curand.h"
#include "layers/final_linear_layer.cuh"
#include "utils/debug.cuh"
#include "utils/load_weights.cuh"
#include "gpt2_weights.cuh"
#include <fstream>

// Function to display usage instructions
void printUsage();

// Helper function to load configuration
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
        if (arg.find("--layers=") == 0)
        {
            config.num_layers = std::stoi(arg.substr(9));
        }
        else if (arg.find("--hidden_dim=") == 0)
        {
            config.hidden_dim = std::stoi(arg.substr(13));
        }
        else if (arg.find("--heads=") == 0)
        {
            config.num_heads = std::stoi(arg.substr(8));
        }
        else if (arg.find("--weights=") == 0)
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

// Helper function to display initialized parameters
/**
 * @brief Displays model parameters
 * @param config Config object containing parameters
 *
 * Prints initialized model parameters to console.
 */
void displayParameters(const Config &config)
{
    debugPrint("Initializing Transformer model with the following parameters:\n");
    debugPrint("Number of layers: %d\n", config.num_layers);
    debugPrint("Hidden dimension size: %d\n", config.hidden_dim);
    debugPrint("Number of attention heads: %d\n", config.num_heads);
}

// Helper function to load vocabulary
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
 * @param d_token_embeddings Token embedding matrix
 * @param d_positional_encoding Positional encoding matrix
 * @param config Model configuration
 *
 * Runs interactive command line interface for model inference.
 */
void runCLIServer(const std::vector<std::string> &vocabulary, float *d_token_embeddings, float *d_positional_encoding, const Config &config)
{
    std::cout << "Transformer CLI server is running. Type 'exit' to quit.\n";
    std::string input;
    while (true)
    {
        std::cout << "> ";
        std::getline(std::cin, input);

        if (input == "exit")
        {
            break;
        }

        // Tokenize the input text
        std::vector<int> token_ids = tokenize(input, vocabulary);

        // Check for sequence length
        if (token_ids.size() > config.max_seq_len)
        {
            std::cout << "Input exceeds maximum sequence length of " << config.max_seq_len << ". Truncating input.\n";
            token_ids.resize(config.max_seq_len);
        }

        int seq_len = token_ids.size();

        // Get input embeddings
        float *d_input_embeddings = nullptr;
        getInputEmbeddings(token_ids, d_token_embeddings, &d_input_embeddings, config);

        // Sum with positional encodings
        sumEmbeddingsAndPositionalEncoding(d_input_embeddings, d_positional_encoding, seq_len, config.embedding_dim);

        // TODO: Process the combined embeddings using the Transformer model
        // Placeholder for model execution
        std::cout << "Transformer model execution finished for input: " << input << "\n";

        // Free device memory for input embeddings
        cudaFree(d_input_embeddings);
    }
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

    // Display the initialized parameters
    displayParameters(config);

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

    // Create token embeddings
    float *d_token_embeddings = nullptr;
    createTokenEmbeddings(config, &d_token_embeddings);

    // Create positional encodings
    float *d_positional_encoding = nullptr;
    createPositionalEncoding(config.max_seq_len, config.embedding_dim, &d_positional_encoding);

    // Print the positional encoding
    debugPrint("Positional encoding created with dimensions: %d x %d\n",
               config.max_seq_len, config.embedding_dim);

    // Initialize Encoder (if using encoder-decoder architecture)
    Encoder encoder(config);

    // Initialize Decoder
    Decoder decoder(config);

    // Create and initialize the FinalLinearLayer
    FinalLinearLayer final_linear_layer(config, cublas, cudnn, nullptr);
    final_linear_layer.initialize();

    // If weights file was specified, check architecture and try to load it
    if (!weights_file.empty())
    {
        if (config.model_arch == ModelArchitecture::GPT2)
        {
            GPT2Weights* weights = loadGPT2ModelWeights(weights_file);
            if (!weights) {
                std::cerr << "Failed to load weights from: " << weights_file << std::endl;
                return 1;
            }
            debugPrint("Successfully loaded GPT-2 model weights\n");

            // Forward weights to encoder
            encoder.loadWeights(weights);

            // Forward weights to decoder
            decoder.loadWeights(weights);

            // Forward final layer norm weights to final linear layer
            final_linear_layer.loadWeights(weights->getFinalLayerNormWeight(), 
                                         weights->getFinalLayerNormBias());

            delete weights;
        }
        else
        {
            std::cerr << "Error: Model architecture is not supported. Cannot load weights." << std::endl;
            return 1;
        }
    }

    // Allocate memory for encoder input and output
    float *d_encoder_input = nullptr;
    float *d_encoder_output = nullptr;
    size_t input_size = config.max_seq_len * config.hidden_dim * sizeof(float);
    cudaMalloc(&d_encoder_input, input_size);
    cudaMalloc(&d_encoder_output, input_size);

    // Copy the input embeddings to encoder input
    cudaMemcpy(d_encoder_input, d_token_embeddings, input_size, cudaMemcpyDeviceToDevice);

    // Create CUDA stream
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // Run Encoder forward pass
    encoder.forward(d_encoder_output, d_encoder_input, config.batch_size, config.max_seq_len, stream);

    // Synchronize after encoder
    cudaStreamSynchronize(stream);

    // Print intermediate encoder output
    std::vector<float> h_encoder_output(config.max_seq_len * config.hidden_dim);
    cudaMemcpy(h_encoder_output.data(), d_encoder_output, input_size, cudaMemcpyDeviceToHost);
    debugPrint("Encoder output (first 10 elements): ");
    for (int i = 0; i < 10 && i < h_encoder_output.size(); ++i)
    {
        debugPrint("%f ", h_encoder_output[i]);
    }
    debugPrint("\n");

    // Allocate memory for decoder input and output
    float *d_decoder_input = nullptr;
    float *d_decoder_output = nullptr;
    size_t decoder_input_size = config.batch_size * config.hidden_dim * sizeof(float);
    cudaMalloc(&d_decoder_input, decoder_input_size);
    cudaMalloc(&d_decoder_output, decoder_input_size);

    // Initialize generation variables
    std::vector<int> generated_tokens;
    int current_token_id = config.start_token_id; // Assuming start_token_id is defined in config
    int generation_step = 0;

    // Allocate memory for the current token embedding
    float *d_current_token_embedding = nullptr;
    cudaMalloc(&d_current_token_embedding, decoder_input_size);

    debugPrint("\nGenerating tokens:\n");
    int seq_len = 1; // Sequence length is 1 for autoregressive decoding
    while (generation_step < config.max_generation_length)
    {
        // Get the embedding for the current token
        getTokenEmbedding(current_token_id, d_token_embeddings, d_current_token_embedding, config);

        // Prepare decoder input
        cudaMemcpy(d_decoder_input, d_current_token_embedding, decoder_input_size, cudaMemcpyDeviceToDevice);

        // Run Decoder forward pass
        decoder.forward(d_decoder_output,
                        d_decoder_input,
                        d_encoder_output,
                        config.batch_size,
                        seq_len,
                        stream);

        // Print intermediate decoder output
        std::vector<float> h_decoder_output(config.hidden_dim);
        cudaMemcpy(h_decoder_output.data(), d_decoder_output, config.hidden_dim * sizeof(float), cudaMemcpyDeviceToHost);
        debugPrint("Decoder output (first 10 elements): ");
        for (int i = 0; i < 10 && i < h_decoder_output.size(); ++i)
        {
            debugPrint("%f ", h_decoder_output[i]);
        }
        debugPrint("\n");

        // Allocate memory for d_logits outside the forward function
        float *d_logits = nullptr;
        size_t logits_size = config.batch_size * seq_len * config.vocab_size * sizeof(float);
        cudaMalloc(&d_logits, logits_size);

        // Updated function call matches the new signature
        final_linear_layer.forward(d_decoder_output, d_logits, 1); // seq_len = 1 during decoding

        // Copy logits to host to select the next token
        std::vector<float> h_logits(config.batch_size * seq_len * config.vocab_size);
        cudaMemcpy(h_logits.data(), d_logits, logits_size, cudaMemcpyDeviceToHost);

        // Print top 5 intermediate logits
        std::vector<std::pair<float, int>> logits_with_indices;
        for (int i = 0; i < h_logits.size(); ++i)
        {
            logits_with_indices.emplace_back(h_logits[i], i);
        }
        std::partial_sort(logits_with_indices.begin(), logits_with_indices.begin() + 5, logits_with_indices.end(), std::greater<>());

        debugPrint("Top 5 Logits: ");
        for (int i = 0; i < 5; ++i)
        {
            debugPrint("%f (index %d) ", logits_with_indices[i].first, logits_with_indices[i].second);
        }
        debugPrint("\n");

        // Select the next token (e.g., using argmax)
        auto max_iter = std::max_element(h_logits.begin(), h_logits.end());
        int next_token_id = std::distance(h_logits.begin(), max_iter);

        // Print the generated token
        debugPrint("Token %d: %s\n", generation_step + 1, vocabulary[next_token_id].c_str());

        // Append the token to generated sequence
        generated_tokens.push_back(next_token_id);

        // Check for stop token
        if (next_token_id == config.stop_token_id)
        {
            break;
        }

        // Update current token for next iteration
        current_token_id = next_token_id;
        generation_step++;

        // Remember to free d_logits after use
        cudaFree(d_logits);
    }

    // Synchronize and destroy the stream
    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);

    // Cleanup decoder inputs and outputs
    cudaFree(d_decoder_input);
    cudaFree(d_decoder_output);
    cudaFree(d_current_token_embedding);

    // Cleanup encoder inputs and outputs
    cudaFree(d_encoder_input);
    cudaFree(d_encoder_output);

    // Cleanup token embeddings and positional encodings
    cudaFree(d_token_embeddings);
    cudaFree(d_positional_encoding);

    // Destroy cuBLAS handle
    cublasDestroy(cublas);

    // Destroy cuRAND generator
    curandDestroyGenerator(curand_gen);

    // Destroy cuDNN handle
    cudnnDestroy(cudnn);

    // Convert generated token IDs to tokens and output the result
    std::string generated_text = decodeTokens(generated_tokens, vocabulary);
    std::cout << "\nComplete generated text: " << generated_text << std::endl;

    return 0;
}

void printUsage()
{
    std::cout << "Usage: transformer [options]\n";
    std::cout << "Options:\n";
    std::cout << "  --layers=N        Set the number of layers (overrides config file)\n";
    std::cout << "  --hidden_dim=N    Set the hidden dimension size (overrides config file)\n";
    std::cout << "  --heads=N         Set the number of attention heads (overrides config file)\n";
    std::cout << "  --weights=FILE    Load model weights from SafeTensors file\n";
    std::cout << "  --help, -h        Show this help message\n";
}
