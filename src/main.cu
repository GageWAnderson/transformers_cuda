#include <iostream>
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

// Function to display usage instructions
void printUsage();

// Helper function to load configuration
bool loadConfiguration(Config &config)
{
    if (!config.loadFromFile("config/config.ini"))
    {
        std::cout << "Proceeding with default configuration values.\n";
        return false;
    }
    return true;
}

// Helper function to parse command-line arguments
bool parseArguments(int argc, char *argv[], Config &config)
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
void displayParameters(const Config &config)
{
    std::cout << "Initializing Transformer model with the following parameters:\n";
    std::cout << "Number of layers: " << config.num_layers << "\n";
    std::cout << "Hidden dimension size: " << config.hidden_dim << "\n";
    std::cout << "Number of attention heads: " << config.num_heads << "\n";
}

// Helper function to load vocabulary
void loadAndDisplayVocabulary(const std::string &vocab_file, std::vector<std::string> &vocabulary)
{
    loadVocabulary(vocab_file, vocabulary);
    std::cout << "Loaded vocabulary with " << vocabulary.size() << " tokens.\n";
}

// Helper function to initialize cuDNN
cudnnHandle_t initializeCUDNN()
{
    cudnnHandle_t cudnn;
    checkCUDNN(cudnnCreate(&cudnn));
    return cudnn;
}

// Helper function to run CLI server loop
void runCLIServer(const std::vector<std::string> &vocabulary, float *d_token_embeddings, float *d_positional_encoding, const Config &config);

int main(int argc, char *argv[])
{
    Config config;

    // Load configurations from the config file
    loadConfiguration(config);

    // Parse command-line arguments to override config values
    if (!parseArguments(argc, argv, config))
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

    // Create token embeddings
    float *d_token_embeddings = nullptr;
    createTokenEmbeddings(config, &d_token_embeddings);

    // Create positional encodings
    float *d_positional_encoding = nullptr;
    createPositionalEncoding(config.max_seq_len, config.embedding_dim, &d_positional_encoding);

    // Print the positional encoding
    std::cout << "Positional encoding created with dimensions: "
              << config.max_seq_len << " x " << config.embedding_dim << "\n";

    // Initialize Encoder
    Encoder encoder(config);

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
    encoder.forward(d_encoder_output, d_encoder_input, config.max_seq_len, stream);

    // Synchronize and destroy stream
    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);

    // Do something with d_encoder_output
    // For now, just print a message
    std::cout << "Encoder forward pass completed.\n";

    // Cleanup
    cudaFree(d_encoder_input);
    cudaFree(d_encoder_output);

    // Cleanup
    cudnnDestroy(cudnn);
    cudaFree(d_token_embeddings);
    cudaFree(d_positional_encoding);

    return 0;
}

void printUsage()
{
    std::cout << "Usage: transformer [options]\n";
    std::cout << "Options:\n";
    std::cout << "  --layers=N        Set the number of layers (overrides config file)\n";
    std::cout << "  --hidden_dim=N    Set the hidden dimension size (overrides config file)\n";
    std::cout << "  --heads=N         Set the number of attention heads (overrides config file)\n";
    std::cout << "  --help, -h        Show this help message\n";
}

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
