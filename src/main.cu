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
// #include "layers/final_linear_layer.cuh"
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
 * @brief CUDA kernel for linear transformation
 * @param input Input tensor
 * @param weights Weight matrix
 * @param output Output tensor
 * @param vocab_size Size of vocabulary
 * @param batch_seq_len Combined batch and sequence length
 * @param hidden_dim Hidden dimension size
 *
 * Performs matrix multiplication between input and weights to produce logits.
 * Each thread computes one element of the output matrix.
 */
__global__ void linearTransformKernel(const float *input, const float *weights, float *output,
                                      int vocab_size, int batch_seq_len, int hidden_dim)
{
    // Calculate global thread indices
    int row = blockIdx.x * blockDim.x + threadIdx.x; // For vocab_size dimension
    int col = blockIdx.y * blockDim.y + threadIdx.y; // For batch_seq_len dimension

    if (row < vocab_size && col < batch_seq_len)
    {
        float sum = 0.0f;
// Use FP32 accumulator for better numerical stability
#pragma unroll
        for (int k = 0; k < hidden_dim; k++)
        {
            // Ensure proper memory access pattern
            float input_val = input[col * hidden_dim + k];
            float weight_val = weights[row * hidden_dim + k];
            sum = fmaf(weight_val, input_val, sum); // Use fmaf for better precision
        }

        output[col * vocab_size + row] = sum;
    }
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

// Helper function to apply temperature to logits
/**
 * @brief Applies temperature to logits
 * @param logits Vector of logits
 * @param vocab_size Size of vocabulary
 * @param temperature Temperature value
 *
 * Adjusts logits by dividing by the temperature.
 */
void applyTemperature(std::vector<float> &logits, int vocab_size, float temperature)
{
    for (int i = 0; i < vocab_size; i++)
    {
        logits[i] /= temperature;
    }
}

/**
 * @brief Selects the next token based on temperature-adjusted logits
 * @param logits Vector of logits
 * @param vocab_size Size of vocabulary
 * @param temperature Temperature value
 * @return Selected token ID
 *
 * Applies temperature to logits, converts to probabilities, and samples the next token.
 */
int selectNextToken(const std::vector<float> &logits, int vocab_size, float temperature)
{
    std::vector<float> adjusted_logits = logits;

    // Apply temperature to logits
    applyTemperature(adjusted_logits, vocab_size, temperature);

    // Convert to probabilities
    float sum = 0.0f;
    for (int i = 0; i < vocab_size; i++)
    {
        adjusted_logits[i] = exp(adjusted_logits[i]);
        sum += adjusted_logits[i];
    }
    for (int i = 0; i < vocab_size; i++)
    {
        adjusted_logits[i] /= sum;
    }

    // Sample from the distribution
    float r = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    float cdf = 0.0f;
    int next_token_id = 0;
    for (int i = 0; i < vocab_size; i++)
    {
        cdf += adjusted_logits[i];
        if (r < cdf)
        {
            next_token_id = i;
            break;
        }
    }

    return next_token_id;
}

// Helper function to run CLI server loop
/**
 * @brief Runs interactive CLI server
 * @param vocabulary Model vocabulary
 * @param wte_layer Token embedding layer
 * @param wpe_layer Positional encoding layer
 * @param config Model configuration
 * @param decoder Decoder object
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
    cudnnHandle_t cudnn,
    cublasHandle_t cublas)
{
    std::cout << "Transformer CLI server is running. Type 'exit' to quit.\n";

    // Initialize pointers to nullptr
    float *d_sequence_embeddings = nullptr;
    float *d_decoder_output = nullptr;
    float *d_logits = nullptr;
    size_t current_buffer_size = 0;

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

        // Tokenize the input text
        std::vector<int> input_tokens = tokenize(input, vocabulary);
        std::vector<int> context_window;

        if (config.start_token_id >= 0)
        {
            context_window.push_back(config.start_token_id);
        }

        context_window.insert(context_window.end(), input_tokens.begin(), input_tokens.end());

        int generation_step = 0;
        int current_seq_len = context_window.size();
        std::vector<int> generated_tokens;

        while (generation_step < config.max_generation_length)
        {
            // Calculate required buffer size for current sequence length
            size_t required_size = current_seq_len * config.hidden_dim * sizeof(float);
            size_t logits_size = config.batch_size * config.vocab_size * sizeof(float);

            // Reallocate buffers if needed
            if (required_size > current_buffer_size)
            {
                // Free existing buffers if they exist
                if (d_sequence_embeddings)
                    cudaFree(d_sequence_embeddings);
                if (d_decoder_output)
                    cudaFree(d_decoder_output);
                if (d_logits)
                    cudaFree(d_logits);

                // Allocate new buffers with extra padding to reduce reallocations
                size_t padded_size = required_size * 2; // Double the size for future growth
                cudaMalloc(&d_sequence_embeddings, padded_size);
                cudaMalloc(&d_decoder_output, padded_size);
                cudaMalloc(&d_logits, logits_size);
                current_buffer_size = padded_size;
            }

            // Optimization: only compute embeddings for the new token if not the first iteration
            if (generation_step == 0)
            {
                // First iteration: compute embeddings for the entire initial sequence
                wte_layer.forward(
                    context_window,
                    d_sequence_embeddings,
                    config.batch_size,
                    current_seq_len,
                    stream);

                wpe_layer.forward(
                    d_sequence_embeddings,
                    current_seq_len,
                    config.batch_size,
                    stream,
                    0);
            }
            else
            {
                // Subsequent iterations: only compute embeddings for the new token
                int new_token_idx = current_seq_len - 1;
                float *d_new_token_embedding = d_sequence_embeddings + (new_token_idx * config.hidden_dim);

                // Compute token embedding for just the new token
                std::vector<int> single_token = {context_window.back()};
                wte_layer.forward(
                    single_token,
                    d_new_token_embedding,
                    config.batch_size,
                    1,
                    stream);

                // Add positional embedding for the new position
                wpe_layer.forward(
                    d_new_token_embedding,
                    1,
                    config.batch_size,
                    stream,
                    new_token_idx);
            }

            // Reset sequence embeddings buffer for new sequence length
            cudaMemset(d_sequence_embeddings, 0, current_seq_len * config.hidden_dim * sizeof(float));
            cudaMemset(d_decoder_output, 0, current_seq_len * config.hidden_dim * sizeof(float));

            // Compute embeddings for the entire sequence
            debugPrint("Computing embeddings for the entire sequence\n");
            wte_layer.forward(
                context_window,
                d_sequence_embeddings,
                config.batch_size,
                current_seq_len,
                stream);

            debugPrint("Computing position embeddings for the entire sequence\n");
            wpe_layer.forward(
                d_sequence_embeddings,
                current_seq_len,
                config.batch_size,
                stream,
                0); // Start from position 0

            debugPrint("Running decoder with current sequence\n");
            decoder.forward(
                d_decoder_output,
                d_sequence_embeddings,
                nullptr,
                config.batch_size,
                current_seq_len,
                stream);

            // Get the last token's hidden state
            float *d_last_hidden = d_decoder_output + (current_seq_len - 1) * config.hidden_dim;

            debugPrint("Performing linear transformation\n");

            // 1) Matrix multiply: (batch_seq_len=1) x (hidden_dim) times (hidden_dim) x (vocab_size)
            dim3 blockDim(16, 16);
            dim3 gridDim(
                (config.vocab_size + blockDim.x - 1) / blockDim.x,
                (config.batch_size /* * seq_len */ + blockDim.y - 1) / blockDim.y);

            linearTransformKernel<<<gridDim, blockDim>>>(
                d_last_hidden,
                weights->getTokenEmbedding(),
                d_logits,
                config.vocab_size,
                config.batch_size * 1,
                config.hidden_dim);

            cudaError_t error = cudaGetLastError();
            if (error != cudaSuccess)
            {
                std::cerr << "CUDA error in linear transform kernel: "
                          << cudaGetErrorString(error) << std::endl;
            }

            debugPrint("Applying softmax to logits\n");
            applySoftmax(cudnn, d_logits, d_logits, config.batch_size * 1, config.vocab_size);

            std::vector<float> h_logits(config.batch_size * config.vocab_size);
            cudaMemcpy(h_logits.data(), d_logits,
                       config.batch_size * config.vocab_size * sizeof(float),
                       cudaMemcpyDeviceToHost);

            int next_token_id = selectNextToken(h_logits, config.vocab_size, config.temperature);

            context_window.push_back(next_token_id);
            generated_tokens.push_back(next_token_id);
            current_seq_len++;
            generation_step++;

            if (next_token_id == config.stop_token_id)
            {
                break;
            }
        }
        printGeneratedTokens(generated_tokens, vocabulary);
    }

    // Cleanup
    if (d_sequence_embeddings)
        cudaFree(d_sequence_embeddings);
    if (d_decoder_output)
        cudaFree(d_decoder_output);
    if (d_logits)
        cudaFree(d_logits);
    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);
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

    // Run the CLI server with all necessary components
    runCLIServer(
        vocabulary,
        wte_layer,
        wpe_layer,
        weights,
        config,
        decoder,
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
