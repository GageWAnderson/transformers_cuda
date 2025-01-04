# Transformer CUDA Implementation

This project implements the Transformer model using CUDA C++ for efficient execution on NVIDIA GPUs of Ampere generation or newer. It leverages cuDNN as a key dependency to optimize neural network operations.

## cuDNN Integration

- **Host Library**: cuDNN provides an API for managing tensor data and neural network operations, which are executed on the GPU. You call cuDNN functions from host code (i.e., in C++), but the operations themselves happen on the device (GPU).
- **Integration with CUDA Kernels**: If you’re writing custom CUDA kernels for your transformer, you can still use cuDNN functions within your host code to handle high-level operations (e.g., softmax, activation) while calling CUDA kernels for the low-level, custom operations.

## Project Structure

- `docs/`: Documentation files.
- `data/`: Input data and output directory.
- `include/`: Header files.
- `src/`: Source files.
- `tests/`: Unit and integration tests.
- `scripts/`: Helper scripts for automation.
- `build/`: Build directory (ignored by version control).

## Getting Started

1. **Prerequisites:**

   - NVIDIA GPU with Ampere architecture or newer.
   - CUDA Toolkit installed.

2. **Build the Project:**

   ```bash
   make
   ```

3. **Run the Transformer:**

   ```bash
   ./bin/transformer --input_file=data/input.txt --output_file=data/output/output.txt
   ```

## License

This project is licensed under the MIT License. 