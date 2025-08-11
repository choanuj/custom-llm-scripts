## Introduction

This repository provides a set of Python scripts to create, train, convert to GGUF format, and run inference on a custom Large Language Model (LLM) based on a GPT-like transformer architecture. It's designed for educational and experimental purposes, allowing users to build a small-scale LLM from scratch using PyTorch.

Key features:-
- **Decoder-Only Transformer**: Simple GPT-style model for text generation.
- **Command-Line Focus**: Easy-to-follow steps for creation, training, conversion, and usage.
- **GGUF Conversion**: Optimize for efficient inference with quantization (e.g., Q4_0).
- **Optional GUI**: An integrated Tkinter GUI (`llm_gui.py`) for visual management.
- **Resource Monitoring & Logging**: Built-in for tracking CPU/GPU usage and errors.

This setup is resource-efficient for small models/datasets but scales with hardware (GPU recommended). It's inspired by tutorials like "LLMs-from-scratch" and supports extensions for reasoning or multi-modal tasks.

**Disclaimer**: Training large models requires significant compute (e.g., GPU/TPU). Start small to test.

## Files Overview

- **`llm_model.py`**: Defines the GPT-like model architecture and allows user-configurable hyperparameters. Saves `model_config.pth`.
- **`train_llm.py`**: Trains the model on a text dataset. Saves `trained_model.pth`.
- **`convert_to_gguf.py`**: Converts the trained PyTorch model to GGUF format for efficient inference. Requires `llama.cpp`.
- **`inference_gguf.py`**: Runs text generation (inference) on the GGUF model with a query.
- **`llm_gui.py`**: Optional GUI for all steps (creation, training, conversion, inference) with real-time monitoring.
- **`requirements.txt`**: List of Python dependencies.
- **`input.txt` (example)**: Sample text dataset for training (create your own or download from sources like Hugging Face).

Logs: `gui.log`, `conversion.log`, `inference.log` (generated during runs).

## Prerequisites

1. **Python 3.8+**: Install from [python.org](https://www.python.org/).
2. **Dependencies**: Run `pip install -r requirements.txt`.
   - For Mac GPU (MPS): `CMAKE_ARGS="-DLLAMA_METAL=on" pip install llama-cpp-python`.
3. **llama.cpp**: For GGUF conversion.
   ```bash
   git clone https://github.com/ggerganov/llama.cpp
   cd llama.cpp
   make
   ```
   Place the `llama.cpp` folder in your working directory or update the path in `convert_to_gguf.py`.
4. **Hardware**: CPU works for small models; GPU (NVIDIA CUDA or Apple MPS) for training/inference.
5. **Dataset**: A text file (e.g., `input.txt`) with clean data.
6. Sources:
   - Hugging Face: The Pile, C4 (download via `datasets` library: `pip install datasets`).
   - Kaggle or Project Gutenberg for free texts.
   - Ethical note: Use public-domain or licensed data; avoid biases.

## Step-by-Step Command-Line Guide

Follow these steps in a terminal (Mac/Linux: Terminal; Windows: Command Prompt/PowerShell). Navigate to your directory: `cd path/to/custom-llm-scripts`.

### 1. Create the LLM Model
This script initializes the model architecture and saves its configuration.

```bash
python llm_model.py
```

- **Inputs** (prompted):
  - Vocabulary size (e.g., 10000 – smaller for testing).
  - Embedding dimension (e.g., 256).
  - Attention heads (e.g., 8).
  - Layers (e.g., 6 – fewer for faster training).
  - Max sequence length (block_size, e.g., 128).
  - Dropout (e.g., 0.1).
  - Press Enter for defaults.
- **Outputs**: `model_config.pth` file; console shows parameter count (e.g., "10.24M parameters").
- **Explanation**: Builds a decoder-only transformer (self-attention, feed-forward layers). Customizable for experimentation (e.g., increase layers for better performance, but more compute needed).
- **Time**: Instant.
- **Tips**: Start small (e.g., vocab_size=5000, layers=4) to avoid memory issues.

### 2. Train the LLM
Trains the model on your dataset using the config from Step 1.

```bash
python train_llm.py
```

- **Inputs** (prompted):
  - Batch size (e.g., 32 – smaller if OOM errors).
  - Learning rate (e.g., 0.0003 – tune for convergence).
  - Epochs (e.g., 5 – more for better training).
  - Dataset path (e.g., `input.txt`).
  - Press Enter for defaults.
- **Outputs**: `trained_model.pth` (weights); per-epoch loss in console; sample generated text.
- **Explanation**: 
  - Loads config and dataset.
  - Preprocesses: Character-level tokenization (splits text into chars, creates vocab).
  - Training: Autoregressive (next-token prediction) with AdamW optimizer and cross-entropy loss.
  - Auto-detects device (GPU if available).
  - Generates sample text post-training to verify.
- **Time**: Minutes to hours (depends on dataset size, model, hardware; e.g., 1MB dataset, 5 epochs: ~10-30 mins on GPU).
- **Tips**: 
  - Dataset: Use diverse text (e.g., books) for better generalization. Clean data to remove noise.
  - Overfitting: Monitor loss; add dropout if needed.
  - Reasoning: Fine-tune on datasets like GSM8K (modify script to load Hugging Face datasets).

### 3. Convert to GGUF Format
Optimizes the trained model for inference (smaller size, faster on CPU/GPU).

```bash
python convert_to_gguf.py --quant_type Q4_0
```

- **Options**:
  - `--quant_type`: Q4_0 (default, low precision, small file), Q5_K (better quality), F16 (full precision).
- **Outputs**: `model.gguf.Q4_0.gguf` (or specified type); logs in `conversion.log`.
- **Explanation**:
  - Exports PyTorch model to Hugging Face format.
  - Uses `llama.cpp` to convert and quantize (reduces bit precision for efficiency).
  - GGUF benefits: Runs on low-resource devices (e.g., CPU), faster inference.
- **Time**: Seconds to minutes.
- **Tips**: 
  - Update `llama_cpp_path` in script if not in directory.
  - Test with F16 first if quantization degrades quality.

### 4. Use the Trained Model (Inference)
Generate text from a query using the GGUF model.

```bash
python inference_gguf.py --gguf_path model.gguf.Q4_0.gguf --query "Generate a story about AI" --max_tokens 100 --use_gpu
```

- **Options**:
  - `--gguf_path`: Path to GGUF file from Step 3.
  - `--query`: Input prompt.
  - `--max_tokens`: Output length (default: 100).
  - `--use_gpu`: Enable GPU (MPS on Mac).
- **Outputs**: Generated text in console; resource usage and logs in `inference.log`.
- **Explanation**:
  - Loads GGUF with `llama-cpp-python`.
  - Autoregressive generation: Predicts tokens one-by-one.
  - Monitors resources (CPU, RAM, GPU).
- **Time**: Seconds.
- **Tips**: 
  - For reasoning: Use chain-of-thought prompts (e.g., "Solve step-by-step: 2+2?").
  - Extend: Modify for chat-like interactions.

## GUI Usage (Optional)
For a visual interface:
```bash
python llm_gui.py
```
- Sections: Model Creation (enter params), Dataset Management (upload text), Training (set params, start), Conversion/Inference (buttons).
- Explanation: Integrates all steps with progress bars, logs, and resource monitoring. Use for beginners; command line for efficiency.

## Troubleshooting
- **OOM Errors**: Reduce batch_size, model size, or use GPU.
- **File Not Found**: Verify paths (e.g., `input.txt`).
- **GPU Issues**: Check `torch.backends.mps.is_available()` (Mac) or CUDA setup.
- **Conversion Fail**: Build `llama.cpp` correctly; check logs.
- **Dataset Biases**: Use clean, diverse data.
- **Logs**: Review *.log files for details.
- **Scaling**: For large models, use cloud (e.g., AWS/GCP) or fine-tune pre-trained (Hugging Face).

## License
MIT License – Feel free to use, modify, and distribute.

Contributions welcome! Fork and PR.
```
