# GPU Server Setup Guide

This guide covers setting up CoffeeBench to run local models on GPU servers.

## Hardware Requirements

### Tested Configuration
- **2x NVIDIA RTX 3090** (24GB VRAM each, 48GB total)
- CUDA 11.8+ or 12.x
- Ubuntu 20.04+ or similar Linux distribution

## Installation

### 1. Install CUDA and cuDNN
```bash
# Verify CUDA installation
nvidia-smi
nvcc --version
```

### 2. Install CoffeeBench with vLLM

```bash
# Clone the repository
cd /path/to/CoffeeBench

# Install with vLLM for optimized multi-GPU inference
uv sync --group vllm
```

### 3. Set Environment Variables

Create a `.env` file in the project root:

```bash
# Hugging Face token (required for gated models like Llama)
HF_TOKEN=your_hf_token_here

# Enable vLLM backend (optional, can also specify per-model)
USE_VLLM=true

# GPU memory optimization
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# For faster downloads from HuggingFace
HF_HUB_ENABLE_HF_TRANSFER=1
```

## vLLM Backend

This setup uses **vLLM** for optimized multi-GPU inference:

- **Automatic tensor parallelism**: Distributes models across both 3090s
- **PagedAttention**: Efficient memory management for long contexts
- **Continuous batching**: Optimized throughput
- **Usage**: `local-vllm:meta-llama/Llama-3.1-8B-Instruct`

```toml
[models]
default = "local-vllm:Qwen/Qwen2.5-32B-Instruct"
```

## Model Recommendations for 2x RTX 3090 (48GB Total)

### Tier 1: Optimal Performance (fits comfortably)
These models will run efficiently with good throughput:

1. **Qwen2.5-32B-Instruct** (~20GB VRAM)
   - Excellent reasoning and tool use
   - Fast inference on 2x 3090s
   ```toml
   default = "local-vllm:Qwen/Qwen2.5-32B-Instruct"
   ```

2. **Llama-3.1-70B-Instruct** (4-bit quantized, ~40GB VRAM)
   - Strong performance, fits with quantization
   - Use with bitsandbytes or AWQ quantization
   ```toml
   default = "local-vllm:meta-llama/Meta-Llama-3.1-70B-Instruct-AWQ"
   ```

3. **Mistral-Small-Instruct-2409** (22B, ~14GB VRAM)
   - Good balance of size and capability
   ```toml
   default = "local-vllm:mistralai/Mistral-Small-Instruct-2409"
   ```

### Tier 2: Maximum Capability (requires careful tuning)
These larger models need optimization:

4. **Qwen2.5-72B-Instruct** (4-bit quantized, ~42GB VRAM)
   - Top-tier reasoning
   - Requires AWQ/GPTQ quantization
   ```toml
   default = "local-vllm:Qwen/Qwen2.5-72B-Instruct-AWQ"
   ```

### Tier 3: Smaller Models (for baseline comparisons)
5. **Qwen2.5-14B-Instruct** (~8GB VRAM)
6. **Llama-3.2-8B-Instruct** (~5GB VRAM)

## Running Experiments

```bash
# Set environment variable
export USE_VLLM=true

# Run experiment
python -m coffeebench.main --experiment experiments/roaster_focal_qwen_32b.toml
```

## Performance Optimization

### 1. vLLM Configuration
vLLM automatically handles:
- Tensor parallelism across GPUs
- PagedAttention for efficient memory
- Continuous batching

### 2. Memory Management
```bash
# If you encounter OOM errors, reduce max context length
export VLLM_MAX_MODEL_LEN=8192

# Or enable CPU offloading (slower but more memory)
export VLLM_CPU_OFFLOAD=true
```

### 3. Monitoring
```bash
# Monitor GPU usage
watch -n 1 nvidia-smi

# Check vLLM logs for throughput
# vLLM prints tokens/sec in stdout
```

## Troubleshooting

### CUDA Out of Memory
1. Use vLLM backend (more memory efficient)
2. Try quantized models (AWQ, GPTQ)
3. Reduce `max_model_len` in vLLM
4. Use smaller batch sizes

### Slow Inference
1. Ensure vLLM is installed and enabled
2. Check GPU utilization with `nvidia-smi`
3. Verify tensor parallelism is active (vLLM logs)

### Model Download Issues
```bash
# Pre-download models
huggingface-cli download meta-llama/Meta-Llama-3.1-70B-Instruct

# Use local cache only
export COFFEEBENCH_LOCAL_ONLY=true
```

## Scaling to More GPUs

If you request additional GPUs (e.g., 4x A100s or H100s), you can run:

### 70B+ Models (Unquantized)
- **Llama-3.1-70B-Instruct** (full precision, ~140GB)
- **Qwen2.5-72B-Instruct** (full precision, ~144GB)

### 405B Models
- **Llama-3.1-405B-Instruct** (requires 8x A100 80GB or 4x H100 80GB)
  ```toml
  default = "local-vllm:meta-llama/Meta-Llama-3.1-405B-Instruct"
  ```

vLLM will automatically distribute these across all available GPUs using tensor parallelism.

## Recommended Setup for Your 2x 3090s

**Best overall choice**: Qwen2.5-32B-Instruct with vLLM
- Excellent performance
- Fast inference
- Good tool use capabilities
- Fits comfortably in 48GB

```bash
# Install
uv sync --group vllm

# Set environment
export USE_VLLM=true
export HF_TOKEN=your_token

# Run experiment
python -m coffeebench.main --experiment experiments/roaster_focal_qwen_32b.toml
```
