# Model Recommendations for 2x RTX 3090 (48GB Total VRAM)

## Quick Start: Best Choice

**Qwen2.5-32B-Instruct** with vLLM backend
- **VRAM Usage**: ~20GB (fits comfortably)
- **Performance**: Excellent reasoning and tool use
- **Throughput**: Fast inference with vLLM optimization
- **Config**: `experiments/roaster_focal_qwen_32b.toml`

```bash
uv sync --group vllm
export USE_VLLM=true
export HF_TOKEN=your_hf_token
python -m coffeebench.main --config experiments/roaster_focal_qwen_32b.toml
```

---

## All Recommended Models

### Tier 1: Optimal for 2x 3090s

#### 1. **Qwen2.5-32B-Instruct** ⭐ RECOMMENDED
- **Size**: 32B parameters (~20GB VRAM)
- **Strengths**: Best balance of capability and efficiency
- **Use Case**: Primary experiments, production runs
- **Config**: `roaster_focal_qwen_32b.toml`
- **Model ID**: `Qwen/Qwen2.5-32B-Instruct`

#### 2. **Mistral-Small-Instruct-2409**
- **Size**: 22B parameters (~14GB VRAM)
- **Strengths**: Fast inference, good instruction following
- **Use Case**: Faster iterations, baseline comparisons
- **Config**: `roaster_focal_mistral_small.toml`
- **Model ID**: `mistralai/Mistral-Small-Instruct-2409`

#### 3. **Llama-3.1-70B-Instruct (AWQ Quantized)**
- **Size**: 70B parameters, 4-bit quantized (~40GB VRAM)
- **Strengths**: Highest capability on 2x 3090s
- **Use Case**: Maximum performance experiments
- **Config**: `roaster_focal_llama_70b_awq.toml`
- **Model ID**: `hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4`
- **Note**: Requires quantized version to fit in 48GB

### Tier 2: Smaller Models (for baselines)

#### 4. **Qwen2.5-14B-Instruct**
- **Size**: 14B parameters (~8GB VRAM)
- **Strengths**: Fast, efficient, good baseline
- **Config**: `roaster_focal_qwen_14b.toml`
- **Model ID**: `Qwen/Qwen2.5-14B-Instruct`

#### 5. **Llama-3.2-8B-Instruct**
- **Size**: 8B parameters (~5GB VRAM)
- **Strengths**: Very fast, good for rapid testing
- **Existing Config**: `roaster_focal_llama_3b.toml` (update to 8B)
- **Model ID**: `meta-llama/Llama-3.2-8B-Instruct`

---

## Performance Comparison

| Model | Size | VRAM | Relative Speed | Capability | Best For |
|-------|------|------|----------------|------------|----------|
| Qwen2.5-32B | 32B | ~20GB | Fast | Excellent | **Primary choice** |
| Mistral-Small | 22B | ~14GB | Very Fast | Good | Fast iterations |
| Llama-3.1-70B-AWQ | 70B | ~40GB | Moderate | Best | Max capability |
| Qwen2.5-14B | 14B | ~8GB | Very Fast | Good | Baselines |
| Llama-3.2-8B | 8B | ~5GB | Fastest | Moderate | Rapid testing |

---

## If You Request More GPUs

### 4x A100 (80GB each = 320GB total)
You can run much larger models unquantized:

1. **Llama-3.1-405B-Instruct** (full precision)
   - The largest open model available
   - Requires ~250GB VRAM
   - Excellent for research-grade experiments
   ```toml
   default = "local-vllm:meta-llama/Meta-Llama-3.1-405B-Instruct"
   ```

2. **Qwen2.5-72B-Instruct** (full precision)
   - Top-tier reasoning without quantization
   - Requires ~144GB VRAM
   ```toml
   default = "local-vllm:Qwen/Qwen2.5-72B-Instruct"
   ```

3. **Llama-3.1-70B-Instruct** (full precision)
   - Better than quantized version
   - Requires ~140GB VRAM
   ```toml
   default = "local-vllm:meta-llama/Meta-Llama-3.1-70B-Instruct"
   ```

### 8x H100 (80GB each = 640GB total)
You can run the absolute largest models:

1. **Llama-3.1-405B-Instruct** with plenty of headroom
2. Multiple concurrent model instances
3. Extremely long context lengths (100K+ tokens)

---

## Model Selection Guide

### Choose **Qwen2.5-32B-Instruct** if:
- ✅ You want the best overall performance
- ✅ You need good reasoning and tool use
- ✅ You want fast inference
- ✅ This is your primary experiment model

### Choose **Mistral-Small-Instruct-2409** if:
- ✅ You need faster iterations
- ✅ You're doing many quick experiments
- ✅ You want a good baseline model

### Choose **Llama-3.1-70B-AWQ** if:
- ✅ You need maximum capability on 2x 3090s
- ✅ You can accept slightly slower inference
- ✅ You want to test against larger models

### Choose **Qwen2.5-14B** or **Llama-3.2-8B** if:
- ✅ You need baseline comparisons
- ✅ You want very fast testing cycles
- ✅ You're debugging the benchmark itself

---

## Installation Commands

```bash
# 1. Install dependencies
cd /path/to/CoffeeBench
uv sync --group vllm

# 2. Set up environment
cat > .env << EOF
HF_TOKEN=your_huggingface_token_here
USE_VLLM=true
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
HF_HUB_ENABLE_HF_TRANSFER=1
EOF

# 3. Pre-download recommended model (optional but recommended)
huggingface-cli download Qwen/Qwen2.5-32B-Instruct

# 4. Run experiment
python -m coffeebench.main --config experiments/roaster_focal_qwen_32b.toml
```

---

## Troubleshooting

### Out of Memory Errors
1. Switch to a smaller model (Qwen2.5-14B or Mistral-Small)
2. Reduce context length: `export VLLM_MAX_MODEL_LEN=8192`
3. Use quantized versions (AWQ/GPTQ)

### Slow Inference
1. Verify vLLM is being used: check logs for "vLLM" messages
2. Check GPU utilization: `nvidia-smi`
3. Ensure both GPUs are being used: vLLM should show tensor parallelism

### Model Download Issues
1. Set HF_TOKEN in environment
2. Pre-download: `huggingface-cli download <model_id>`
3. For Llama models, accept license on HuggingFace website first

---

## Summary

**For your 2x RTX 3090 setup, start with Qwen2.5-32B-Instruct using vLLM.** It offers the best balance of performance, speed, and reliability. The experiment config is ready at `experiments/roaster_focal_qwen_32b.toml`.

If you need more capability and can request additional GPUs, 4x A100s would allow you to run Llama-3.1-405B or Qwen2.5-72B at full precision, which would be excellent for research-grade experiments.
