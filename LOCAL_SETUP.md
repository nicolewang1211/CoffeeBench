# Running CoffeeBench Locally on Mac with Llama and Qwen

This guide explains how to run CoffeeBench completely locally on your Mac using models like Llama and Qwen, without any API costs.

## Prerequisites

- Mac with Apple Silicon (M1/M2/M3) or Intel
- At least 16GB RAM (32GB+ recommended for larger models)
- Python 3.12+
- ~10-20GB free disk space for model downloads

## Installation

### 1. Install Base Dependencies

```bash
uv sync
```

### 2. Install Local Model Dependencies

For Hugging Face transformers (recommended for Mac):

```bash
uv sync --group local
```

For GGUF models (optional, more memory efficient):

```bash
uv sync --group gguf
```

## Pre-download Models (Recommended)

Run this once to download and cache all models you need, so the benchmark runs don't re-download:

```bash
# Download all models referenced in experiments/*.toml
uv run python download_models.py --all-configs

# Or download specific models
uv run python download_models.py Qwen/Qwen2.5-3B-Instruct meta-llama/Llama-3.2-3B-Instruct
```

Models are cached in `~/.cache/huggingface` by default. After the first download, subsequent runs load from disk.

## Quick Test

Test that local models work on your system:

```bash
uv run python test_local_models.py
```

This will download and test small models (~3B parameters) that should run well on most Macs. If you already downloaded the model above, this will load from cache and run quickly.

## Available Model Configurations

### Smaller Models (Recommended for Testing)

**Qwen 3B** - Fast, efficient, good quality:
```bash
uv run python -m coffeebench.main \
    --config experiments/roaster_focal_qwen_3b.toml --seed 0
```

**Llama 3.2 3B** - Fast, efficient:
```bash
uv run python -m coffeebench.main \
    --config experiments/roaster_focal_llama_3b.toml --seed 0
```

### Larger Models (Better Quality, More RAM)

**Qwen 7B** - Better quality, requires more RAM:
```bash
uv run python -m coffeebench.main \
    --config experiments/roaster_focal_qwen.toml --seed 0
```

**Llama 3.1 8B** - High quality, requires 16GB+ RAM:
```bash
uv run python -m coffeebench.main \
    --config experiments/roaster_focal_llama.toml --seed 0
```

## Custom Model Configuration

You can create your own config file or override models via command line:

### Using Config Files

Create a new `.toml` file in `experiments/`:

```toml
[experiment]
name = "my_local_test"
description = "Testing with local models"

[run]
seeds = [0]
max_days = 10  # Shorter run for testing
main_agent = "roaster_A"

[models]
default = "local:Qwen/Qwen2.5-3B-Instruct"
roaster_A = "local:Qwen/Qwen2.5-7B-Instruct"  # Different model for focal agent
```

### Command Line Override

```bash
uv run python -m coffeebench.main \
    --config experiments/roaster_focal_qwen_3b.toml \
    --seed 0 \
    --models 'roaster_A:local:Qwen/Qwen2.5-7B-Instruct'
```

## Supported Local Models

### Llama Models (Meta)
- `local:meta-llama/Llama-3.2-1B-Instruct` (smallest, fastest)
- `local:meta-llama/Llama-3.2-3B-Instruct` (good balance)
- `local:meta-llama/Llama-3.1-8B-Instruct` (high quality)

### Qwen Models (Alibaba)
- `local:Qwen/Qwen2.5-3B-Instruct` (good balance)
- `local:Qwen/Qwen2.5-7B-Instruct` (high quality)
- `local:Qwen/Qwen2.5-14B-Instruct` (best quality, requires 32GB+ RAM)

### Other Compatible Models
Any Hugging Face model with instruction tuning should work:
- `local:mistralai/Mistral-7B-Instruct-v0.3`
- `local:google/gemma-2-9b-it`
- `local:microsoft/Phi-3-mini-4k-instruct`

## Performance Tips

### For Mac with Apple Silicon (M1/M2/M3)

The models will automatically use Metal Performance Shaders (MPS) for GPU acceleration:

```bash
# Models will auto-detect and use MPS
uv run python -m coffeebench.main \
    --config experiments/roaster_focal_qwen_3b.toml --seed 0
```

### Memory Management

If you run out of memory:

1. **Use smaller models**: Start with 3B models
2. **Reduce max_days**: Test with shorter runs (e.g., 10 days)
3. **Use GGUF format**: More memory efficient

```bash
# Example with GGUF (requires llama-cpp-python)
# Download a GGUF model first, then:
uv run python -m coffeebench.main \
    --config experiments/roaster_focal_qwen_3b.toml \
    --seed 0 \
    --models 'default:local:path/to/model.gguf'
```

### Speed Optimization

For faster inference on Mac:

1. **Close other applications** to free up RAM
2. **Use smaller models** for initial testing
3. **Reduce context length** if needed (models auto-manage this)

## Hugging Face Token Setup (Recommended)

### What does the HF token do?

A Hugging Face token does **not** make downloads much faster by itself, but it unlocks two important things:

1. **Access to gated models** — The official Llama models (`meta-llama/*`) require you to accept a license on Hugging Face before you can download them.
2. **Higher download limits** — Authenticated users get higher rate limits, which can help avoid throttling during large downloads.

### How to set it up

1. **Get your token** at https://huggingface.co/settings/tokens
   - Create a token with at least `read` access.

2. **Add it to your `.env` file:**

```bash
HF_TOKEN="hf_..."
```

3. **Or set it in your terminal before running:**

```bash
export HF_TOKEN="hf_..."
uv run python test_local_models.py
```

4. **For official Llama models**, you also need to accept the license on the model page:
   - Go to https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct
   - Click **“Access repository”** and accept the terms.

### Faster downloads

If your network is slow, here are the best ways to speed things up:

1. **Pre-download the model** before the benchmark run. This ensures the download only happens once:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

model_name = "Qwen/Qwen2.5-3B-Instruct"
hf_token = os.getenv("HF_TOKEN")

AutoTokenizer.from_pretrained(model_name, token=hf_token, trust_remote_code=True)
AutoModelForCausalLM.from_pretrained(model_name, token=hf_token, trust_remote_code=True)
print("Download complete")
```

2. **Use `hf_transfer` for multi-part downloads** (can be 2-5x faster on good connections):

```bash
pip install hf_transfer
export HF_HUB_ENABLE_HF_TRANSFER=1
uv run python test_local_models.py
```

3. **Use a faster mirror or cache** if you are in a region with slow Hugging Face connectivity.

## Monitoring

Use the web dashboard to monitor your local simulation in real-time:

```bash
# In a separate terminal
uv run streamlit run coffeebench/web.py
```

Then open http://localhost:8501 in your browser.

## Troubleshooting

### Model Download Issues

Models are automatically downloaded from Hugging Face and cached. If you have issues:

```bash
# Set Hugging Face cache directory (default is already ~/.cache/huggingface)
export HF_HOME=~/.cache/huggingface

# Login to Hugging Face (for gated models like Llama)
huggingface-cli login
```

#### Avoid re-downloading models

Once you pre-download a model, it will be cached. Make sure you don't change `HF_HOME` between runs, or the cache will appear empty.

To force the benchmark to use **only cached** models (no network access):

```bash
export COFFEEBENCH_LOCAL_ONLY=1
uv run python -m coffeebench.main --config experiments/roaster_focal_qwen_3b.toml --seed 0
```

This will fail if the model is not cached, so run `download_models.py` first.

### Segmentation Fault / Crash (exit code 139)

If the benchmark crashes with `exit code 139` or segfault after loading models:

```bash
# This is an MPS (Metal Performance Shaders) bug on some Macs
# Workaround: disable MPS and use CPU instead
export PYTORCH_MPS_DISABLE=1
uv run python -m coffeebench.main --config experiments/roaster_focal_qwen_3b.toml --seed 0
```

**Why this happens**: PyTorch's MPS backend on Mac can crash during multi-agent inference. CPU mode is slower but stable.

**Performance impact**: CPU is ~2-3x slower than MPS, but the benchmark will complete successfully. A 90-day run with 3B models on CPU takes ~3-6 hours.

### Out of Memory

```
RuntimeError: MPS backend out of memory
```

Solutions:
- Use a smaller model (3B instead of 7B)
- Reduce max_days in config
- Close other applications
- Restart your Mac to clear memory
- Try CPU mode: `export PYTORCH_MPS_DISABLE=1`

### Slow Performance

Local models are slower than API calls. Expected times:
- **3B models**: ~5-10 seconds per agent decision
- **7B models**: ~10-20 seconds per agent decision
- **Full 90-day run**: Several hours to days depending on model size

For faster testing:
- Use smaller models
- Reduce max_days to 10-30 for testing
- Test with single seed first

### Model Not Found

```
OSError: meta-llama/Llama-3.1-8B-Instruct does not appear to be a valid model
```

Some models require Hugging Face authentication:

```bash
huggingface-cli login
# Then accept the model license on Hugging Face website
```

## Cost Comparison

| Setup | Cost per 90-day run | Speed |
|-------|-------------------|-------|
| API (Sonnet) | ~$200+ | Fast (hours) |
| Local 3B | $0 | Slow (days) |
| Local 7B | $0 | Very slow (days) |

**Recommendation**: Start with local 3B models for testing, then use API models for full experiments.

## Example Workflow

1. **Quick test** (5-10 minutes):
```bash
uv run python test_local_models.py
```

2. **Short simulation** (1-2 hours):
```bash
uv run python -m coffeebench.main \
    --config experiments/roaster_focal_qwen_3b.toml \
    --seed 0 \
    --max-days 10
```

3. **Full simulation** (overnight):
```bash
uv run python -m coffeebench.main \
    --config experiments/roaster_focal_qwen_3b.toml \
    --seed 0
```

## Additional Resources

- [Hugging Face Model Hub](https://huggingface.co/models)
- [Llama Models](https://huggingface.co/meta-llama)
- [Qwen Models](https://huggingface.co/Qwen)
- [CoffeeBench Paper](https://arxiv.org/abs/2606.16613)
