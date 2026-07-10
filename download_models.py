#!/usr/bin/env python3
"""Pre-download and cache local models so you don't re-download every run.

Models are stored in the Hugging Face cache (default: ~/.cache/huggingface).
Once cached, `local_model.py` will load them from disk automatically.

Usage:
    # Download one model
    uv run python download_models.py Qwen/Qwen2.5-3B-Instruct

    # Download multiple models
    uv run python download_models.py Qwen/Qwen2.5-3B-Instruct meta-llama/Llama-3.2-3B-Instruct

    # Download all models referenced in experiment configs
    uv run python download_models.py --all-configs

    # Faster downloads with hf-transfer
    HF_HUB_ENABLE_HF_TRANSFER=1 uv run python download_models.py Qwen/Qwen2.5-3B-Instruct
"""

import argparse
import os
import re
import sys
from pathlib import Path

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError as e:
    print("Error: transformers is not installed.")
    print("Run: uv sync --group local")
    sys.exit(1)


def parse_toml_model_refs(path: str) -> set[str]:
    """Parse model references from a TOML config file.

    Looks for strings like `default = "local:Qwen/Qwen2.5-3B-Instruct"`.
    """
    refs = set()
    try:
        text = Path(path).read_text()
    except FileNotFoundError:
        return refs

    for line in text.splitlines():
        if '"local:' in line and '"' in line:
            match = re.search(r'local:([^"]+)', line)
            if match:
                model_name = match.group(1)
                # Strip trailing -thinking/-no-thinking suffixes
                if model_name.endswith("-thinking") or model_name.endswith("-no-thinking"):
                    model_name = model_name.rsplit("-", 1)[0]
                refs.add(model_name)
    return refs


def download_model(model_name: str, token: str | None = None) -> bool:
    """Download and cache tokenizer + model weights for one model."""
    print(f"\n{'='*60}")
    print(f"Downloading: {model_name}")
    print(f"{'='*60}")
    print("Cache directory:", os.getenv("HF_HOME", "~/.cache/huggingface"))
    print("This may take a while depending on the model size and your internet.\n")

    try:
        print("[1/2] Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            token=token,
            trust_remote_code=True,
        )

        print("[2/2] Downloading model weights...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            token=token,
            trust_remote_code=True,
            torch_dtype="auto",
        )

        print(f"\n✓ {model_name} cached successfully")
        print(f"  Model size in memory: {model.num_parameters() / 1e9:.2f}B parameters")
        return True

    except Exception as e:
        print(f"\n✗ Failed to download {model_name}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Pre-download and cache local models for CoffeeBench."
    )
    parser.add_argument(
        "models",
        nargs="*",
        help="Hugging Face model names to download, e.g. Qwen/Qwen2.5-3B-Instruct",
    )
    parser.add_argument(
        "--all-configs",
        action="store_true",
        help="Download all models referenced in experiments/*.toml config files",
    )
    args = parser.parse_args()

    token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")

    models_to_download = set(args.models)

    if args.all_configs:
        experiments_dir = Path(__file__).parent / "experiments"
        if experiments_dir.exists():
            for config_file in experiments_dir.glob("*.toml"):
                models_to_download.update(parse_toml_model_refs(config_file))

    if not models_to_download:
        print("No models specified. Usage:")
        print("  uv run python download_models.py Qwen/Qwen2.5-3B-Instruct")
        print("  uv run python download_models.py --all-configs")
        sys.exit(0)

    print(f"Downloading {len(models_to_download)} model(s)...")
    print(f"HF_TOKEN {'set' if token else 'not set'} (needed for gated models like Llama)")

    results = {}
    for model in sorted(models_to_download):
        results[model] = download_model(model, token=token)

    print(f"\n{'='*60}")
    print("Download Summary")
    print(f"{'='*60}")
    for model, success in results.items():
        status = "✓ DONE" if success else "✗ FAILED"
        print(f"{status}: {model}")

    if not all(results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
