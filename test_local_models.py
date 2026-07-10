#!/usr/bin/env python3
"""Quick test script to verify local models work on your Mac.

This script tests local model inference without running the full benchmark.

Usage:
    # Test all default models
    uv run python test_local_models.py

    # Test one specific model
    uv run python test_local_models.py local:Qwen/Qwen2.5-3B-Instruct

For faster downloads, set HF_HUB_ENABLE_HF_TRANSFER=1:
    HF_HUB_ENABLE_HF_TRANSFER=1 uv run python test_local_models.py
"""

import os
import sys
from coffeebench.models import get_model


def test_model(model_spec: str):
    """Test a single model with a simple query."""
    print(f"\n{'='*60}")
    print(f"Testing: {model_spec}")
    print(f"{'='*60}")
    print("NOTE: First run will download the model. This can take 5-30 minutes.")
    print("      Press Ctrl+C to cancel. Set HF_HUB_ENABLE_HF_TRANSFER=1 for faster downloads.")
    
    try:
        # Initialize model
        print("\nLoading model (this may take a while on first run)...")
        model = get_model(model_spec)
        
        # Simple test query
        messages = [
            {"role": "user", "content": "Hello! Please respond with a brief greeting."}
        ]
        
        print("Sending test query...")
        response = model.query(messages)
        
        print(f"\n✓ Success!")
        print(f"Response: {response.content[:200]}...")
        
        # Show stats
        stats = model.get_usage_stats()
        print(f"\nStats:")
        print(f"  - Calls: {stats['n_model_calls']}")
        print(f"  - Input tokens: {stats['total_input_tokens']}")
        print(f"  - Output tokens: {stats['total_output_tokens']}")
        print(f"  - Cost: ${stats['model_cost']:.2f}")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Test local model(s) passed as argument or default list."""
    print("CoffeeBench Local Model Test")
    print("="*60)

    # Enable hf-transfer if installed, but warn user they need to set the env var
    if os.getenv("HF_HUB_ENABLE_HF_TRANSFER") != "1":
        print("\nTip: For faster downloads, run with:")
        print("  HF_HUB_ENABLE_HF_TRANSFER=1 uv run python test_local_models.py")
    else:
        print("\nhf-transfer enabled for faster downloads.")
    
    if len(sys.argv) > 1:
        # Test models passed as command line arguments
        test_models = sys.argv[1:]
    else:
        # Default models to test (from smallest to largest for Mac compatibility)
        test_models = [
            # Smaller models that should work well on Mac
            "local:Qwen/Qwen2.5-3B-Instruct",
            "local:meta-llama/Llama-3.2-3B-Instruct",
            
            # Medium models (may require more RAM)
            # Uncomment if you have enough RAM (16GB+)
            # "local:Qwen/Qwen2.5-7B-Instruct",
            # "local:meta-llama/Llama-3.1-8B-Instruct",
        ]
    
    results = {}
    for model_spec in test_models:
        success = test_model(model_spec)
        results[model_spec] = success
        print(f"\n{'='*60}")
    
    # Summary
    print("Test Summary")
    print(f"{'='*60}")
    for model_spec, success in results.items():
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status}: {model_spec}")
    
    # Exit with error if any tests failed
    if not all(results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
