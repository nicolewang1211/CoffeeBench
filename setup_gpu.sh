#!/bin/bash
# GPU setup script - run this after git pull on GPU server

set -e

echo "Setting up CoffeeBench for GPU..."

# Activate venv
source .venv/bin/activate

# Sync dependencies
uv sync

# Reinstall PyTorch and torchvision with CUDA support
echo "Installing PyTorch with CUDA 12.1 support..."
uv pip install --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"

echo "Setup complete! You can now run experiments."
