#!/usr/bin/env bash
set -e  # Exit on error

echo "🔹 Checking for uv..."
if ! command -v uv &> /dev/null; then
    echo "uv not found. Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
else
    echo "uv already installed."
fi

echo "🔹 Creating .venv..."
uv venv .venv

echo "🔹 Activating virtual environment..."
source .venv/bin/activate

echo "🔹 Upgrading pip..."
uv pip install --upgrade pip

echo "🔹 Installing PyTorch (CUDA 12.1 example)..."
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo "🔹 Installing vLLM and Transformers..."
uv pip install vllm transformers accelerate pandas

echo "🔹 Running model downloader for the default local model set..."
python model_downloader.py

echo "🔹 Setup complete!"
