#!/bin/bash
# illama installation helper script for Ubuntu

set -e

echo "=================================================="
echo "  illama Installation Script for Intel Arc GPUs"
echo "=================================================="

# Check if running as root
if [ "$EUID" -eq 0 ]; then
    echo "Please run as a regular user (not root)"
    exit 1
fi

# Check Ubuntu version
. /etc/os-release
echo "Detected OS: $NAME $VERSION"

# Install Intel GPU tools
echo ""
echo "Installing Intel GPU tools..."
sudo apt update
sudo apt install -y intel-gpu-tools level-zero

# Verify GPU
echo ""
echo "Checking Intel GPU..."
if command -v intel_gpu_top &> /dev/null; then
    echo "✓ intel_gpu_top available"
else
    echo "✗ intel_gpu_top not found"
fi

# Create working directories
echo ""
echo "Creating illama directories..."
sudo mkdir -p /opt/illama/{registry,cache,logs}
sudo chown -R $USER:$USER /opt/illama

sudo mkdir -p /opt/openwebui
sudo chown -R $USER:$USER /opt/openwebui

# Install Python dependencies
echo ""
echo "Installing Python dependencies..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi
source venv/bin/activate

pip install --upgrade pip
pip install -e ".[dev]"

echo ""
echo "=================================================="
echo "  Installation complete!"
echo "=================================================="
echo ""
echo "Next steps:"
echo "  1. Set your Hugging Face token: export HF_TOKEN='your-token'"
echo "  2. Run doctor check: illama doctor"
echo "  3. Pull a model: illama pull Qwen/Qwen3-8B"
echo "  4. Start the server: illama serve"
echo ""
echo "For Docker deployment, see docker/docker-compose.yml"
