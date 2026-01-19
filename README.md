# illama

<p align="center">
  <strong>Ollama-like LLM experience on Intel Arc GPUs using OpenVINO</strong>
</p>

<p align="center">
  <a href="#features">Features</a> •
  <a href="#quick-start">Quick Start</a> •
  <a href="#installation">Installation</a> •
  <a href="#usage">Usage</a> •
  <a href="#docker">Docker</a> •
  <a href="#documentation">Docs</a>
</p>

---

**illama** (intel lamma) provides an Ollama-like experience for running LLMs locally on Intel Arc GPUs. It uses OpenVINO for optimized inference and provides both a CLI and OpenAI-compatible API.

## Features

- 🚀 **Ollama-like CLI** - Familiar commands: `illama pull`, `illama run`, `illama ps`
- 🔌 **OpenAI-compatible API** - Works with OpenWebUI and other OpenAI clients
- 💾 **Single-model loading** - Optimized for consumer GPUs with limited VRAM
- ⏱️ **Idle auto-eviction** - Frees GPU memory when not in use
- 📦 **INT4/INT8/FP16 quantization** - Flexible precision options
- 🐳 **Docker support** - Ready for Portainer stack deployment

## Hardware Requirements

- **Intel Arc GPU** (B50, A770, A750, A380)
- Ubuntu 24.04+ (or compatible Linux)
- Intel GPU drivers (Level Zero runtime)
- 8GB+ GPU VRAM recommended

## Quick Start

```bash
# Install
pip install -e .

# Check system
illama doctor

# Pull a model (converts to OpenVINO INT4)
illama pull Qwen/Qwen3-8B --weight-format int4

# Run interactively
illama run Qwen3-8B
>>> Hello, how are you?

# Or start the API server
illama serve
```

## Installation

### Prerequisites

1. **Intel GPU drivers**:
   ```bash
   sudo apt update
   sudo apt install -y intel-gpu-tools level-zero
   ```

2. **Verify GPU**:
   ```bash
   sudo intel_gpu_top
   ```

### Install illama

```bash
# Clone the repository
git clone https://github.com/mkhomutskyi/illama.git
cd illama

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install
pip install -e .

# For development
pip install -e ".[dev]"
```

### System-wide Installation (Optional)

**Option A: Symlink (recommended for development)**
```bash
# After pip install, create symlink to the venv's illama
sudo ln -s $(pwd)/venv/bin/illama /usr/local/bin/illama
```

**Option B: Standalone binary with PyInstaller**
```bash
# Install PyInstaller
pip install pyinstaller

# Build standalone binary
pyinstaller --onefile --name illama illama_cli/__main__.py

# Install to system
sudo mv dist/illama /usr/local/bin/
sudo chmod +x /usr/local/bin/illama

# Verify
illama --version
```

**Option C: pip install globally** (not recommended - may conflict with system packages)
```bash
sudo pip install .
```

### Configure HuggingFace Token

Some models require authentication:

```bash
export HF_TOKEN="your-huggingface-token"
# or
huggingface-cli login
```

## Usage

### CLI Commands

| Command | Description |
|---------|-------------|
| `illama pull <model>` | Download, convert, and register a model |
| `illama rm <model>` | Remove model from registry |
| `illama list` | List registered models |
| `illama ps` | Show loaded model status |
| `illama run <model> [prompt]` | Chat with a model |
| `illama serve` | Start the API server |
| `illama doctor` | System diagnostics |

### Pull a Model

```bash
# From HuggingFace (auto-converts to OpenVINO)
illama pull microsoft/Phi-4-mini-reasoning --weight-format int4

# Different quantization
illama pull Qwen/Qwen3-8B --weight-format int8
```

### Chat

```bash
# Interactive mode
illama run Phi-4-mini-reasoning
>>> What is the capital of France?

# Single prompt
illama run Qwen3-8B "Explain quantum computing"
```

### API Server

```bash
# Start server
illama serve --port 11434

# Test
curl http://localhost:11434/v1/models | jq

# Chat completion
curl -X POST http://localhost:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3-8B",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

## Docker

### Quick Deploy with Docker Compose

```bash
cd docker
docker compose up -d
```

### Portainer Stack

1. Go to Portainer → Stacks → Add Stack
2. Paste contents of `docker/docker-compose.yml`
3. Set environment variable `HF_TOKEN`
4. Deploy

### OpenWebUI Integration

After deploying, access OpenWebUI at `http://your-server:3000`.

Configuration:
- Settings → Connections
- OpenAI API Base URL: `http://illama-manager:11434/v1`

## Configuration

Environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `ILLAMA_DEVICE` | `GPU` | Device: GPU, CPU, AUTO |
| `ILLAMA_ONE_MODEL` | `1` | Single-model policy |
| `ILLAMA_IDLE_TTL_SEC` | `600` | Idle timeout (seconds) |
| `ILLAMA_PORT` | `11434` | API server port |
| `HF_TOKEN` | - | HuggingFace token |

## Documentation

- [Architecture](docs/architecture.md) - System design and components
- [Model Compatibility](docs/model-compatibility.md) - Supported models and mappings
- [Quantization](docs/quantization.md) - INT4/INT8/FP16 guide
- [Troubleshooting](docs/troubleshooting.md) - Common issues and solutions

## Quantization

**Recommended**: Use INT4 for best performance/memory on Intel Arc.

| Format | VRAM (7B model) | Speed |
|--------|-----------------|-------|
| FP16 | ~14 GB | Baseline |
| INT8 | ~7 GB | Faster |
| INT4 | ~3.5 GB | Fastest |

## Limitations

- **GGUF not supported** - Use llama.cpp SYCL for GGUF models
- **VLMs experimental** - Vision models need extra setup
- **Gated models** - Require HuggingFace acceptance (Llama, Gemma, etc.)

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

Apache 2.0 - see [LICENSE](LICENSE)

## Acknowledgments

- [OpenVINO](https://github.com/openvinotoolkit/openvino) - Intel's inference toolkit
- [Ollama](https://github.com/ollama/ollama) - Inspiration for CLI/UX
- [OpenWebUI](https://github.com/open-webui/open-webui) - Web interface
- [Optimum Intel](https://github.com/huggingface/optimum-intel) - Model conversion
