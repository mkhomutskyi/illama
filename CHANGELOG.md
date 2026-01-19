# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0] - 2026-01-19

### Added

- **Auto-discovery of model folders**: When `index.json` is missing, the registry automatically scans for model folders and creates the index file

- **Performance metrics in API response** (Ollama-style)
  - `eval_count` - tokens generated
  - `prompt_eval_count` - tokens in prompt
  - `eval_duration` - generation time in nanoseconds
  - `total_duration` - total request time in nanoseconds
  - `tokens_per_second` - calculated TPS

- **CLI verbose mode** (`-v` / `--verbose`)
  - Shows performance metrics after response
  - Displays TPS, token counts, and generation duration

### Fixed

- **Reasoning models support** (gpt-oss, DeepSeek-R1, etc.)
  - Increased default `max_tokens` from 512 to 4096 to prevent truncated responses
  - Added `hide_thinking` option to filter `<think>...</think>` content from output
  - Fixed streaming implementation with proper callback-based token delivery
  - Updated CLI defaults to match server settings

- **Dockerfile improvements**
  - Switched to `ppa:kobuk-team/intel-graphics` for Intel Arc GPU drivers (fixes build failures)
  - Pinned OpenVINO to 2025.4.1 and openvino-genai to 2025.4.1.0
  - Added `kernels` and `triton==3.4` dependencies for gpt-oss compatibility

- **CLI PyInstaller compatibility**
  - Fixed `__main__.py` to handle both relative imports (pip install) and absolute imports (PyInstaller binary)

## [0.1.0] - 2026-01-19

### Added

- **illama-manager**: OpenAI-compatible API server for Intel Arc GPUs
  - `GET /v1/models` - List available models in registry
  - `POST /v1/chat/completions` - Chat completions with streaming support
  - `POST /v1/embeddings` - Embedding generation (optional)
  - Single-model-at-a-time loading policy
  - Configurable idle TTL for automatic model unloading
  - OpenVINO GenAI backend for Intel Arc GPU inference

- **illama CLI**: Ollama-like command-line interface
  - `illama pull <model>` - Download, convert, and register models
  - `illama rm <model>` - Remove models from registry
  - `illama list` - List registered models
  - `illama ps` - Show loaded model and runtime stats
  - `illama run <model> [prompt]` - Load model and run prompt
  - `illama serve` - Run local development server
  - `illama doctor` - System diagnostics (GPU, drivers, OpenVINO)

- **Docker support**
  - Dockerfile for illama-manager with OpenVINO runtime
  - docker-compose.yml for Portainer stack deployment
  - Integration with OpenWebUI

- **Model support**
  - HuggingFace model downloading via `huggingface-cli`
  - OpenVINO IR conversion via `optimum-intel`
  - INT4/INT8/FP16 weight format support
  - Model registry with metadata storage

- **Bulk sync script** (`illama_sync_from_metrics.py`)
  - Parse Ollama metrics markdown tables
  - Map Ollama model names to HuggingFace repos
  - Batch model preparation with skip logic for GGUF

- **Documentation**
  - Architecture overview
  - Model compatibility guide
  - Quantization recommendations
  - Troubleshooting guide

### Hardware Support

- Intel Arc B50 (primary target)
- Intel Arc A770/A750 (compatible)
- Intel Arc A380 (compatible, reduced memory)

### Known Limitations

- GGUF models (llama.cpp format) not supported - use llama.cpp SYCL stack instead
- Vision-Language Models (VLMs) require additional setup
- Some HuggingFace models are gated and require acceptance (Llama, Gemma, Codestral)

[Unreleased]: https://github.com/mkhomutskyi/illama/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/mkhomutskyi/illama/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/mkhomutskyi/illama/releases/tag/v0.1.0
