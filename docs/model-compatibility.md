# Model Compatibility

This document lists model compatibility and mappings for illama.

## Supported Model Sources

illama can pull models from:

1. **HuggingFace Hub** (primary) - Models in Transformers format
2. **Pre-converted OpenVINO IR** - Already optimized for OpenVINO

> [!IMPORTANT]
> **GGUF models are NOT supported**. GGUF is a llama.cpp format. Use a llama.cpp SYCL stack instead if you need GGUF support.

## Model Mappings

The following table maps common Ollama model names to their HuggingFace source repos:

### Microsoft Phi Models

| Ollama Name | HuggingFace Repo | Notes |
|-------------|------------------|-------|
| `phi4-reasoning:plus` | `microsoft/Phi-4-reasoning-plus` | |
| `phi4-mini-reasoning:3.8b` | `microsoft/Phi-4-mini-reasoning` | |

### Qwen Models

| Ollama Name | HuggingFace Repo | Notes |
|-------------|------------------|-------|
| `qwen3:8b` | `Qwen/Qwen3-8B` | |
| `qwen3:14b` | `Qwen/Qwen3-14B` | |
| `qwen3:30b` | `Qwen/Qwen3-30B-A3B-Instruct-2507` | |
| `qwen3-coder:30b` | `Qwen/Qwen3-Coder-30B-A3B-Instruct` | |
| `qwen3-vl:8b` | `Qwen/Qwen3-VL-8B-Instruct` | VLM - experimental |

### DeepSeek Models

| Ollama Name | HuggingFace Repo | Notes |
|-------------|------------------|-------|
| `deepseek-coder:6.7b` | `deepseek-ai/deepseek-coder-6.7b-instruct` | |
| `deepseek-coder-v2:16b` | `deepseek-ai/DeepSeek-Coder-V2-Instruct` | |
| `deepseek-r1:14b` | `deepseek-ai/DeepSeek-R1-Distill-Qwen-14B` | |
| `deepseek-r1:32b` | `deepseek-ai/DeepSeek-R1-Distill-Qwen-32B` | |

### NVIDIA Models

| Ollama Name | HuggingFace Repo | Notes |
|-------------|------------------|-------|
| `nemotron-3-nano:30b` | `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` | |

### AllenAI Models

| Ollama Name | HuggingFace Repo | Notes |
|-------------|------------------|-------|
| `olmo2:7b` | `allenai/OLMo-2-1124-7B-Instruct` | |

### Mistral Models

| Ollama Name | HuggingFace Repo | Notes |
|-------------|------------------|-------|
| `codestral:22b` | `mistralai/Codestral-22B-v0.1` | **Gated** |
| `mistral-small:24b` | `mistralai/Mistral-Small-24B-Instruct-2501` | |

### Google Models

| Ollama Name | HuggingFace Repo | Notes |
|-------------|------------------|-------|
| `gemma3:4b` | `google/gemma-3-4b-it` | **Gated** |
| `gemma3:12b` | `google/gemma-3-12b-it` | **Gated** |

### IBM Granite Models

| Ollama Name | HuggingFace Repo | Notes |
|-------------|------------------|-------|
| `granite4:3b` | `ibm-granite/granite-4.0-micro` | |
| `granite-code:20b` | `ibm-granite/granite-20b-code-instruct-8k` | |

### Other Models

| Ollama Name | HuggingFace Repo | Notes |
|-------------|------------------|-------|
| `rnj-1:8b` | `EssentialAI/rnj-1-instruct` | |

## Gated Models

Some HuggingFace models require accepting terms before downloading:

> [!WARNING]
> **Gated models require authentication**. You must:
> 1. Create a HuggingFace account
> 2. Accept the model's terms on its HuggingFace page
> 3. Set `HF_TOKEN` environment variable or run `huggingface-cli login`

Known gated models:
- Meta Llama (3.1, 3.2, etc.)
- Google Gemma (2, 3)
- Mistral Codestral
- Google embeddinggemma

## Embedding Models

Embedding models can be supported via `/v1/embeddings` endpoint:

| HuggingFace Repo | Notes |
|------------------|-------|
| `nomic-ai/nomic-embed-text-v1.5` | |
| `nomic-ai/nomic-embed-text-v2-moe` | |
| `google/embeddinggemma-300m` | **Gated** |
| `sentence-transformers/paraphrase-multilingual-mpnet-base-v2` | |

> [!NOTE]
> Embedding models are skipped by default during bulk sync.
> Use `--include-embeddings` flag to include them.

## Vision-Language Models (VLMs)

VLMs like `qwen3-vl` require additional setup for the vision processor.
Treat VLM support as **experimental** in the current stack.

## Adding New Model Mappings

Edit the `OLLAMA_TO_HF` dictionary in:
- `scripts/illama_sync_from_metrics.py` (for bulk sync)
- Or use `illama pull <hf-repo>` directly with the HuggingFace repo ID
