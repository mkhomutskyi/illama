# Quantization Guide

This document explains quantization options for illama on Intel Arc GPUs.

## Overview

Quantization reduces model size and memory usage by using lower-precision data types.

| Format | Description | Memory | Quality | Speed |
|--------|-------------|--------|---------|-------|
| FP16/BF16 | Full precision | Highest | Best | Baseline |
| INT8 | 8-bit integer | ~50% of FP16 | Very Good | Faster |
| INT4 | 4-bit integer | ~25% of FP16 | Good | Fastest |

## Recommendations for Intel Arc B50

> [!TIP]
> **Default to INT4 weight-only** for 7B-30B class models.
> This provides the best throughput/memory tradeoff on Intel Arc GPUs.

### When to use INT4

- Models 7B-30B parameters
- Consumer GPUs with limited VRAM (8-16GB)
- Best inference speed
- Acceptable quality for most use cases

### When to use INT8

- If INT4 shows quality degradation
- Smaller models (< 7B) where VRAM isn't a concern
- Tasks requiring higher precision (math, code)

### When to use FP16/BF16

- Benchmarking baseline quality
- Models specifically trained in BF16
- When you have ample VRAM

## OpenVINO Quantization Methods

OpenVINO + NNCF supports:

### Post-Training Quantization (PTQ)

- **INT8 PTQ**: Traditional 8-bit quantization
- Fast to apply, doesn't require training data

### Weight-Only Compression

- **INT8 weight-only**: Compressed weights, FP16 activations
- **INT4 weight-only**: Maximum compression
- Best for LLMs on memory-constrained devices

### Dynamic Quantization

- Enabled by default on Intel CPU/GPU
- Quantizes activations on-the-fly

## Using illama with Different Formats

### Pull with specific weight format

```bash
# INT4 (default, recommended)
illama pull microsoft/Phi-4-mini-reasoning --weight-format int4

# INT8
illama pull Qwen/Qwen3-8B --weight-format int8

# FP16
illama pull deepseek-ai/deepseek-coder-6.7b-instruct --weight-format fp16
```

### Bulk sync with format

```bash
python scripts/illama_sync_from_metrics.py \
  --metrics metrics.md \
  --weight-format int4
```

## What About GGUF Quantization?

> [!IMPORTANT]
> GGUF quantization formats (Q4_K_M, IQ3_M, etc.) are **llama.cpp formats**.
> They cannot be used directly with OpenVINO.

If you need GGUF models:
- Use a llama.cpp SYCL stack instead
- Or convert the original HuggingFace model to OpenVINO

Formats **not** supported directly:
- GGUF / llama.cpp Q*_K_*
- llama.cpp IQ* formats
- AWQ (directly, though NNCF can apply similar compression)
- GPTQ (directly, though NNCF can apply similar compression)

## NF4 Support

Recent OpenVINO releases support **NF4** (4-bit NormalFloat) as an FP8 LUT representation.
This may be available through newer optimum-intel versions.

## Verifying Model Format

After pulling a model, check the registry:

```bash
illama list
# Shows model name and weight format

cat /opt/illama/registry/index.json | jq
# Full metadata including weight_format
```

## Performance Impact

Approximate memory usage for a 7B model:

| Format | VRAM Usage | Tokens/sec (Arc B50) |
|--------|------------|----------------------|
| FP16 | ~14 GB | 15-20 |
| INT8 | ~7 GB | 25-35 |
| INT4 | ~3.5 GB | 40-50 |

> [!NOTE]
> Actual performance varies by model architecture and batch size.
> Run `illama doctor` to verify your GPU is properly detected.
