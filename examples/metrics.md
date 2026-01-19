# Example: Ollama Metrics Table

This is an example metrics table that can be used with `illama_sync_from_metrics.py`.

| Model | Status | Speed | GPU Usage | Notes |
|-------|--------|-------|-----------|-------|
| phi4-reasoning:plus | OK | 15.95 t/s | 100% GPU | |
| phi4-mini-reasoning:3.8b | OK | 48.68 t/s | 100% GPU | |
| qwen3:14b | OK | 18.04 t/s | 100% GPU | |
| qwen3:8b | OK | 35.12 t/s | 100% GPU | |
| deepseek-coder:6.7b | OK | 43.21 t/s | 100% GPU | |
| gemma3:4b | OK | 52.30 t/s | 100% GPU | Gated |
| olmo2:7b | OK | 38.45 t/s | 100% GPU | |

## Usage

```bash
python scripts/illama_sync_from_metrics.py \
  --metrics examples/metrics.md \
  --weight-format int4 \
  --dry-run
```
