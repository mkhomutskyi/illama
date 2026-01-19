#!/usr/bin/env python3
"""Bulk model preparation for illama from an Ollama-like metrics markdown table.

- Extracts model names from the first column of markdown rows like:
  | qwen3:14b | OK | 18.04 t/s | 100% GPU | ... |

- Resolves known Ollama names to Hugging Face repos.
- Skips GGUF (llama.cpp) entries by default.
- Calls `illama pull` for supported entries.

Usage:
  python3 illama_sync_from_metrics.py --metrics metrics.md --weight-format int4
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

# Known mappings from your metrics list → HF Transformers repos.
# You can add more mappings here.
OLLAMA_TO_HF = {
    "phi4-reasoning:plus": "microsoft/Phi-4-reasoning-plus",
    "phi4-mini-reasoning:3.8b": "microsoft/Phi-4-mini-reasoning",

    "qwen3:8b": "Qwen/Qwen3-8B",
    "qwen3:14b": "Qwen/Qwen3-14B",
    "qwen3:30b": "Qwen/Qwen3-30B-A3B-Instruct-2507",
    "qwen3-coder:30b": "Qwen/Qwen3-Coder-30B-A3B-Instruct",
    "qwen3-vl:8b": "Qwen/Qwen3-VL-8B-Instruct",

    "deepseek-coder:6.7b": "deepseek-ai/deepseek-coder-6.7b-instruct",
    "deepseek-coder-v2:16b": "deepseek-ai/DeepSeek-Coder-V2-Instruct",
    "deepseek-r1:14b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
    "deepseek-r1:32b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",

    "nemotron-3-nano:30b": "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",

    "olmo2:7b": "allenai/OLMo-2-1124-7B-Instruct",

    "codestral:22b": "mistralai/Codestral-22B-v0.1",
    "mistral-small:24b": "mistralai/Mistral-Small-24B-Instruct-2501",

    "gemma3:4b": "google/gemma-3-4b-it",
    "gemma3:12b": "google/gemma-3-12b-it",

    "granite4:3b": "ibm-granite/granite-4.0-micro",
    "granite-code:20b": "ibm-granite/granite-20b-code-instruct-8k",

    "rnj-1:8b": "EssentialAI/rnj-1-instruct",
}

EMBEDDING_KEYWORDS = ("embed", "embedding", "paraphrase-")


def parse_models_from_md(md_text: str) -> list[str]:
    models: list[str] = []
    for line in md_text.splitlines():
        line = line.strip()
        if not line.startswith("|"):
            continue
        # Grab first cell
        parts = [p.strip() for p in line.strip("|").split("|")]
        if not parts:
            continue
        model = parts[0]
        # skip header separators
        if set(model) <= {"-", ":"}:
            continue
        # looks like a model
        if model:
            models.append(model)
    # de-dupe preserving order
    seen = set()
    out = []
    for m in models:
        if m not in seen:
            out.append(m)
            seen.add(m)
    return out


def is_gguf(model_name: str) -> bool:
    # Your metrics list includes many like:
    # hf.co/<user>/<repo>-GGUF:Q4_K_M
    if model_name.startswith("hf.co/"):
        return True
    if "GGUF" in model_name.upper():
        return True
    return False


def looks_like_hf_repo(model_name: str) -> bool:
    # Allow "Org/Repo:tag" → take Org/Repo
    if "/" in model_name and not model_name.startswith("hf.co/"):
        return True
    return False


def run(cmd: list[str], dry_run: bool = False) -> int:
    print("+", " ".join(cmd))
    if dry_run:
        return 0
    return subprocess.call(cmd)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Bulk model preparation for illama from Ollama metrics table"
    )
    ap.add_argument("--metrics", required=True, help="Path to markdown table file")
    ap.add_argument("--weight-format", default="int4", choices=["fp16", "int8", "int4"])
    ap.add_argument("--include-gguf", action="store_true", help="Download GGUF repos too (won't convert to OpenVINO)")
    ap.add_argument("--include-embeddings", action="store_true", help="Include embedding models")
    ap.add_argument("--dry-run", action="store_true")

    args = ap.parse_args()

    md_path = Path(args.metrics)
    text = md_path.read_text(encoding="utf-8")
    models = parse_models_from_md(text)

    skipped = []
    pulled = []

    for m in models:
        # embeddings
        if (not args.include_embeddings) and any(k in m.lower() for k in EMBEDDING_KEYWORDS):
            skipped.append((m, "embedding-model (skipped by default)"))
            continue

        # GGUF
        if is_gguf(m):
            if not args.include_gguf:
                skipped.append((m, "GGUF/llama.cpp format (OpenVINO cannot import)"))
                continue
            # best-effort: just download the repo referenced after hf.co/
            if m.startswith("hf.co/"):
                repo_and_tag = m[len("hf.co/"):]
                repo = repo_and_tag.split(":", 1)[0]
                # download, don't convert
                rc = run(["huggingface-cli", "download", repo, "--local-dir", f"/opt/illama/cache/gguf/{repo}"], dry_run=args.dry_run)
                if rc != 0:
                    skipped.append((m, "GGUF download failed"))
                else:
                    pulled.append((m, "downloaded GGUF (no conversion)"))
            else:
                skipped.append((m, "GGUF entry not in hf.co/... form"))
            continue

        # Resolve to HF repo
        hf_repo = OLLAMA_TO_HF.get(m)

        # Try interpreting Org/Repo:tag as HF repo if no mapping
        if hf_repo is None and looks_like_hf_repo(m):
            hf_repo = m.split(":", 1)[0]

        if hf_repo is None:
            skipped.append((m, "no HF mapping found"))
            continue

        # Convert + register via illama CLI
        rc = run(["illama", "pull", hf_repo, "--weight-format", args.weight_format], dry_run=args.dry_run)
        if rc != 0:
            skipped.append((m, f"illama pull failed for {hf_repo}"))
        else:
            pulled.append((m, hf_repo))

    print("\n=== SUMMARY ===")
    print(f"Pulled: {len(pulled)}")
    for src, target in pulled:
        print(f"  OK  {src} -> {target}")

    print(f"\nSkipped: {len(skipped)}")
    for src, why in skipped:
        print(f"  SKIP {src}  ({why})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
