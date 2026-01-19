"""illama CLI - Ollama-like command-line interface for Intel Arc GPUs.

USAGE:
  illama [global-options] <command> [args]

GLOBAL OPTIONS:
  --host <url>             illama-manager base URL (default: http://localhost:11434)
  --registry-dir <path>    local registry dir (default: /opt/illama/registry)
  --cache-dir <path>       local cache dir (default: /opt/illama/cache)
  --device <CPU|GPU|AUTO>  override device (default: GPU)
  --json                   machine-readable output
  -h, --help               show help

COMMANDS:
  pull <model> [flags]     download + convert + register a model
  rm <model>               remove from registry (and delete artifacts)
  list                     list models in registry
  ps                       show loaded model + runtime stats
  run <model> [prompt]     ensure model loaded, then run a prompt
  serve                    run a local dev server (optional)
  doctor                   print system checks (GPU, drivers, OpenVINO)
"""

from __future__ import annotations

import json as json_lib
import os
import shutil
import subprocess
import sys
from pathlib import Path

import click

from . import __version__
from .client import IllamaClient

# Default paths
DEFAULT_HOST = os.getenv("ILLAMA_HOST", "http://localhost:11434")
DEFAULT_REGISTRY_DIR = os.getenv("ILLAMA_REGISTRY_DIR", "/opt/illama/registry")
DEFAULT_CACHE_DIR = os.getenv("ILLAMA_CACHE_DIR", "/opt/illama/cache")
DEFAULT_DEVICE = os.getenv("ILLAMA_DEVICE", "GPU")


@click.group()
@click.version_option(__version__, prog_name="illama")
@click.option("--host", default=DEFAULT_HOST, help="illama-manager base URL")
@click.option("--registry-dir", default=DEFAULT_REGISTRY_DIR, help="Local registry dir")
@click.option("--cache-dir", default=DEFAULT_CACHE_DIR, help="Local cache dir")
@click.option("--device", default=DEFAULT_DEVICE, type=click.Choice(["CPU", "GPU", "AUTO"]))
@click.option("--json", "json_output", is_flag=True, help="Machine-readable output")
@click.pass_context
def cli(ctx, host, registry_dir, cache_dir, device, json_output):
    """illama (intel lamma) — OpenVINO LLM manager for Intel GPU"""
    ctx.ensure_object(dict)
    ctx.obj["host"] = host
    ctx.obj["registry_dir"] = Path(registry_dir)
    ctx.obj["cache_dir"] = Path(cache_dir)
    ctx.obj["device"] = device
    ctx.obj["json"] = json_output
    ctx.obj["client"] = IllamaClient(host)


@cli.command()
@click.argument("model")
@click.option("--source", default="hf", type=click.Choice(["hf", "ollama"]))
@click.option("--hf-repo", help="HuggingFace repo id (overrides model name)")
@click.option("--revision", help="HF revision/branch")
@click.option("--weight-format", default="int4", type=click.Choice(["fp16", "int8", "int4"]))
@click.option("--include-embeddings", is_flag=True, help="Allow embedding models")
@click.option("--cleanup/--no-cleanup", default=True, help="Remove temp files after success")
@click.pass_context
def pull(ctx, model, source, hf_repo, revision, weight_format, include_embeddings, cleanup):
    """Download, convert, and register a model."""
    registry_dir = ctx.obj["registry_dir"]
    cache_dir = ctx.obj["cache_dir"]
    device = ctx.obj["device"]

    # Resolve repo
    repo = hf_repo or model

    click.echo(f"Pulling {repo} with {weight_format} weight format...")

    # Create directories
    registry_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Download from HuggingFace
    download_dir = cache_dir / "downloads" / repo.replace("/", "_")
    click.echo(f"Downloading to {download_dir}...")

    cmd = ["huggingface-cli", "download", repo, "--local-dir", str(download_dir)]
    if revision:
        cmd.extend(["--revision", revision])

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        click.echo(f"Download failed: {result.stderr}", err=True)
        sys.exit(1)

    click.echo("Download complete. Converting to OpenVINO IR...")

    # Convert using optimum-intel
    model_name = repo.split("/")[-1]
    output_dir = registry_dir / model_name

    convert_cmd = [
        "optimum-cli",
        "export",
        "openvino",
        "--model",
        str(download_dir),
        "--weight-format",
        weight_format,
        str(output_dir),
    ]

    result = subprocess.run(convert_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        click.echo(f"Conversion failed: {result.stderr}", err=True)
        sys.exit(1)

    click.echo(f"Conversion complete. Model saved to {output_dir}")

    # Update registry index
    index_file = registry_dir / "index.json"
    if index_file.exists():
        index = json_lib.loads(index_file.read_text())
    else:
        index = {}

    from datetime import datetime

    index[model_name] = {
        "name": model_name,
        "hf_repo": repo,
        "weight_format": weight_format,
        "family": "text",
        "local_path": str(output_dir),
        "created_at": datetime.utcnow().isoformat(),
    }

    index_file.write_text(json_lib.dumps(index, indent=2))

    # Cleanup
    if cleanup and download_dir.exists():
        click.echo("Cleaning up download cache...")
        shutil.rmtree(download_dir)

    if ctx.obj["json"]:
        click.echo(json_lib.dumps({"success": True, "model": model_name, "path": str(output_dir)}))
    else:
        click.echo(f"✓ Model {model_name} registered successfully!")


@cli.command()
@click.argument("model")
@click.pass_context
def rm(ctx, model):
    """Remove a model from the registry."""
    registry_dir = ctx.obj["registry_dir"]
    index_file = registry_dir / "index.json"

    if not index_file.exists():
        click.echo("Registry is empty.", err=True)
        sys.exit(1)

    index = json_lib.loads(index_file.read_text())

    if model not in index:
        click.echo(f"Model {model} not found in registry.", err=True)
        sys.exit(1)

    model_info = index.pop(model)
    index_file.write_text(json_lib.dumps(index, indent=2))

    # Delete artifacts
    if model_info.get("local_path"):
        model_path = Path(model_info["local_path"])
        if model_path.exists():
            shutil.rmtree(model_path)
            click.echo(f"Deleted artifacts: {model_path}")

    if ctx.obj["json"]:
        click.echo(json_lib.dumps({"success": True, "model": model}))
    else:
        click.echo(f"✓ Removed {model}")


@cli.command(name="list")
@click.pass_context
def list_models(ctx):
    """List models in the registry."""
    client = ctx.obj["client"]

    try:
        models = client.list_models()
    except Exception as e:
        # Fallback to local registry
        registry_dir = ctx.obj["registry_dir"]
        index_file = registry_dir / "index.json"
        if index_file.exists():
            index = json_lib.loads(index_file.read_text())
            models = [{"id": name, **info} for name, info in index.items()]
        else:
            models = []

    if ctx.obj["json"]:
        click.echo(json_lib.dumps(models, indent=2))
    else:
        if not models:
            click.echo("No models registered.")
        else:
            click.echo("NAME\t\t\tSIZE\t\tMODIFIED")
            for m in models:
                name = m.get("id", m.get("name", "unknown"))
                click.echo(f"{name}")


@cli.command()
@click.pass_context
def ps(ctx):
    """Show loaded model and runtime stats."""
    client = ctx.obj["client"]

    try:
        status = client.process_status()
    except Exception as e:
        click.echo(f"Could not connect to illama-manager: {e}", err=True)
        sys.exit(1)

    if ctx.obj["json"]:
        click.echo(json_lib.dumps(status, indent=2))
    else:
        if not status.get("loaded"):
            click.echo("No model currently loaded.")
        else:
            click.echo(f"Model: {status['model']}")
            click.echo(f"Loaded at: {status['loaded_at']}")
            click.echo(f"Idle: {status['idle_seconds']:.1f}s")


@cli.command()
@click.argument("model")
@click.argument("prompt", required=False)
@click.option("--max-tokens", default=4096, type=int)
@click.option("--temperature", default=0.7, type=float)
@click.pass_context
def run(ctx, model, prompt, max_tokens, temperature):
    """Ensure model loaded, then run a prompt."""
    client = ctx.obj["client"]

    if not prompt:
        # Interactive mode
        click.echo(f"Loading {model}... (type 'exit' to quit)")
        while True:
            try:
                prompt = click.prompt(">>> ", prompt_suffix="")
            except (EOFError, KeyboardInterrupt):
                break
            if prompt.lower() in ("exit", "quit", "/bye"):
                break

            try:
                response = client.chat(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    temperature=temperature,
                    stream=True,
                )
                for token in response:
                    click.echo(token, nl=False)
                click.echo()
            except Exception as e:
                click.echo(f"Error: {e}", err=True)
    else:
        # Single prompt mode
        try:
            response = client.chat(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
                stream=True,
            )
            for token in response:
                click.echo(token, nl=False)
            click.echo()
        except Exception as e:
            click.echo(f"Error: {e}", err=True)
            sys.exit(1)


@cli.command()
@click.pass_context
def doctor(ctx):
    """Print system checks (GPU, drivers, OpenVINO)."""
    checks = []

    # Check Level Zero
    try:
        result = subprocess.run(["zeinfo"], capture_output=True, text=True)
        if result.returncode == 0:
            checks.append(("Level Zero", "OK", result.stdout.split("\n")[0] if result.stdout else ""))
        else:
            checks.append(("Level Zero", "NOT FOUND", "zeinfo command failed"))
    except FileNotFoundError:
        checks.append(("Level Zero", "NOT FOUND", "Install: apt install level-zero"))

    # Check intel_gpu_top
    try:
        result = subprocess.run(["which", "intel_gpu_top"], capture_output=True, text=True)
        if result.returncode == 0:
            checks.append(("intel_gpu_top", "OK", result.stdout.strip()))
        else:
            checks.append(("intel_gpu_top", "NOT FOUND", "Install: apt install intel-gpu-tools"))
    except FileNotFoundError:
        checks.append(("intel_gpu_top", "NOT FOUND", "Install: apt install intel-gpu-tools"))

    # Check OpenVINO
    try:
        import openvino

        checks.append(("OpenVINO", "OK", openvino.__version__))
    except ImportError:
        checks.append(("OpenVINO", "NOT FOUND", "Install: pip install openvino"))

    # Check OpenVINO GenAI
    try:
        import openvino_genai

        checks.append(("OpenVINO GenAI", "OK", getattr(openvino_genai, "__version__", "installed")))
    except ImportError:
        checks.append(("OpenVINO GenAI", "NOT FOUND", "Install: pip install openvino-genai"))

    # Check optimum-intel
    try:
        import optimum.intel

        checks.append(("optimum-intel", "OK", getattr(optimum.intel, "__version__", "installed")))
    except ImportError:
        checks.append(("optimum-intel", "NOT FOUND", "Install: pip install optimum[openvino]"))

    # Check huggingface-cli
    try:
        result = subprocess.run(["huggingface-cli", "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            checks.append(("huggingface-cli", "OK", result.stdout.strip()))
        else:
            checks.append(("huggingface-cli", "ERROR", "Command failed"))
    except FileNotFoundError:
        checks.append(("huggingface-cli", "NOT FOUND", "Install: pip install huggingface_hub"))

    # Check illama-manager connectivity
    client = ctx.obj["client"]
    try:
        health = client.health()
        checks.append(("illama-manager", "OK", f"v{health.get('version', 'unknown')}"))
    except Exception as e:
        checks.append(("illama-manager", "NOT REACHABLE", str(e)[:50]))

    # Output
    if ctx.obj["json"]:
        click.echo(json_lib.dumps([{"name": n, "status": s, "info": i} for n, s, i in checks], indent=2))
    else:
        click.echo("COMPONENT\t\tSTATUS\t\tINFO")
        click.echo("-" * 60)
        for name, status, info in checks:
            icon = "✓" if status == "OK" else "✗"
            click.echo(f"{icon} {name:<20}\t{status:<12}\t{info}")


@cli.command()
@click.option("--host", default="0.0.0.0")
@click.option("--port", default=11434, type=int)
@click.pass_context
def serve(ctx, host, port):
    """Run a local development server."""
    click.echo(f"Starting illama-manager on {host}:{port}...")
    try:
        import uvicorn

        from illama_manager.server import app

        uvicorn.run(app, host=host, port=port)
    except ImportError:
        click.echo("illama_manager package not found. Install with: pip install -e .", err=True)
        sys.exit(1)


if __name__ == "__main__":
    cli()
