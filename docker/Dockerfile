# illama-manager Docker image
# OpenVINO GenAI server for Intel Arc GPUs

# =============================================================================
# Stage 1: Base with OpenVINO runtime
# =============================================================================
FROM ubuntu:24.04 AS base

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    curl \
    wget \
    git \
    # Intel GPU support
    intel-opencl-icd \
    intel-level-zero-gpu \
    level-zero \
    # Build essentials (for some pip packages)
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Intel GPU compute runtime (for Arc support)
# Note: This may need adjustment based on your Ubuntu version
RUN wget -qO - https://repositories.intel.com/gpu/intel-graphics.key | \
    gpg --dearmor -o /usr/share/keyrings/intel-graphics.gpg && \
    echo "deb [arch=amd64 signed-by=/usr/share/keyrings/intel-graphics.gpg] https://repositories.intel.com/gpu/ubuntu noble unified" | \
    tee /etc/apt/sources.list.d/intel-gpu.list && \
    apt-get update && apt-get install -y --no-install-recommends \
    intel-opencl-icd \
    intel-level-zero-gpu \
    level-zero \
    && rm -rf /var/lib/apt/lists/*

# =============================================================================
# Stage 2: Python dependencies
# =============================================================================
FROM base AS deps

WORKDIR /app

# Create virtual environment
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# =============================================================================
# Stage 3: Application
# =============================================================================
FROM deps AS app

WORKDIR /app

# Copy application code
COPY illama_manager/ ./illama_manager/
COPY pyproject.toml ./

# Install the package
RUN pip install --no-cache-dir -e .

# Create data directories
RUN mkdir -p /data/registry /data/cache /data/logs

# Environment defaults
ENV ILLAMA_DEVICE=GPU
ENV ILLAMA_ONE_MODEL=1
ENV ILLAMA_IDLE_TTL_SEC=600
ENV ILLAMA_REGISTRY_DIR=/data/registry
ENV ILLAMA_CACHE_DIR=/data/cache
ENV ILLAMA_LOG_LEVEL=INFO
ENV ILLAMA_HOST=0.0.0.0
ENV ILLAMA_PORT=11434

# Expose API port
EXPOSE 11434

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:11434/health || exit 1

# Run the server
CMD ["python", "-m", "illama_manager"]
