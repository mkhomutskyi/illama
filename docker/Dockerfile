# illama-manager Docker image
# OpenVINO GenAI server for Intel Arc GPUs

FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# =============================================================================
# Stage 1: Setup Intel Repositories for Arc GPUs (B50/B580/A770/A750)
# =============================================================================
RUN apt-get update && apt-get install -y \
    wget gpg software-properties-common python3-pip python3-venv git curl \
    && add-apt-repository -y ppa:kobuk-team/intel-graphics \
    && apt-get update

# =============================================================================
# Stage 2: Install Intel GPU Drivers & Tools
# =============================================================================
RUN apt-get install -y \
    libze-intel-gpu1 libze1 intel-opencl-icd clinfo \
    libtbb12 \
    && rm -rf /var/lib/apt/lists/*

# =============================================================================
# Stage 3: Install Python Dependencies
# =============================================================================
# Using --break-system-packages for Ubuntu 24.04's externally-managed Python
RUN pip3 install --break-system-packages \
    # Server framework
    fastapi \
    uvicorn[standard] \
    pydantic \
    # HuggingFace
    huggingface_hub \
    transformers \
    # OpenVINO (latest versions)
    openvino==2025.4.1 \
    openvino-genai==2025.4.1.0 \
    # Model conversion
    optimum-intel \
    optimum \
    nncf \
    accelerate \
    # Required for gpt-oss and some models
    kernels \
    triton==3.4

# =============================================================================
# Stage 4: Setup Application
# =============================================================================
WORKDIR /app

# Copy application code
COPY illama_manager/ ./illama_manager/
COPY pyproject.toml ./
COPY requirements.txt ./

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
CMD ["python3", "-m", "illama_manager"]
