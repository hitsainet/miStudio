# Backend Dockerfile for miStudio
# Supports both API server (uvicorn) and Celery worker

# Use NVIDIA CUDA base image with Python support
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Set Python environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=utf-8 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies and Python 3.12
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa -y \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-venv \
    python3.12-dev \
    build-essential \
    git \
    curl \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.12 /usr/bin/python3 \
    && ln -sf /usr/bin/python3.12 /usr/bin/python \
    && python3.12 -m ensurepip --upgrade

# Set working directory
WORKDIR /app

# Create non-root user for security (used by entrypoint to drop privileges)
RUN groupadd -r mistudio && useradd -r -g mistudio -s /bin/bash mistudio

# Create initial data directories (will be properly initialized by entrypoint)
# These may be overwritten by volume mounts, which is why entrypoint re-creates them
RUN mkdir -p /data/huggingface_cache /data/models /data/datasets /data/saes \
    /data/activations /data/checkpoints /data/exports /data/trainings /data/tmp \
    && chown -R mistudio:mistudio /data

# Copy requirements first for better layer caching
COPY requirements.txt .

# Install Python dependencies
# Note: PyTorch with CUDA is included in requirements.txt
# Explicitly upgrade pip, setuptools, and wheel to fix CVEs in base image
RUN python3.12 -m pip install --upgrade pip "setuptools>=78.1.1" "wheel>=0.38.1" \
    && python3.12 -m pip install -r requirements.txt

# Pre-install spaCy model for NLP analysis
# This prevents runtime download failures which crash Celery workers
# (spacy.cli.download calls sys.exit(1) on failure, killing the worker process)
RUN python3.12 -m spacy download en_core_web_sm

# Copy application code
COPY --chown=mistudio:mistudio . .

# Copy and set up entrypoint script
COPY docker-entrypoint.sh /docker-entrypoint.sh
RUN chmod +x /docker-entrypoint.sh

# Fix file permissions (Docker preserves host file modes which may be restrictive)
RUN chmod -R u+r,g+r /app && \
    find /app -type d -exec chmod u+x,g+x {} \;

# Expose API port
EXPOSE 8000

# Health check for API service
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:${API_PORT:-8000}/api/v1/system/health || exit 1

# Default environment variables
ENV SERVICE_TYPE=api \
    API_PORT=8000 \
    LOG_LEVEL=INFO \
    DATA_DIR=/data \
    HF_HOME=/data/huggingface_cache \
    HF_CACHE_DIR=/data/huggingface_cache

# Note: Container starts as root, entrypoint drops to mistudio user after initializing data dirs
ENTRYPOINT ["/docker-entrypoint.sh"]
