# LLM Server Dockerfile
# Single-stage build for simplicity and reliability

FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=UTF-8 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    python3-dev \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3 /usr/bin/python

WORKDIR /app

# Upgrade pip and install build tools
RUN pip install --upgrade pip setuptools wheel packaging

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create non-root user for security
RUN useradd -m -u 1000 llm && \
    mkdir -p /data/ollm_models /data/ollm_cache && \
    chown -R llm:llm /app /data

# Copy application code
COPY --chown=llm:llm . /app/services/ollm_server/

# Switch to non-root user
USER llm

# Expose port
EXPOSE 11434

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:11434/health || exit 1

# Set Python path
ENV PYTHONPATH=/app

# Run the server
CMD ["python", "-m", "services.ollm_server.main"]
