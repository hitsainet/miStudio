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

# Install system dependencies and Python 3.11
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa -y \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    python3-pip \
    build-essential \
    git \
    curl \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3 \
    && ln -sf /usr/bin/python3.11 /usr/bin/python

# Set working directory
WORKDIR /app

# Create non-root user for security
RUN groupadd -r mistudio && useradd -r -g mistudio mistudio

# Create data directories
RUN mkdir -p /data/huggingface_cache /data/models /data/datasets /data/saes \
    && chown -R mistudio:mistudio /data

# Copy requirements first for better layer caching
COPY requirements.txt .

# Install Python dependencies
# Note: PyTorch with CUDA is included in requirements.txt
RUN python3.11 -m pip install --upgrade pip setuptools wheel \
    && python3.11 -m pip install -r requirements.txt

# Copy application code
COPY --chown=mistudio:mistudio . .

# Fix file permissions (Docker preserves host file modes which may be restrictive)
RUN chmod -R u+r,g+r /app && \
    find /app -type d -exec chmod u+x,g+x {} \;

# Create entrypoint script
RUN echo '#!/bin/bash\n\
set -e\n\
\n\
# Run database migrations\n\
if [ "$RUN_MIGRATIONS" = "true" ]; then\n\
    echo "Running database migrations..."\n\
    alembic upgrade head\n\
fi\n\
\n\
# Determine which service to run\n\
case "$SERVICE_TYPE" in\n\
    "api")\n\
        echo "Starting FastAPI server..."\n\
        exec uvicorn src.main:app --host 0.0.0.0 --port ${API_PORT:-8000} ${UVICORN_ARGS:-}\n\
        ;;\n\
    "celery-worker")\n\
        echo "Starting Celery worker..."\n\
        exec celery -A src.core.celery_app worker \\\n\
            -Q ${CELERY_QUEUES:-high_priority,datasets,processing,training,extraction,sae,low_priority} \\\n\
            -c ${CELERY_CONCURRENCY:-1} \\\n\
            --loglevel=${LOG_LEVEL:-info} \\\n\
            --hostname=worker@%h \\\n\
            --max-tasks-per-child=${CELERY_MAX_TASKS:-5}\n\
        ;;\n\
    "celery-beat")\n\
        echo "Starting Celery beat scheduler..."\n\
        exec celery -A src.core.celery_app beat --loglevel=${LOG_LEVEL:-info} --schedule=/tmp/celerybeat-schedule\n\
        ;;\n\
    *)\n\
        echo "Unknown SERVICE_TYPE: $SERVICE_TYPE"\n\
        echo "Valid options: api, celery-worker, celery-beat"\n\
        exit 1\n\
        ;;\n\
esac\n\
' > /entrypoint.sh && chmod +x /entrypoint.sh

# Switch to non-root user
USER mistudio

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

ENTRYPOINT ["/entrypoint.sh"]
