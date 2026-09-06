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
# Note: We add the deadsnakes PPA directly instead of using add-apt-repository
# to avoid Launchpad API calls which can fail during outages
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    gnupg \
    ca-certificates \
    curl \
    # SCOPED KEYRING, NOT THE GLOBAL ONE (MIS-E2E-148).
    #
    # `apt-key adv` puts the deadsnakes key in the global trusted keyring, where
    # it is trusted to sign packages from EVERY repository. `signed-by=` binds
    # it to this source alone, so a compromise of the deadsnakes key cannot
    # authorise a package from anywhere else. (`apt-key` is also deprecated and
    # slated for removal.)
    #
    # FETCHED OVER HTTPS, NOT FROM A KEYSERVER. The first version of this used
    # `gpg --keyserver … --recv-keys`, which broke the build: `--recv-keys`
    # needs `dirmngr` (not installed) and a writable `/root/.gnupg` (absent in
    # the buildkit sandbox) — "gpg: keyserver receive failed: No dirmngr".
    # An HTTPS fetch piped through `--dearmor` needs neither, and does not
    # depend on the keyserver's HKP port being reachable from the runner.
    && curl -fsSL "https://keyserver.ubuntu.com/pks/lookup?op=get&search=0xF23C5A6CF475977595C89F51BA6932366A755776" \
        | gpg --dearmor -o /usr/share/keyrings/deadsnakes.gpg \
    && echo "deb [signed-by=/usr/share/keyrings/deadsnakes.gpg] https://ppa.launchpadcontent.net/deadsnakes/ppa/ubuntu jammy main" \
        > /etc/apt/sources.list.d/deadsnakes.list \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-venv \
    python3.12-dev \
    build-essential \
    git \
    curl \
    libpq-dev \
    && apt-get upgrade -y gnupg gnupg2 \
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
# Explicitly upgrade pip, setuptools, and wheel to fix CVEs in base image.
# setuptools>=82 bundles vendored wheel>=0.46.3, closing CVE-2026-24049
# (path traversal in wheel <0.46.2) at the vendored copy Scout flags.
RUN python3.12 -m pip install --upgrade pip "setuptools>=82.0.1" "wheel>=0.46.2" \
    && python3.12 -m pip install -r requirements.txt

# Pre-install spaCy model for NLP analysis
# This prevents runtime download failures which crash Celery workers
# (spacy.cli.download calls sys.exit(1) on failure, killing the worker process)
RUN python3.12 -m spacy download en_core_web_sm

# Copy application code
COPY --chown=mistudio:mistudio . .

# THE VERSION MUST BE IN THE IMAGE (MIS-E2E-028).
#
# `GET /api/v1/version` returned {"version":"unknown"} on every deployed pod.
# The build context is `backend/` and `VERSION` lives at the REPO ROOT, so it
# never entered the image and the endpoint's three candidate paths all missed.
# The endpoint that exists to answer "which build is this?" could not answer it
# in the only environment where the question matters.
#
# An ARG rather than a COPY: the file is outside this build context, so a COPY
# cannot reach it without moving the context. The build passes it in; the
# entrypoint has no work to do.
ARG APP_VERSION=unknown
ENV MISTUDIO_VERSION=${APP_VERSION}
RUN printf '%s' "${APP_VERSION}" > /app/VERSION

# Copy LICENSE file for open source compliance
COPY --chown=mistudio:mistudio LICENSE /app/LICENSE

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
    BACKEND_DIR=/app \
    HF_HOME=/data/huggingface_cache \
    HF_CACHE_DIR=/data/huggingface_cache

# Note: Container starts as root, entrypoint drops to mistudio user after initializing data dirs
ENTRYPOINT ["/docker-entrypoint.sh"]
