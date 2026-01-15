#!/bin/bash
set -e

# ==============================================================================
# miStudio Docker Entrypoint
# ==============================================================================
# This script ensures proper initialization before starting the service:
# 1. Creates required data directories if they don't exist
# 2. Fixes ownership/permissions for mounted volumes
# 3. Runs database migrations if requested
# 4. Starts the appropriate service (API, Celery worker, or Celery beat)
# ==============================================================================

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# ==============================================================================
# Data Directory Initialization
# ==============================================================================
# All data subdirectories that miStudio needs
DATA_SUBDIRS=(
    "huggingface_cache"
    "models"
    "datasets"
    "saes"
    "activations"
    "checkpoints"
    "exports"
    "trainings"
    "tmp"
)

DATA_DIR="${DATA_DIR:-/data}"

init_data_directories() {
    log_info "Initializing data directories in ${DATA_DIR}..."

    # Create main data directory if it doesn't exist
    if [ ! -d "$DATA_DIR" ]; then
        log_info "Creating main data directory: ${DATA_DIR}"
        mkdir -p "$DATA_DIR"
    fi

    # Create all subdirectories
    for subdir in "${DATA_SUBDIRS[@]}"; do
        target_dir="${DATA_DIR}/${subdir}"
        if [ ! -d "$target_dir" ]; then
            log_info "Creating directory: ${target_dir}"
            mkdir -p "$target_dir"
        fi
    done

    # Fix ownership - this handles the case where the volume is mounted
    # from the host with different ownership
    # Get the UID/GID of the mistudio user (created in Dockerfile)
    MISTUDIO_UID=$(id -u mistudio 2>/dev/null || echo "1000")
    MISTUDIO_GID=$(id -g mistudio 2>/dev/null || echo "1000")

    log_info "Setting ownership to mistudio (${MISTUDIO_UID}:${MISTUDIO_GID})..."

    # Change ownership of all data directories
    # Use -R to recurse, but be careful with large existing data
    chown -R "${MISTUDIO_UID}:${MISTUDIO_GID}" "$DATA_DIR" 2>/dev/null || {
        log_warn "Could not change ownership of all files (this is OK if some files are in use)"
        # At minimum, ensure the directories themselves are writable
        for subdir in "${DATA_SUBDIRS[@]}"; do
            chown "${MISTUDIO_UID}:${MISTUDIO_GID}" "${DATA_DIR}/${subdir}" 2>/dev/null || true
        done
    }

    # Ensure directories are writable
    chmod 755 "$DATA_DIR"
    for subdir in "${DATA_SUBDIRS[@]}"; do
        chmod 755 "${DATA_DIR}/${subdir}" 2>/dev/null || true
    done

    log_info "Data directories initialized successfully"
}

# ==============================================================================
# Database Migration
# ==============================================================================
# Migrations run automatically for API service to ensure schema is always current.
# For worker/beat services, migrations are skipped by default to avoid race conditions.
# Override with RUN_MIGRATIONS=true/false environment variable.
run_migrations() {
    # Determine if migrations should run:
    # - API service: runs migrations by default (unless RUN_MIGRATIONS=false)
    # - Other services: skip migrations by default (unless RUN_MIGRATIONS=true)
    local should_run="false"

    if [ "$SERVICE_TYPE" = "api" ]; then
        # API service runs migrations by default
        if [ "$RUN_MIGRATIONS" != "false" ]; then
            should_run="true"
        fi
    else
        # Worker/beat services skip migrations by default
        if [ "$RUN_MIGRATIONS" = "true" ]; then
            should_run="true"
        fi
    fi

    if [ "$should_run" = "true" ]; then
        log_info "Running database migrations..."
        # Run as mistudio user, with error handling
        if su -s /bin/bash mistudio -c "cd /app && alembic upgrade head"; then
            log_info "Database migrations completed successfully"
        else
            log_error "Database migrations failed!"
            # For API service, migration failure is fatal
            if [ "$SERVICE_TYPE" = "api" ]; then
                log_error "Cannot start API without successful migrations"
                exit 1
            else
                log_warn "Continuing without migrations (worker/beat may encounter schema issues)"
            fi
        fi
    else
        log_info "Skipping database migrations (RUN_MIGRATIONS=${RUN_MIGRATIONS:-not set}, SERVICE_TYPE=${SERVICE_TYPE})"
    fi
}

# ==============================================================================
# Service Startup
# ==============================================================================
start_service() {
    case "$SERVICE_TYPE" in
        "api")
            log_info "Starting FastAPI server on port ${API_PORT:-8000}..."
            exec su -s /bin/bash mistudio -c "uvicorn src.main:app --host 0.0.0.0 --port ${API_PORT:-8000} ${UVICORN_ARGS:-}"
            ;;
        "celery-worker")
            log_info "Starting Celery worker..."
            # --pool=solo is REQUIRED for CUDA/GPU tasks (fork breaks CUDA initialization)
            exec su -s /bin/bash mistudio -c "celery -A src.core.celery_app worker \
                -Q ${CELERY_QUEUES:-high_priority,datasets,processing,training,extraction,sae,low_priority} \
                -c ${CELERY_CONCURRENCY:-1} \
                --pool=solo \
                --loglevel=${LOG_LEVEL:-info} \
                --hostname=worker@%h \
                --max-tasks-per-child=${CELERY_MAX_TASKS:-100}"
            ;;
        "celery-beat")
            log_info "Starting Celery beat scheduler..."
            exec su -s /bin/bash mistudio -c "celery -A src.core.celery_app beat --loglevel=${LOG_LEVEL:-info} --schedule=/tmp/celerybeat-schedule"
            ;;
        *)
            log_error "Unknown SERVICE_TYPE: $SERVICE_TYPE"
            log_error "Valid options: api, celery-worker, celery-beat"
            exit 1
            ;;
    esac
}

# ==============================================================================
# Main
# ==============================================================================
main() {
    log_info "miStudio Docker Entrypoint - Starting initialization..."
    log_info "Service type: ${SERVICE_TYPE:-api}"
    log_info "Data directory: ${DATA_DIR}"

    # Initialize data directories (runs as root to fix permissions)
    init_data_directories

    # Run migrations (only for API service typically)
    run_migrations

    # Start the service (drops to mistudio user)
    start_service
}

main "$@"
