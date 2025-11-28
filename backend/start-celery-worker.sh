#!/bin/bash
#
# Celery Worker Startup Script
#
# This script ensures Celery workers listen to all required queues by default.
#
# Usage:
#   ./start-celery-worker.sh                    # Start with all queues (development default)
#   ./start-celery-worker.sh datasets           # Start with specific queue(s)
#   ./start-celery-worker.sh datasets,processing # Multiple queues
#
# For production deployment with systemd or Docker, use the scripts/start-workers.sh
# or docker-compose.workers.yml for proper multi-worker configuration.
#

# Default to all queues if no argument provided
QUEUES="${1:-high_priority,datasets,processing,training,extraction,sae,low_priority}"
CONCURRENCY="${2:-1}"  # Reduced to 1 for GPU memory safety

# Navigate to backend directory
cd "$(dirname "$0")"

# Check if virtual environment is activated
if [ -z "$VIRTUAL_ENV" ]; then
    echo "⚠️  Virtual environment not activated. Activating..."
    if [ -f "venv/bin/activate" ]; then
        source venv/bin/activate
    else
        echo "❌ Error: Virtual environment not found at ./venv"
        echo "Please create it first: python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt"
        exit 1
    fi
fi

echo "======================================"
echo "Starting Celery Worker"
echo "======================================"
echo "Queues:      $QUEUES"
echo "Concurrency: $CONCURRENCY"
echo "Working Dir: $(pwd)"
echo "======================================"
echo ""

# Start Celery worker with explicit queue configuration
# IMPORTANT: --max-tasks-per-child=5 ensures worker processes restart periodically
# This is CRITICAL for GPU memory cleanup while allowing longer tasks to complete
# Setting to 5 instead of 1 prevents worker restart during task execution (e.g., during
# activation concatenation which happens at 90% progress)
celery -A src.core.celery_app worker \
    -Q "$QUEUES" \
    -c "$CONCURRENCY" \
    --loglevel=info \
    --hostname="worker@%h" \
    --max-tasks-per-child=5
