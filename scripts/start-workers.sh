#!/bin/bash
#
# Start Celery workers for local development
#
# Usage:
#   ./scripts/start-workers.sh [profile]
#
# Profiles:
#   minimal    - Single worker for all queues (default, good for development)
#   full       - All specialized workers (requires more resources)
#   datasets   - Only dataset-related workers (downloads + processing)
#   training   - Only GPU workers (training + extraction)
#

set -e

PROFILE="${1:-minimal}"
BACKEND_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../backend" && pwd)"

cd "$BACKEND_DIR"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Error: Virtual environment not found at $BACKEND_DIR/venv"
    echo "Please create it first: python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt"
    exit 1
fi

# Activate virtual environment
source venv/bin/activate

# Function to start a worker in the background
start_worker() {
    local name=$1
    local queues=$2
    local concurrency=$3
    local extra_args="${4:-}"

    echo "Starting worker: $name (queues: $queues, concurrency: $concurrency)"

    celery -A src.core.celery_app worker \
        -Q "$queues" \
        -c "$concurrency" \
        --loglevel=info \
        --hostname="$name@%h" \
        $extra_args \
        > "logs/worker-$name.log" 2>&1 &

    echo $! > "logs/worker-$name.pid"
    echo "  → Started with PID $(cat logs/worker-$name.pid)"
}

# Create logs directory if it doesn't exist
mkdir -p logs

echo "======================================"
echo "Starting Celery workers (profile: $PROFILE)"
echo "======================================"
echo

case "$PROFILE" in
    minimal)
        echo "Minimal profile: Single worker for all queues"
        start_worker "all-queues" "high_priority,datasets,processing,training,extraction,low_priority" 8
        ;;

    full)
        echo "Full profile: All specialized workers"
        start_worker "high-priority" "high_priority" 8
        start_worker "datasets" "datasets" 4
        start_worker "processing" "processing" 4
        start_worker "training" "training" 1 "--max-memory-per-child=4000000"
        start_worker "extraction" "extraction" 2 "--max-memory-per-child=2000000"
        start_worker "low-priority" "low_priority" 2
        ;;

    datasets)
        echo "Datasets profile: Download and processing workers only"
        start_worker "high-priority" "high_priority" 4
        start_worker "datasets" "datasets" 4
        start_worker "processing" "processing" 4
        ;;

    training)
        echo "Training profile: GPU workers only"
        start_worker "training" "training" 1 "--max-memory-per-child=4000000"
        start_worker "extraction" "extraction" 2 "--max-memory-per-child=2000000"
        ;;

    *)
        echo "Error: Unknown profile '$PROFILE'"
        echo "Available profiles: minimal, full, datasets, training"
        exit 1
        ;;
esac

echo
echo "======================================"
echo "Workers started successfully!"
echo "======================================"
echo
echo "Logs: $BACKEND_DIR/logs/worker-*.log"
echo "PIDs: $BACKEND_DIR/logs/worker-*.pid"
echo
echo "To stop workers: ./scripts/stop-workers.sh"
echo "To view logs: tail -f logs/worker-*.log"
echo
