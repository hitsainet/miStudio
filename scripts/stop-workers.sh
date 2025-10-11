#!/bin/bash
#
# Stop all Celery workers started by start-workers.sh
#

set -e

BACKEND_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../backend" && pwd)"

cd "$BACKEND_DIR"

echo "======================================"
echo "Stopping Celery workers"
echo "======================================"
echo

if [ ! -d "logs" ]; then
    echo "No logs directory found. No workers to stop."
    exit 0
fi

# Find all PID files
PID_FILES=$(find logs -name "worker-*.pid" 2>/dev/null || true)

if [ -z "$PID_FILES" ]; then
    echo "No worker PID files found. No workers to stop."
    exit 0
fi

# Stop each worker
for pid_file in $PID_FILES; do
    if [ -f "$pid_file" ]; then
        pid=$(cat "$pid_file")
        worker_name=$(basename "$pid_file" .pid)

        if ps -p "$pid" > /dev/null 2>&1; then
            echo "Stopping $worker_name (PID: $pid)"
            kill "$pid" 2>/dev/null || true

            # Wait up to 10 seconds for graceful shutdown
            timeout=10
            while ps -p "$pid" > /dev/null 2>&1 && [ $timeout -gt 0 ]; do
                sleep 1
                timeout=$((timeout - 1))
            done

            # Force kill if still running
            if ps -p "$pid" > /dev/null 2>&1; then
                echo "  → Force killing $worker_name"
                kill -9 "$pid" 2>/dev/null || true
            else
                echo "  → Stopped gracefully"
            fi
        else
            echo "Worker $worker_name (PID: $pid) not running"
        fi

        rm -f "$pid_file"
    fi
done

echo
echo "======================================"
echo "All workers stopped"
echo "======================================"
