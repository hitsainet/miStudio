#!/bin/bash
#
# Celery Process Manager for MechInterp Studio
#
# Proper process management using PID files for reliable start/stop/restart.
# This replaces the need for pkill pattern matching.
#
# Usage:
#   ./celery.sh start    # Start worker and beat
#   ./celery.sh stop     # Gracefully stop all Celery processes
#   ./celery.sh restart  # Stop then start
#   ./celery.sh status   # Show running status
#
# PID Files:
#   /tmp/mistudio-celery-worker.pid
#   /tmp/mistudio-celery-beat.pid
#

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# PID file locations
WORKER_PID_FILE="/tmp/mistudio-celery-worker.pid"
BEAT_PID_FILE="/tmp/mistudio-celery-beat.pid"
BEAT_SCHEDULE_FILE="/tmp/mistudio-celerybeat-schedule"

# Log file locations
WORKER_LOG="/tmp/celery-worker.log"
BEAT_LOG="/tmp/celery-beat.log"

# Queues configuration
QUEUES="high_priority,datasets,processing,training,extraction,sae,low_priority"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Redis configuration (try localhost first, fall back to Docker host)
REDIS_HOST="${REDIS_HOST:-localhost}"
REDIS_PORT="${REDIS_PORT:-6379}"

# Watchdog settings
WATCHDOG_INTERVAL=30  # Check every 30 seconds
WATCHDOG_LOG="/tmp/celery-watchdog.log"
WATCHDOG_PID_FILE="/tmp/mistudio-celery-watchdog.pid"

# Wait for Redis to be available
wait_for_redis() {
    local max_attempts=${1:-30}
    local attempt=1

    echo -n "Waiting for Redis at $REDIS_HOST:$REDIS_PORT..."

    while [ $attempt -le $max_attempts ]; do
        # Try redis-cli first (if installed), then nc, then docker
        if command -v redis-cli &> /dev/null; then
            if redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" ping 2>/dev/null | grep -q "PONG"; then
                echo -e " ${GREEN}connected${NC}"
                return 0
            fi
        elif command -v nc &> /dev/null; then
            if nc -z "$REDIS_HOST" "$REDIS_PORT" 2>/dev/null; then
                echo -e " ${GREEN}connected${NC}"
                return 0
            fi
        else
            # Fall back to Docker if available
            if docker exec mistudio-redis redis-cli ping 2>/dev/null | grep -q "PONG"; then
                echo -e " ${GREEN}connected${NC}"
                return 0
            fi
        fi

        echo -n "."
        sleep 1
        attempt=$((attempt + 1))
    done

    echo -e " ${RED}failed${NC}"
    echo "Redis is not available. Make sure Docker services are running:"
    echo "  docker-compose -f docker-compose.dev.yml up -d redis"
    return 1
}

# Check if worker is healthy (can connect to broker)
is_worker_healthy() {
    if ! is_running "$WORKER_PID_FILE"; then
        return 1
    fi

    # Check if the worker log shows recent activity or no errors
    local log_tail=$(tail -5 "$WORKER_LOG" 2>/dev/null)

    # Check for connection refused errors in recent logs
    if echo "$log_tail" | grep -q "Connection refused\|Cannot connect to"; then
        return 1
    fi

    # Check if process is actually responsive
    local pid=$(get_pid "$WORKER_PID_FILE")
    if ! kill -0 "$pid" 2>/dev/null; then
        return 1
    fi

    return 0
}

# Ensure virtual environment
ensure_venv() {
    if [ -z "$VIRTUAL_ENV" ]; then
        if [ -f "venv/bin/activate" ]; then
            source venv/bin/activate
        else
            echo -e "${RED}Error: Virtual environment not found at ./venv${NC}"
            exit 1
        fi
    fi
}

# Check if process is running from PID file
is_running() {
    local pid_file=$1
    if [ -f "$pid_file" ]; then
        local pid=$(cat "$pid_file" 2>/dev/null)
        if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
            return 0  # Running
        fi
    fi
    return 1  # Not running
}

# Get PID from file
get_pid() {
    local pid_file=$1
    if [ -f "$pid_file" ]; then
        cat "$pid_file" 2>/dev/null
    fi
}

# Stop a process gracefully using PID file
stop_process() {
    local name=$1
    local pid_file=$2
    local timeout=${3:-10}

    if ! is_running "$pid_file"; then
        echo -e "${YELLOW}$name is not running${NC}"
        rm -f "$pid_file" 2>/dev/null
        return 0
    fi

    local pid=$(get_pid "$pid_file")
    echo -n "Stopping $name (PID: $pid)..."

    # Send SIGTERM for graceful shutdown
    kill "$pid" 2>/dev/null

    # Wait for process to exit
    local count=0
    while kill -0 "$pid" 2>/dev/null && [ $count -lt $timeout ]; do
        echo -n "."
        sleep 1
        count=$((count + 1))
    done

    # Force kill if still running
    if kill -0 "$pid" 2>/dev/null; then
        echo -n " forcing..."
        kill -9 "$pid" 2>/dev/null
        sleep 1
    fi

    # Clean up PID file
    rm -f "$pid_file" 2>/dev/null

    if ! kill -0 "$pid" 2>/dev/null; then
        echo -e " ${GREEN}stopped${NC}"
        return 0
    else
        echo -e " ${RED}failed${NC}"
        return 1
    fi
}

# Start Celery worker
start_worker() {
    if is_running "$WORKER_PID_FILE"; then
        local pid=$(get_pid "$WORKER_PID_FILE")
        echo -e "${YELLOW}Worker already running (PID: $pid)${NC}"
        return 0
    fi

    ensure_venv

    # Clean up stale PID file
    rm -f "$WORKER_PID_FILE" 2>/dev/null

    echo -n "Starting Celery worker..."

    # Start worker in background with PID file
    # --pool=solo is REQUIRED for CUDA/GPU tasks
    nohup celery -A src.core.celery_app worker \
        -Q "$QUEUES" \
        -c 1 \
        --pool=solo \
        --loglevel=info \
        --hostname="worker@%h" \
        --max-tasks-per-child=5 \
        --pidfile="$WORKER_PID_FILE" \
        > "$WORKER_LOG" 2>&1 &

    # Wait for PID file to be created
    local count=0
    while [ ! -f "$WORKER_PID_FILE" ] && [ $count -lt 10 ]; do
        sleep 1
        count=$((count + 1))
    done

    if is_running "$WORKER_PID_FILE"; then
        local pid=$(get_pid "$WORKER_PID_FILE")
        echo -e " ${GREEN}started (PID: $pid)${NC}"
        return 0
    else
        echo -e " ${RED}failed${NC}"
        echo "Check logs: $WORKER_LOG"
        return 1
    fi
}

# Start Celery beat
start_beat() {
    if is_running "$BEAT_PID_FILE"; then
        local pid=$(get_pid "$BEAT_PID_FILE")
        echo -e "${YELLOW}Beat already running (PID: $pid)${NC}"
        return 0
    fi

    ensure_venv

    # Clean up stale files
    rm -f "$BEAT_PID_FILE" 2>/dev/null
    rm -f "$BEAT_SCHEDULE_FILE"* 2>/dev/null

    echo -n "Starting Celery beat..."

    # Start beat in background with PID file
    nohup celery -A src.core.celery_app beat \
        --loglevel=info \
        --pidfile="$BEAT_PID_FILE" \
        --schedule="$BEAT_SCHEDULE_FILE" \
        > "$BEAT_LOG" 2>&1 &

    # Wait for PID file to be created
    local count=0
    while [ ! -f "$BEAT_PID_FILE" ] && [ $count -lt 10 ]; do
        sleep 1
        count=$((count + 1))
    done

    if is_running "$BEAT_PID_FILE"; then
        local pid=$(get_pid "$BEAT_PID_FILE")
        echo -e " ${GREEN}started (PID: $pid)${NC}"
        return 0
    else
        echo -e " ${RED}failed${NC}"
        echo "Check logs: $BEAT_LOG"
        return 1
    fi
}

# Command handlers
cmd_start() {
    echo "Starting Celery services..."

    # Wait for Redis first
    if ! wait_for_redis 30; then
        echo -e "${RED}Cannot start Celery without Redis${NC}"
        return 1
    fi

    start_worker
    start_beat
    echo ""
    cmd_status
}

cmd_stop() {
    echo "Stopping Celery services..."
    stop_process "Beat" "$BEAT_PID_FILE" 5
    stop_process "Worker" "$WORKER_PID_FILE" 15

    # Clean up schedule file
    rm -f "$BEAT_SCHEDULE_FILE"* 2>/dev/null
}

cmd_restart() {
    cmd_stop
    echo ""
    sleep 2
    cmd_start
}

cmd_status() {
    echo "Celery Status:"
    echo "=============="

    if is_running "$WORKER_PID_FILE"; then
        local pid=$(get_pid "$WORKER_PID_FILE")
        if is_worker_healthy; then
            echo -e "Worker: ${GREEN}running${NC} (PID: $pid)"
        else
            echo -e "Worker: ${YELLOW}running but unhealthy${NC} (PID: $pid)"
        fi
    else
        echo -e "Worker: ${RED}stopped${NC}"
    fi

    if is_running "$BEAT_PID_FILE"; then
        local pid=$(get_pid "$BEAT_PID_FILE")
        echo -e "Beat:   ${GREEN}running${NC} (PID: $pid)"
    else
        echo -e "Beat:   ${RED}stopped${NC}"
    fi

    if is_running "$WATCHDOG_PID_FILE"; then
        local pid=$(get_pid "$WATCHDOG_PID_FILE")
        echo -e "Watchdog: ${GREEN}running${NC} (PID: $pid)"
    else
        echo -e "Watchdog: ${YELLOW}not running${NC} (use './celery.sh watchdog' to enable)"
    fi

    echo ""
    echo "Logs:"
    echo "  Worker: $WORKER_LOG"
    echo "  Beat:   $BEAT_LOG"
    echo "  Watchdog: $WATCHDOG_LOG"
}

# Watchdog: monitors and auto-restarts failed Celery services
cmd_watchdog() {
    if is_running "$WATCHDOG_PID_FILE"; then
        local pid=$(get_pid "$WATCHDOG_PID_FILE")
        echo -e "${YELLOW}Watchdog already running (PID: $pid)${NC}"
        return 0
    fi

    echo "Starting Celery watchdog (monitoring every ${WATCHDOG_INTERVAL}s)..."

    # Start watchdog in background
    (
        echo $$ > "$WATCHDOG_PID_FILE"

        log_msg() {
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" >> "$WATCHDOG_LOG"
        }

        log_msg "Watchdog started (PID: $$)"

        while true; do
            # Check if Redis is available
            if ! docker exec mistudio-redis redis-cli ping 2>/dev/null | grep -q "PONG"; then
                log_msg "WARNING: Redis not responding, waiting..."
                sleep "$WATCHDOG_INTERVAL"
                continue
            fi

            # Check and restart worker if needed
            if ! is_running "$WORKER_PID_FILE" || ! is_worker_healthy; then
                log_msg "Worker down or unhealthy, restarting..."

                # Stop if running but unhealthy
                if is_running "$WORKER_PID_FILE"; then
                    stop_process "Worker" "$WORKER_PID_FILE" 10 >> "$WATCHDOG_LOG" 2>&1
                fi

                # Wait a moment then restart
                sleep 2
                start_worker >> "$WATCHDOG_LOG" 2>&1

                if is_running "$WORKER_PID_FILE"; then
                    log_msg "Worker restarted successfully"
                else
                    log_msg "ERROR: Failed to restart worker"
                fi
            fi

            # Check and restart beat if needed
            if ! is_running "$BEAT_PID_FILE"; then
                log_msg "Beat scheduler down, restarting..."

                sleep 2
                start_beat >> "$WATCHDOG_LOG" 2>&1

                if is_running "$BEAT_PID_FILE"; then
                    log_msg "Beat restarted successfully"
                else
                    log_msg "ERROR: Failed to restart beat"
                fi
            fi

            sleep "$WATCHDOG_INTERVAL"
        done
    ) &

    disown

    sleep 2
    if is_running "$WATCHDOG_PID_FILE"; then
        local pid=$(get_pid "$WATCHDOG_PID_FILE")
        echo -e "Watchdog ${GREEN}started${NC} (PID: $pid)"
        echo "  Log: $WATCHDOG_LOG"
    else
        echo -e "Watchdog ${RED}failed to start${NC}"
        return 1
    fi
}

cmd_stop_watchdog() {
    if ! is_running "$WATCHDOG_PID_FILE"; then
        echo -e "${YELLOW}Watchdog is not running${NC}"
        return 0
    fi

    stop_process "Watchdog" "$WATCHDOG_PID_FILE" 5
}

# Main
case "${1:-}" in
    start)
        cmd_start
        ;;
    stop)
        cmd_stop_watchdog 2>/dev/null || true
        cmd_stop
        ;;
    restart)
        cmd_stop_watchdog 2>/dev/null || true
        cmd_restart
        ;;
    status)
        cmd_status
        ;;
    watchdog)
        # Start services if not running, then start watchdog
        if ! is_running "$WORKER_PID_FILE" || ! is_running "$BEAT_PID_FILE"; then
            cmd_start
        fi
        cmd_watchdog
        ;;
    stop-watchdog)
        cmd_stop_watchdog
        ;;
    *)
        echo "Usage: $0 {start|stop|restart|status|watchdog|stop-watchdog}"
        echo ""
        echo "Commands:"
        echo "  start         Start worker and beat (waits for Redis)"
        echo "  stop          Stop all Celery services"
        echo "  restart       Restart all services"
        echo "  status        Show service status"
        echo "  watchdog      Start services + auto-restart monitor"
        echo "  stop-watchdog Stop the watchdog monitor only"
        exit 1
        ;;
esac
