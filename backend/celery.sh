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
STEERING_PID_FILE="/tmp/mistudio-celery-steering.pid"
BEAT_PID_FILE="/tmp/mistudio-celery-beat.pid"
BEAT_SCHEDULE_FILE="/tmp/mistudio-celerybeat-schedule"

# Log file locations
WORKER_LOG="/tmp/celery-worker.log"
STEERING_LOG="/tmp/celery-steering.log"
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

# Memory thresholds for proactive restart (percentage)
SYSTEM_RAM_THRESHOLD=90      # Restart if system RAM usage exceeds 90%
WORKER_RAM_THRESHOLD_MB=12000 # Restart if worker uses more than 12GB RAM
GPU_MEMORY_THRESHOLD=90      # Restart if GPU memory exceeds 90%

# Get system RAM usage percentage
get_system_ram_usage() {
    # Returns RAM usage as integer percentage (0-100)
    free | awk '/Mem:/ {printf "%.0f", $3/$2 * 100}'
}

# Get worker process RAM usage in MB
get_worker_ram_mb() {
    local pid_file=$1
    if [ -f "$pid_file" ]; then
        local pid=$(cat "$pid_file" 2>/dev/null)
        if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
            # Get RSS (Resident Set Size) in KB, convert to MB
            local rss_kb=$(ps -o rss= -p "$pid" 2>/dev/null | tr -d ' ')
            if [ -n "$rss_kb" ]; then
                echo $((rss_kb / 1024))
                return 0
            fi
        fi
    fi
    echo "0"
}

# Get GPU memory usage percentage (returns 0 if no GPU or nvidia-smi unavailable)
get_gpu_memory_usage() {
    if command -v nvidia-smi &> /dev/null; then
        # Get memory used percentage for GPU 0
        nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 | awk -F', ' '{
            if ($2 > 0) printf "%.0f", $1/$2 * 100
            else print "0"
        }'
    else
        echo "0"
    fi
}

# Check if worker needs restart due to memory pressure
# Returns 0 and prints reason if pressure detected, otherwise returns 1
check_memory_pressure() {
    local pid_file=$1

    # Check system RAM (with default to 0 if command fails)
    local sys_ram
    sys_ram=$(get_system_ram_usage 2>/dev/null) || sys_ram=0
    sys_ram=${sys_ram:-0}
    if [ "$sys_ram" -ge "$SYSTEM_RAM_THRESHOLD" ] 2>/dev/null; then
        echo "System RAM at ${sys_ram}% (threshold: ${SYSTEM_RAM_THRESHOLD}%)"
        return 0
    fi

    # Check worker RAM (with default to 0 if command fails)
    local worker_ram
    worker_ram=$(get_worker_ram_mb "$pid_file" 2>/dev/null) || worker_ram=0
    worker_ram=${worker_ram:-0}
    if [ "$worker_ram" -ge "$WORKER_RAM_THRESHOLD_MB" ] 2>/dev/null; then
        echo "Worker RAM at ${worker_ram}MB (threshold: ${WORKER_RAM_THRESHOLD_MB}MB)"
        return 0
    fi

    # Check GPU memory (with default to 0 if command fails)
    local gpu_mem
    gpu_mem=$(get_gpu_memory_usage 2>/dev/null) || gpu_mem=0
    gpu_mem=${gpu_mem:-0}
    if [ "$gpu_mem" -ge "$GPU_MEMORY_THRESHOLD" ] 2>/dev/null; then
        echo "GPU memory at ${gpu_mem}% (threshold: ${GPU_MEMORY_THRESHOLD}%)"
        return 0
    fi

    return 1  # No memory pressure
}

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
# Uses SIGTERM and waits for children to be reaped before returning
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
    local pgid=$(ps -o pgid= -p "$pid" 2>/dev/null | tr -d ' ')
    echo -n "Stopping $name (PID: $pid, PGID: $pgid)..."

    # Send SIGTERM for graceful shutdown
    kill "$pid" 2>/dev/null

    # Wait for process AND its children to exit (prevents zombies)
    local count=0
    while [ $count -lt $timeout ]; do
        # Check if main process is still running
        if ! kill -0 "$pid" 2>/dev/null; then
            # Main process stopped, now wait for any children in same process group
            if [ -n "$pgid" ]; then
                local children=$(pgrep -g "$pgid" 2>/dev/null | wc -l)
                if [ "$children" -eq 0 ]; then
                    break  # All children reaped
                fi
                echo -n "c"  # Waiting for children
            else
                break
            fi
        else
            echo -n "."
        fi
        sleep 1
        count=$((count + 1))
    done

    # If still running after timeout, try SIGTERM on process group first
    if kill -0 "$pid" 2>/dev/null; then
        echo -n " (SIGTERM to group)..."
        if [ -n "$pgid" ]; then
            kill -TERM -"$pgid" 2>/dev/null || true
        fi
        sleep 3
    fi

    # Only SIGKILL as absolute last resort
    if kill -0 "$pid" 2>/dev/null; then
        echo -n " (SIGKILL)..."
        kill -9 "$pid" 2>/dev/null
        sleep 2
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
    #
    # IMPORTANT: --pool=solo is REQUIRED for CUDA/GPU tasks
    # Without solo pool, Celery uses fork() which breaks CUDA initialization
    # Error: "Cannot re-initialize CUDA in forked subprocess"
    #
    # This happens because CUDA maintains state in the parent process that
    # cannot be safely inherited by forked child processes. The solo pool
    # runs tasks in the main process, avoiding the fork altogether.
    #
    # Trade-off: With --pool=solo, concurrency (-c) is effectively 1 since
    # all tasks run sequentially in the main process. For GPU tasks this is
    # usually desired anyway to avoid GPU memory contention.
    #
    # --max-tasks-per-child=100: Restarts worker after 100 tasks to prevent
    # memory leaks from accumulating. With solo pool, this triggers a full
    # worker restart, which also cleans up GPU memory.
    #
    nohup celery -A src.core.celery_app worker \
        -Q "$QUEUES" \
        -c 1 \
        --pool=solo \
        --loglevel=info \
        --hostname="worker@%h" \
        --max-tasks-per-child=100 \
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

# Start Celery steering worker (dedicated GPU worker for steering operations)
#
# This worker runs in a separate process from the main worker because:
# 1. Steering tasks have long timeouts (150s soft, 180s hard)
# 2. Needs aggressive worker recycling (--max-tasks-per-child=50)
# 3. Must not block other tasks (training, extraction, etc.)
# 4. Process isolation: crashes don't affect main worker
#
# With solo pool, each worker runs one task at a time sequentially.
# This prevents GPU memory contention and allows SIGKILL to work properly.
#
start_steering_worker() {
    if is_running "$STEERING_PID_FILE"; then
        local pid=$(get_pid "$STEERING_PID_FILE")
        echo -e "${YELLOW}Steering worker already running (PID: $pid)${NC}"
        return 0
    fi

    ensure_venv

    # Clean up stale PID file
    rm -f "$STEERING_PID_FILE" 2>/dev/null

    echo -n "Starting Celery steering worker..."

    # Start steering worker with dedicated queue and AGGRESSIVE recycling
    # --max-tasks-per-child=1: Recycle worker after EVERY task to free GPU memory
    # This prevents orphaned GPU memory from accumulating across tasks
    nohup celery -A src.core.celery_app worker \
        -Q steering \
        -c 1 \
        --pool=solo \
        --loglevel=info \
        --hostname="steering@%h" \
        --max-tasks-per-child=1 \
        --pidfile="$STEERING_PID_FILE" \
        > "$STEERING_LOG" 2>&1 &

    # Wait for PID file to be created
    local count=0
    while [ ! -f "$STEERING_PID_FILE" ] && [ $count -lt 10 ]; do
        sleep 1
        count=$((count + 1))
    done

    if is_running "$STEERING_PID_FILE"; then
        local pid=$(get_pid "$STEERING_PID_FILE")
        echo -e " ${GREEN}started (PID: $pid)${NC}"
        return 0
    else
        echo -e " ${RED}failed${NC}"
        echo "Check logs: $STEERING_LOG"
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
    # NOTE: Steering worker is NOT auto-started
    # It starts on-demand when user clicks "Start Steering" in the UI
    # This prevents GPU memory being held when steering is not in use
    start_beat
    echo ""
    cmd_status
}

cmd_stop() {
    echo "Stopping Celery services..."
    # Increase timeout for steering worker to allow GPU cleanup
    stop_process "Beat" "$BEAT_PID_FILE" 5
    stop_process "Steering Worker" "$STEERING_PID_FILE" 20
    stop_process "Worker" "$WORKER_PID_FILE" 15

    # Clean up schedule file
    rm -f "$BEAT_SCHEDULE_FILE"* 2>/dev/null

    # Handle any remaining celery processes GRACEFULLY (no SIGKILL)
    # This prevents zombie creation by allowing processes to clean up
    local remaining=$(pgrep -f "celery.*worker" 2>/dev/null | wc -l)
    if [ "$remaining" -gt 0 ]; then
        echo -n "Sending SIGTERM to $remaining remaining celery processes..."
        pkill -TERM -f "celery.*worker" 2>/dev/null || true

        # Wait up to 10 seconds for graceful shutdown
        local count=0
        while [ $count -lt 10 ]; do
            remaining=$(pgrep -f "celery.*worker" 2>/dev/null | wc -l)
            if [ "$remaining" -eq 0 ]; then
                break
            fi
            echo -n "."
            sleep 1
            count=$((count + 1))
        done

        # Only SIGKILL if absolutely necessary after 10s wait
        remaining=$(pgrep -f "celery.*worker" 2>/dev/null | wc -l)
        if [ "$remaining" -gt 0 ]; then
            echo -n " (forcing $remaining)..."
            pkill -9 -f "celery.*worker" 2>/dev/null || true
            sleep 2
        fi
        echo -e " ${GREEN}done${NC}"
    fi

    # Check for zombies
    local zombie_count=$(ps aux | grep -E '\[celery\].*<defunct>' | grep -v grep | wc -l)
    if [ "$zombie_count" -gt 0 ]; then
        echo -e "${YELLOW}WARNING: $zombie_count zombie process(es) remain. Reboot may be required to free GPU memory.${NC}"
    fi
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
            echo -e "Worker:   ${GREEN}running${NC} (PID: $pid)"
        else
            echo -e "Worker:   ${YELLOW}running but unhealthy${NC} (PID: $pid)"
        fi
    else
        echo -e "Worker:   ${RED}stopped${NC}"
    fi

    if is_running "$STEERING_PID_FILE"; then
        local pid=$(get_pid "$STEERING_PID_FILE")
        echo -e "Steering: ${GREEN}running${NC} (PID: $pid)"
    else
        # Steering is on-demand, so not running is normal (yellow, not red)
        echo -e "Steering: ${YELLOW}not running${NC} (use './celery.sh start steering' to enable)"
    fi

    if is_running "$BEAT_PID_FILE"; then
        local pid=$(get_pid "$BEAT_PID_FILE")
        echo -e "Beat:     ${GREEN}running${NC} (PID: $pid)"
    else
        echo -e "Beat:     ${RED}stopped${NC}"
    fi

    if is_running "$WATCHDOG_PID_FILE"; then
        local pid=$(get_pid "$WATCHDOG_PID_FILE")
        echo -e "Watchdog: ${GREEN}running${NC} (PID: $pid)"
    else
        echo -e "Watchdog: ${YELLOW}not running${NC} (use './celery.sh watchdog' to enable)"
    fi

    echo ""
    echo "Memory Status:"
    echo "=============="
    local sys_ram=$(get_system_ram_usage)
    local worker_ram=$(get_worker_ram_mb "$WORKER_PID_FILE")
    local gpu_mem=$(get_gpu_memory_usage)

    # Color code based on thresholds
    if [ "$sys_ram" -ge "$SYSTEM_RAM_THRESHOLD" ]; then
        echo -e "System RAM:  ${RED}${sys_ram}%${NC} (threshold: ${SYSTEM_RAM_THRESHOLD}%)"
    elif [ "$sys_ram" -ge $((SYSTEM_RAM_THRESHOLD - 10)) ]; then
        echo -e "System RAM:  ${YELLOW}${sys_ram}%${NC} (threshold: ${SYSTEM_RAM_THRESHOLD}%)"
    else
        echo -e "System RAM:  ${GREEN}${sys_ram}%${NC} (threshold: ${SYSTEM_RAM_THRESHOLD}%)"
    fi

    if [ "$worker_ram" -ge "$WORKER_RAM_THRESHOLD_MB" ]; then
        echo -e "Worker RAM:  ${RED}${worker_ram}MB${NC} (threshold: ${WORKER_RAM_THRESHOLD_MB}MB)"
    elif [ "$worker_ram" -ge $((WORKER_RAM_THRESHOLD_MB - 2000)) ]; then
        echo -e "Worker RAM:  ${YELLOW}${worker_ram}MB${NC} (threshold: ${WORKER_RAM_THRESHOLD_MB}MB)"
    else
        echo -e "Worker RAM:  ${GREEN}${worker_ram}MB${NC} (threshold: ${WORKER_RAM_THRESHOLD_MB}MB)"
    fi

    if [ "$gpu_mem" -ge "$GPU_MEMORY_THRESHOLD" ]; then
        echo -e "GPU Memory:  ${RED}${gpu_mem}%${NC} (threshold: ${GPU_MEMORY_THRESHOLD}%)"
    elif [ "$gpu_mem" -ge $((GPU_MEMORY_THRESHOLD - 10)) ]; then
        echo -e "GPU Memory:  ${YELLOW}${gpu_mem}%${NC} (threshold: ${GPU_MEMORY_THRESHOLD}%)"
    else
        echo -e "GPU Memory:  ${GREEN}${gpu_mem}%${NC} (threshold: ${GPU_MEMORY_THRESHOLD}%)"
    fi

    echo ""
    echo "Logs:"
    echo "  Worker:   $WORKER_LOG"
    echo "  Steering: $STEERING_LOG"
    echo "  Beat:     $BEAT_LOG"
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
        # Disable exit-on-error in watchdog - we handle errors ourselves
        set +e

        # Use BASHPID instead of $$ to get the actual subshell PID
        echo $BASHPID > "$WATCHDOG_PID_FILE"

        log_msg() {
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" >> "$WATCHDOG_LOG"
        }

        log_msg "Watchdog started (PID: $BASHPID)"

        while true; do
            # Check if Redis is available
            if ! docker exec mistudio-redis redis-cli ping 2>/dev/null | grep -q "PONG"; then
                log_msg "WARNING: Redis not responding, waiting..."
                sleep "$WATCHDOG_INTERVAL"
                continue
            fi

            # Check and restart worker if needed
            worker_running=false
            worker_healthy=false
            is_running "$WORKER_PID_FILE" && worker_running=true
            is_worker_healthy && worker_healthy=true
            if [ "$worker_running" = "false" ] || [ "$worker_healthy" = "false" ]; then
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

            # Check memory pressure and proactively restart if needed
            if [ "$worker_running" = "true" ]; then
                memory_reason=$(check_memory_pressure "$WORKER_PID_FILE" 2>/dev/null) || true
                if [ -n "$memory_reason" ]; then
                    log_msg "MEMORY PRESSURE: $memory_reason - initiating proactive restart..."

                    # Log current memory stats before restart
                    local sys_ram=$(get_system_ram_usage)
                    local worker_ram=$(get_worker_ram_mb "$WORKER_PID_FILE")
                    local gpu_mem=$(get_gpu_memory_usage)
                    log_msg "Memory stats before restart: System RAM=${sys_ram}%, Worker=${worker_ram}MB, GPU=${gpu_mem}%"

                    # Graceful restart
                    stop_process "Worker" "$WORKER_PID_FILE" 15 >> "$WATCHDOG_LOG" 2>&1
                    sleep 3
                    start_worker >> "$WATCHDOG_LOG" 2>&1

                    if is_running "$WORKER_PID_FILE"; then
                        log_msg "Worker restarted due to memory pressure"
                        # Log memory after restart
                        sleep 5
                        local new_sys_ram=$(get_system_ram_usage)
                        local new_worker_ram=$(get_worker_ram_mb "$WORKER_PID_FILE")
                        local new_gpu_mem=$(get_gpu_memory_usage)
                        log_msg "Memory stats after restart: System RAM=${new_sys_ram}%, Worker=${new_worker_ram}MB, GPU=${new_gpu_mem}%"
                    else
                        log_msg "ERROR: Failed to restart worker after memory pressure"
                    fi
                fi
            fi

            # NOTE: Steering worker is NOT auto-restarted by watchdog
            # It is started on-demand via the UI "Start Steering" button
            # This prevents GPU memory being held when steering is not in use

            # Check for zombie processes (defunct) related to celery
            zombie_count=$(ps aux | grep -E '\[celery\].*<defunct>' | grep -v grep | wc -l)
            if [ "$zombie_count" -gt 0 ]; then
                log_msg "WARNING: Found $zombie_count zombie celery process(es) - orphaned GPU memory likely"
                # Check GPU memory for orphaned allocations
                local gpu_used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
                local gpu_procs=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -v "No running" | wc -l)
                if [ "$gpu_used" -gt 1000 ] && [ "$gpu_procs" -eq 0 ]; then
                    log_msg "CRITICAL: ${gpu_used}MiB GPU memory orphaned (no active processes) - reboot may be required"
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
