#!/bin/bash

# MechInterp Studio - Complete Application Startup Script
# This script starts all required services for the application

set -e

PROJECT_ROOT="/home/x-sean/app/miStudio"
DOMAIN="mistudio.mcslab.io"

echo "=================================="
echo "Starting MechInterp Studio"
echo "=================================="

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Function to check if a service is running
check_service() {
    local service_name=$1
    local check_command=$2

    if eval "$check_command" > /dev/null 2>&1; then
        echo -e "${GREEN}✓${NC} $service_name is running"
        return 0
    else
        echo -e "${RED}✗${NC} $service_name is NOT running"
        return 1
    fi
}

# Function to wait for service
wait_for_service() {
    local service_name=$1
    local check_command=$2
    local max_attempts=30
    local attempt=1

    echo -n "Waiting for $service_name..."
    while [ $attempt -le $max_attempts ]; do
        if eval "$check_command" > /dev/null 2>&1; then
            echo -e " ${GREEN}✓${NC}"
            return 0
        fi
        echo -n "."
        sleep 1
        attempt=$((attempt + 1))
    done
    echo -e " ${RED}✗ Timeout${NC}"
    return 1
}

# Step 0: Check CUDA health before starting anything
echo ""
echo "Step 0: Checking CUDA/GPU health..."
if command -v nvidia-smi &> /dev/null; then
    # Check for zombie processes holding GPU memory
    gpu_mem_used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
    gpu_procs=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -v "No running" | wc -l)
    # Count actual zombie processes by checking process state (Z = zombie)
    zombie_count=$(ps aux 2>/dev/null | awk '$8 ~ /^Z/' | wc -l)

    # Check for orphaned GPU memory (memory used but no processes)
    if [ "$gpu_mem_used" -gt 500 ] && [ "$gpu_procs" -eq 0 ]; then
        echo -e "${RED}✗ CRITICAL: ${gpu_mem_used}MiB GPU memory orphaned with no active processes${NC}"
        echo -e "${RED}  CUDA driver may be corrupted. Reboot required.${NC}"
        echo ""
        read -p "Continue anyway? (y/N) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo "Exiting. Please reboot to clear GPU state."
            exit 1
        fi
    elif [ "$zombie_count" -gt 0 ]; then
        echo -e "${YELLOW}⚠ WARNING: Found $zombie_count zombie process(es)${NC}"
        echo -e "${YELLOW}  GPU may have orphaned memory. Consider rebooting.${NC}"
    else
        echo -e "${GREEN}✓${NC} GPU health OK (${gpu_mem_used}MiB used, $gpu_procs active processes)"
        # Show what's using GPU memory if anything
        if [ "$gpu_procs" -gt 0 ]; then
            echo "  GPU processes:"
            # Get unique PIDs with total memory across all GPUs
            nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null | \
            awk -F',' '{
                pid=$1; gsub(/^ +| +$/, "", pid);
                mem=$2; gsub(/^ +| +$|MiB/, "", mem);
                total[pid] += mem
            } END {
                for (pid in total) print pid, total[pid]"MiB"
            }' | while read pid mem; do
                # Get process description from cmdline
                if [ -f "/proc/$pid/cmdline" ]; then
                    cmdline=$(tr '\0' ' ' < /proc/$pid/cmdline 2>/dev/null)
                    # Identify known miStudio processes
                    if echo "$cmdline" | grep -q "steering"; then
                        desc="Celery Steering Worker"
                    elif echo "$cmdline" | grep -q "celery"; then
                        desc="Celery Worker"
                    elif echo "$cmdline" | grep -q "uvicorn"; then
                        desc="FastAPI Backend"
                    elif echo "$cmdline" | grep -q "ollama"; then
                        desc="Ollama LLM Server"
                    else
                        desc=$(echo "$cmdline" | cut -c1-50)
                    fi
                else
                    desc="(process ended)"
                fi
                echo -e "    PID $pid: ${mem} - ${desc}"
            done
        fi
    fi

    # Quick CUDA sanity check using Python
    cd "$PROJECT_ROOT/backend"
    if [ -f "venv/bin/python" ]; then
        cuda_check=$(venv/bin/python -c "
import torch
try:
    if torch.cuda.is_available():
        count = torch.cuda.device_count()
        print(f'OK:{count}')
    else:
        print('NO_CUDA')
except Exception as e:
    print(f'ERROR:{e}')
" 2>&1)

        if [[ "$cuda_check" == OK:* ]]; then
            device_count="${cuda_check#OK:}"
            echo -e "${GREEN}✓${NC} CUDA initialized successfully ($device_count GPU(s) available)"
        elif [[ "$cuda_check" == "NO_CUDA" ]]; then
            echo -e "${YELLOW}⚠${NC} CUDA not available - will use CPU"
        elif [[ "$cuda_check" == ERROR:* ]]; then
            echo -e "${RED}✗ CUDA initialization failed: ${cuda_check#ERROR:}${NC}"
            echo -e "${RED}  CUDA driver is corrupted. Reboot required.${NC}"
            echo ""
            read -p "Continue anyway (will use CPU)? (y/N) " -n 1 -r
            echo
            if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                echo "Exiting. Please reboot to clear GPU state."
                exit 1
            fi
        fi
    fi
    cd "$PROJECT_ROOT"
else
    echo -e "${YELLOW}⚠${NC} nvidia-smi not found - GPU support unavailable"
fi

echo ""
echo "Step 1: Checking /etc/hosts for domain..."
if ! grep -q "$DOMAIN" /etc/hosts; then
    echo -e "${YELLOW}⚠${NC}  Domain not found in /etc/hosts"
    echo "  You need to add this line to /etc/hosts (requires sudo):"
    echo "  127.0.0.1  $DOMAIN"
    echo ""
    echo "  Run: sudo bash -c 'echo \"127.0.0.1  $DOMAIN\" >> /etc/hosts'"
    echo ""
else
    echo -e "${GREEN}✓${NC} Domain configured in /etc/hosts"
fi

echo ""
echo "Step 2: Starting Docker services (PostgreSQL, Redis, Nginx)..."
cd "$PROJECT_ROOT"
docker-compose -f docker-compose.dev.yml up -d postgres redis nginx

# Wait for services to be healthy
wait_for_service "PostgreSQL" "docker exec mistudio-postgres pg_isready -U postgres"
wait_for_service "Redis" "docker exec mistudio-redis redis-cli ping"
echo -e "${GREEN}✓${NC} PostgreSQL and Redis are healthy"

# Start Ollama with GPU support
echo ""
echo "Starting Ollama with GPU support..."
if docker ps -a --format '{{.Names}}' | grep -q "^mistudio-ollama$"; then
    if docker ps --format '{{.Names}}' | grep -q "^mistudio-ollama$"; then
        echo -e "${GREEN}✓${NC} Ollama already running"
    else
        docker start mistudio-ollama > /dev/null
        echo -e "${GREEN}✓${NC} Ollama restarted"
    fi
else
    docker run -d --name mistudio-ollama --gpus all \
        -p 11434:11434 \
        -v ollama_data:/root/.ollama \
        -e OLLAMA_ORIGINS="http://mistudio.mcslab.io,http://localhost:3000,http://localhost" \
        --network mistudio_default \
        --restart unless-stopped \
        ollama/ollama:latest > /dev/null
    echo -e "${GREEN}✓${NC} Ollama started with GPU support"
fi

# Ensure Ollama is on the dev networks for nginx to reach it
docker network connect mistudio_default mistudio-ollama 2>/dev/null || true
docker network connect mistudio_mistudio-dev mistudio-ollama 2>/dev/null || true

wait_for_service "Ollama" "curl -s http://localhost:11434/api/tags"
echo -e "${GREEN}✓${NC} All Docker services are healthy"

echo ""
echo "Step 3: Starting Celery services (worker + beat)..."
cd "$PROJECT_ROOT/backend"
./celery.sh start

echo ""
echo "Step 4: Starting Backend (FastAPI)..."
cd "$PROJECT_ROOT/backend"

# Clean up any existing backend process using PID file
BACKEND_PID_FILE="/tmp/mistudio-backend.pid"
if [ -f "$BACKEND_PID_FILE" ]; then
    OLD_PID=$(cat "$BACKEND_PID_FILE" 2>/dev/null)
    if [ -n "$OLD_PID" ] && kill -0 "$OLD_PID" 2>/dev/null; then
        echo -e "${YELLOW}⚠${NC}  Stopping existing backend (PID: $OLD_PID)..."
        # Kill the entire process group
        kill -TERM -"$OLD_PID" 2>/dev/null || true
        sleep 2
        # Force kill if still running
        kill -KILL -"$OLD_PID" 2>/dev/null || true
        sleep 1
    fi
    rm -f "$BACKEND_PID_FILE"
fi

# Also check if port is in use by orphaned process
if lsof -i :8000 > /dev/null 2>&1; then
    echo -e "${YELLOW}⚠${NC}  Port 8000 still in use, attempting cleanup..."
    fuser -k 8000/tcp 2>/dev/null || true
    sleep 2
fi

source venv/bin/activate

# Start uvicorn in its own process group using setsid
# This allows us to kill the entire process tree later
setsid uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload > /tmp/backend.log 2>&1 &
BACKEND_PID=$!
echo "$BACKEND_PID" > "$BACKEND_PID_FILE"

wait_for_service "Backend" "curl -s http://localhost:8000/api/v1/datasets"
echo "  Backend logs: /tmp/backend.log"
echo "  Backend PID: $BACKEND_PID (saved to $BACKEND_PID_FILE)"

echo ""
echo "Step 5: Starting Frontend (Vite)..."
cd "$PROJECT_ROOT/frontend"

# Clean up any existing frontend process using PID file
FRONTEND_PID_FILE="/tmp/mistudio-frontend.pid"
if [ -f "$FRONTEND_PID_FILE" ]; then
    OLD_PID=$(cat "$FRONTEND_PID_FILE" 2>/dev/null)
    if [ -n "$OLD_PID" ] && kill -0 "$OLD_PID" 2>/dev/null; then
        echo -e "${YELLOW}⚠${NC}  Stopping existing frontend (PID: $OLD_PID)..."
        kill -TERM -"$OLD_PID" 2>/dev/null || true
        sleep 2
        kill -KILL -"$OLD_PID" 2>/dev/null || true
        sleep 1
    fi
    rm -f "$FRONTEND_PID_FILE"
fi

if lsof -i :3000 > /dev/null 2>&1; then
    echo -e "${YELLOW}⚠${NC}  Port 3000 still in use, attempting cleanup..."
    fuser -k 3000/tcp 2>/dev/null || true
    sleep 2
fi

# Start frontend in its own process group
setsid npm run dev > /tmp/frontend.log 2>&1 &
FRONTEND_PID=$!
echo "$FRONTEND_PID" > "$FRONTEND_PID_FILE"

wait_for_service "Frontend" "curl -s http://localhost:3000"
echo "  Frontend logs: /tmp/frontend.log"
echo "  Frontend PID: $FRONTEND_PID (saved to $FRONTEND_PID_FILE)"

echo ""
echo "=================================="
echo "All Services Started!"
echo "=================================="
echo ""
check_service "PostgreSQL (Docker)" "docker exec mistudio-postgres pg_isready -U postgres"
check_service "Redis (Docker)" "docker exec mistudio-redis redis-cli ping"
check_service "Nginx (Docker)" "docker exec mistudio-nginx nginx -t"
check_service "Ollama (Docker)" "curl -s http://localhost:11434/api/tags"
check_service "Celery Worker" "test -f /tmp/mistudio-celery-worker.pid && kill -0 \$(cat /tmp/mistudio-celery-worker.pid 2>/dev/null) 2>/dev/null"
# Steering is on-demand - show status but don't treat as error if not running
if test -f /tmp/mistudio-celery-steering.pid && kill -0 $(cat /tmp/mistudio-celery-steering.pid 2>/dev/null) 2>/dev/null; then
    echo -e "${GREEN}✓${NC} Celery Steering is running"
else
    echo -e "${YELLOW}○${NC} Celery Steering is off (starts on-demand via UI)"
fi
check_service "Celery Beat" "test -f /tmp/mistudio-celery-beat.pid && kill -0 \$(cat /tmp/mistudio-celery-beat.pid 2>/dev/null) 2>/dev/null"
check_service "Backend (FastAPI)" "test -f /tmp/mistudio-backend.pid && kill -0 \$(cat /tmp/mistudio-backend.pid 2>/dev/null) 2>/dev/null && curl -s http://localhost:8000/api/v1/datasets"
check_service "Frontend (Vite)" "test -f /tmp/mistudio-frontend.pid && kill -0 \$(cat /tmp/mistudio-frontend.pid 2>/dev/null) 2>/dev/null && curl -s http://localhost:3000"

echo ""
echo "=================================="
echo "Access URLs:"
echo "=================================="
echo "  Primary: http://$DOMAIN"
echo "  Frontend: http://localhost:3000"
echo "  Backend: http://localhost:8000"
echo "  API Docs: http://localhost:8000/docs"
echo "  Ollama API: http://localhost:11434"
echo ""
echo "Logs:"
echo "  Backend: /tmp/backend.log"
echo "  Frontend: /tmp/frontend.log"
echo "  Celery Worker: /tmp/celery-worker.log"
echo "  Celery Steering: /tmp/celery-steering.log"
echo "  Celery Beat: /tmp/celery-beat.log"
echo "  Ollama: docker logs mistudio-ollama"
echo ""
