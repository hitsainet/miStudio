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
check_service "Celery Steering" "test -f /tmp/mistudio-celery-steering.pid && kill -0 \$(cat /tmp/mistudio-celery-steering.pid 2>/dev/null) 2>/dev/null"
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
