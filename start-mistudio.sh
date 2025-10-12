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
docker-compose -f docker-compose.dev.yml up -d

# Wait for services to be healthy
wait_for_service "PostgreSQL" "docker exec mistudio-postgres pg_isready -U postgres"
wait_for_service "Redis" "docker exec mistudio-redis redis-cli ping"
echo -e "${GREEN}✓${NC} Docker services are healthy"

echo ""
echo "Step 3: Starting Celery worker..."
if pgrep -f "celery.*src.core.celery_app" > /dev/null; then
    echo -e "${GREEN}✓${NC} Celery worker already running"
else
    cd "$PROJECT_ROOT/backend"
    ./start-celery-worker.sh > /tmp/celery-worker.log 2>&1 &
    sleep 3
    if pgrep -f "celery.*src.core.celery_app" > /dev/null; then
        echo -e "${GREEN}✓${NC} Celery worker started"
    else
        echo -e "${RED}✗${NC} Failed to start Celery worker (check /tmp/celery-worker.log)"
    fi
fi

echo ""
echo "Step 4: Starting Backend (FastAPI)..."
cd "$PROJECT_ROOT/backend"
if lsof -i :8000 > /dev/null 2>&1; then
    echo -e "${YELLOW}⚠${NC}  Port 8000 already in use, stopping existing process..."
    pkill -f "uvicorn src.main:app" || true
    sleep 2
fi

source venv/bin/activate
nohup uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload > /tmp/backend.log 2>&1 &
wait_for_service "Backend" "curl -s http://localhost:8000/api/v1/datasets"
echo "  Backend logs: /tmp/backend.log"

echo ""
echo "Step 5: Starting Frontend (Vite)..."
cd "$PROJECT_ROOT/frontend"
if lsof -i :3000 > /dev/null 2>&1; then
    echo -e "${YELLOW}⚠${NC}  Port 3000 already in use, stopping existing process..."
    pkill -f "vite" || true
    sleep 2
fi

nohup npm run dev > /tmp/frontend.log 2>&1 &
wait_for_service "Frontend" "curl -s http://localhost:3000"
echo "  Frontend logs: /tmp/frontend.log"

echo ""
echo "=================================="
echo "All Services Started!"
echo "=================================="
echo ""
check_service "PostgreSQL (Docker)" "docker exec mistudio-postgres pg_isready -U postgres"
check_service "Redis (Docker)" "docker exec mistudio-redis redis-cli ping"
check_service "Nginx (Docker)" "docker exec mistudio-nginx nginx -t"
check_service "Celery Worker" "pgrep -f 'celery.*src.core.celery_app'"
check_service "Backend (FastAPI)" "curl -s http://localhost:8000/api/v1/datasets"
check_service "Frontend (Vite)" "curl -s http://localhost:3000"

echo ""
echo "=================================="
echo "Access URLs:"
echo "=================================="
echo "  Primary: http://$DOMAIN"
echo "  Frontend: http://localhost:3000"
echo "  Backend: http://localhost:8000"
echo "  API Docs: http://localhost:8000/docs"
echo ""
echo "Logs:"
echo "  Backend: /tmp/backend.log"
echo "  Frontend: /tmp/frontend.log"
echo "  Celery: /tmp/celery-worker.log"
echo ""
