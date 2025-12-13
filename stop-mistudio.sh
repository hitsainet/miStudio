#!/bin/bash

# MechInterp Studio - Stop All Services Script

set -e

PROJECT_ROOT="/home/x-sean/app/miStudio"

echo "=================================="
echo "Stopping MechInterp Studio"
echo "=================================="

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo ""
echo "Stopping Frontend (Vite)..."
pkill -f "vite" || echo -e "${YELLOW}⚠${NC}  No Vite process found"

echo ""
echo "Stopping Backend (FastAPI)..."
pkill -f "uvicorn src.main:app" || echo -e "${YELLOW}⚠${NC}  No uvicorn process found"

echo ""
echo "Stopping Celery services (worker + beat)..."
cd "$PROJECT_ROOT/backend"
./celery.sh stop

echo ""
echo "Stopping Docker services (keeping data)..."
cd "$PROJECT_ROOT"

# Stop Ollama separately (started with docker run)
if docker ps --format '{{.Names}}' | grep -q "^mistudio-ollama$"; then
    docker stop mistudio-ollama > /dev/null
fi

# Stop docker-compose services
docker-compose -f docker-compose.dev.yml stop

echo ""
echo -e "${GREEN}✓${NC} All services stopped"
echo ""
echo "To completely remove containers and volumes:"
echo "  cd $PROJECT_ROOT && docker-compose -f docker-compose.dev.yml down -v"
echo "  docker rm mistudio-ollama  # Remove Ollama container"
echo ""
