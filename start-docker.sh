#!/bin/bash

# ==============================================================================
# miStudio Docker Startup Script
# ==============================================================================
# Starts all services using Docker Compose
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "=================================="
echo "Starting MechInterp Studio (Docker)"
echo "=================================="
echo ""

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo -e "${YELLOW}Warning: .env file not found. Using default values.${NC}"
    echo "Copy .env.example to .env and customize for your environment."
    echo ""
fi

# Check for /etc/hosts entry
echo "Step 1: Checking /etc/hosts for domain..."
if grep -q "mistudio.mcslab.io" /etc/hosts; then
    echo -e "${GREEN}✓${NC} Domain configured in /etc/hosts"
else
    echo -e "${YELLOW}!${NC} Domain not in /etc/hosts"
    echo "Add the following line to /etc/hosts:"
    echo "  127.0.0.1  mistudio.mcslab.io"
    echo ""
fi

# Determine if production mode
COMPOSE_FILES="-f docker-compose.yml"
if [ "$1" = "prod" ] || [ "$1" = "production" ]; then
    echo "Mode: PRODUCTION"
    COMPOSE_FILES="-f docker-compose.yml -f docker-compose.prod.yml"
else
    echo "Mode: DEVELOPMENT"
fi
echo ""

# Build and start services
echo "Step 2: Building Docker images..."
docker-compose $COMPOSE_FILES build --parallel
echo -e "${GREEN}✓${NC} Images built"
echo ""

echo "Step 3: Starting services..."
docker-compose $COMPOSE_FILES up -d
echo ""

# Wait for services to be healthy
echo "Step 4: Waiting for services to be healthy..."

# Function to wait for a container to be healthy
wait_for_healthy() {
    local container=$1
    local max_attempts=${2:-30}
    local attempt=1

    while [ $attempt -le $max_attempts ]; do
        status=$(docker inspect --format='{{.State.Health.Status}}' $container 2>/dev/null || echo "starting")
        if [ "$status" = "healthy" ]; then
            return 0
        fi
        echo -n "."
        sleep 2
        attempt=$((attempt + 1))
    done
    return 1
}

# Wait for PostgreSQL
echo -n "  PostgreSQL"
if wait_for_healthy mistudio-postgres; then
    echo -e " ${GREEN}✓${NC}"
else
    echo -e " ${RED}✗${NC} (timeout)"
fi

# Wait for Redis
echo -n "  Redis"
if wait_for_healthy mistudio-redis; then
    echo -e " ${GREEN}✓${NC}"
else
    echo -e " ${RED}✗${NC} (timeout)"
fi

# Wait for Backend (may take longer due to migrations)
echo -n "  Backend"
sleep 5  # Give it a head start
if wait_for_healthy mistudio-backend 60; then
    echo -e " ${GREEN}✓${NC}"
else
    echo -e " ${YELLOW}!${NC} (still starting, check logs)"
fi

# Wait for Frontend
echo -n "  Frontend"
for i in {1..20}; do
    if curl -s -o /dev/null -w "%{http_code}" http://localhost:3000 | grep -q "200\|304"; then
        echo -e " ${GREEN}✓${NC}"
        break
    fi
    echo -n "."
    sleep 2
done

echo ""
echo "=================================="
echo "Service Status"
echo "=================================="
docker-compose $COMPOSE_FILES ps
echo ""

echo "=================================="
echo "Access URLs"
echo "=================================="
echo "  Primary:    http://mistudio.mcslab.io"
echo "  Frontend:   http://localhost:3000"
echo "  Backend:    http://localhost:8000"
echo "  API Docs:   http://localhost:8000/docs"
echo "  Ollama:     http://localhost:11434"
echo ""

echo "=================================="
echo "Logs"
echo "=================================="
echo "  All:        docker-compose logs -f"
echo "  Backend:    docker-compose logs -f backend"
echo "  Frontend:   docker-compose logs -f frontend"
echo "  Celery:     docker-compose logs -f celery-worker"
echo ""
