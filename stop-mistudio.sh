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
RED='\033[0;31m'
NC='\033[0m' # No Color

# Function to gracefully stop a process group
stop_process_group() {
    local name=$1
    local pid_file=$2
    local fallback_pattern=$3

    echo ""
    echo "Stopping $name..."

    if [ -f "$pid_file" ]; then
        PID=$(cat "$pid_file" 2>/dev/null)
        if [ -n "$PID" ]; then
            if kill -0 "$PID" 2>/dev/null; then
                echo "  Sending SIGTERM to process group (PID: $PID)..."
                # Kill entire process group with SIGTERM
                kill -TERM -"$PID" 2>/dev/null || true

                # Wait up to 5 seconds for graceful shutdown
                for i in {1..10}; do
                    if ! kill -0 "$PID" 2>/dev/null; then
                        echo -e "  ${GREEN}✓${NC} $name stopped gracefully"
                        rm -f "$pid_file"
                        return 0
                    fi
                    sleep 0.5
                done

                # Force kill if still running
                echo "  Process still running, sending SIGKILL..."
                kill -KILL -"$PID" 2>/dev/null || true
                sleep 1

                if ! kill -0 "$PID" 2>/dev/null; then
                    echo -e "  ${GREEN}✓${NC} $name force killed"
                else
                    echo -e "  ${RED}✗${NC} Failed to stop $name"
                fi
            else
                echo -e "  ${YELLOW}⚠${NC}  PID $PID not running"
            fi
            rm -f "$pid_file"
        fi
    else
        # Fallback: use pkill if no PID file
        if pgrep -f "$fallback_pattern" > /dev/null 2>&1; then
            echo "  No PID file, using pkill fallback..."
            pkill -TERM -f "$fallback_pattern" 2>/dev/null || true
            sleep 2
            pkill -KILL -f "$fallback_pattern" 2>/dev/null || true
            echo -e "  ${GREEN}✓${NC} $name stopped via pkill"
        else
            echo -e "  ${YELLOW}⚠${NC}  No $name process found"
        fi
    fi
}

stop_process_group "Frontend (Vite)" "/tmp/mistudio-frontend.pid" "vite"
stop_process_group "Backend (FastAPI)" "/tmp/mistudio-backend.pid" "uvicorn src.main:app"

echo ""
echo "Stopping Celery services (worker + beat)..."
cd "$PROJECT_ROOT/backend"
./celery.sh stop

echo ""
echo "Stopping Docker services (keeping data)..."
cd "$PROJECT_ROOT"

# Stop oLLM (started with docker run)
if docker ps --format '{{.Names}}' | grep -q "^mistudio-ollm$"; then
    docker stop mistudio-ollm > /dev/null
    echo -e "${GREEN}✓${NC} oLLM stopped"
fi

# Stop legacy Ollama if running (started with docker run)
if docker ps --format '{{.Names}}' | grep -q "^mistudio-ollama$"; then
    docker stop mistudio-ollama > /dev/null
    echo -e "${GREEN}✓${NC} Ollama stopped"
fi

# Stop docker-compose services
docker-compose -f docker-compose.dev.yml stop

echo ""
echo -e "${GREEN}✓${NC} All services stopped"

# Verify no orphaned sockets
echo ""
echo "Verifying clean shutdown..."
ORPHANED_SOCKETS=0

if ss -tlnp | grep -q ":8000 "; then
    echo -e "${RED}⚠${NC}  WARNING: Port 8000 still has a socket (may be orphaned)"
    ORPHANED_SOCKETS=1
fi

if ss -tlnp | grep -q ":3000 "; then
    echo -e "${RED}⚠${NC}  WARNING: Port 3000 still has a socket (may be orphaned)"
    ORPHANED_SOCKETS=1
fi

# Check for zombie processes
ZOMBIES=$(ps aux | grep -E "\[.*\] <defunct>" | grep -v grep | wc -l)
if [ "$ZOMBIES" -gt 0 ]; then
    echo -e "${RED}⚠${NC}  WARNING: Found $ZOMBIES zombie process(es). A reboot may be required."
    ORPHANED_SOCKETS=1
fi

if [ "$ORPHANED_SOCKETS" -eq 0 ]; then
    echo -e "${GREEN}✓${NC} All ports and processes cleanly released"
fi

echo ""
echo "To completely remove containers and volumes:"
echo "  cd $PROJECT_ROOT && docker-compose -f docker-compose.dev.yml down -v"
echo "  docker rm mistudio-ollm   # Remove oLLM container"
echo "  docker rm mistudio-ollama  # Remove Ollama container (if exists)"
echo ""
