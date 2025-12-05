#!/bin/bash

# ==============================================================================
# miStudio Docker Stop Script
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

GREEN='\033[0;32m'
NC='\033[0m'

echo "=================================="
echo "Stopping MechInterp Studio (Docker)"
echo "=================================="
echo ""

# Stop all services
docker-compose down

echo ""
echo -e "${GREEN}✓${NC} All services stopped"
echo ""
echo "To remove volumes (DELETE ALL DATA):"
echo "  docker-compose down -v"
echo ""
