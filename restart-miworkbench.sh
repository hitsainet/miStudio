#!/bin/bash

# miWorkbench - Restart All Services Script

# The checkout this script lives in — never a hardcoded home-directory path.
PROJECT_ROOT="$(cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")" && pwd)"

echo "=================================="
echo "Restarting miWorkbench"
echo "=================================="
echo ""

# Stop all services
"$PROJECT_ROOT/stop-miworkbench.sh"

# Brief pause to ensure clean shutdown
sleep 2

# Start all services
"$PROJECT_ROOT/start-miworkbench.sh"
