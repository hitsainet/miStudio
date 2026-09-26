#!/bin/bash

# MechInterp Studio - Restart All Services Script

PROJECT_ROOT="/home/x-sean/app/miStudio"

echo "=================================="
echo "Restarting MechInterp Studio"
echo "=================================="
echo ""

# Stop all services
"$PROJECT_ROOT/stop-mistudio.sh"

# Brief pause to ensure clean shutdown
sleep 2

# Start all services
"$PROJECT_ROOT/start-mistudio.sh"
