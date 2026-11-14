#!/bin/bash
#
# miWorkbench Systemd Services Installation Script
#
# This script installs and configures systemd services for miWorkbench.
# It provides automatic startup and restart capabilities for:
#   - Celery Worker (background task processing)
#   - Celery Beat (scheduled tasks)
#   - FastAPI Backend (REST API server)
#
# Prerequisites:
#   - Docker must be installed and running (for Redis, PostgreSQL, Nginx)
#   - Python virtual environment must exist at backend/venv/
#   - User must have sudo access
#
# Usage:
#   ./install-services.sh [install|uninstall|status|logs]
#
# Examples:
#   ./install-services.sh install     # Install and enable all services
#   ./install-services.sh uninstall   # Stop and remove all services
#   ./install-services.sh status      # Show status of all services
#   ./install-services.sh logs        # Follow logs for all services

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BACKEND_DIR="$(dirname "$SCRIPT_DIR")"

# Service files
SERVICES=(
    "miworkbench-celery-worker.service"
    "miworkbench-celery-beat.service"
    "miworkbench-backend.service"
    "miworkbench.target"
)

echo_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

echo_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

echo_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_prerequisites() {
    echo_info "Checking prerequisites..."

    # Check if running as root or with sudo
    if [ "$EUID" -ne 0 ]; then
        echo_error "This script must be run with sudo"
        exit 1
    fi

    # Check if Docker is running
    if ! docker info &>/dev/null; then
        echo_warn "Docker is not running. Make sure to start Docker services first."
    fi

    # Check if virtual environment exists
    if [ ! -d "$BACKEND_DIR/venv" ]; then
        echo_error "Virtual environment not found at $BACKEND_DIR/venv"
        echo_info "Please create it first: cd $BACKEND_DIR && python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt"
        exit 1
    fi

    echo_info "Prerequisites check passed"
}

install_services() {
    echo_info "Installing miWorkbench systemd services..."

    check_prerequisites

    # Copy service files to systemd directory
    for service in "${SERVICES[@]}"; do
        echo_info "Installing $service..."
        cp "$SCRIPT_DIR/$service" /etc/systemd/system/
    done

    # Reload systemd daemon
    echo_info "Reloading systemd daemon..."
    systemctl daemon-reload

    # Enable services (but don't start them yet)
    echo_info "Enabling services..."
    for service in "${SERVICES[@]}"; do
        systemctl enable "$service" || true
    done

    echo ""
    echo_info "Installation complete!"
    echo ""
    echo "To start all miWorkbench services:"
    echo "  sudo systemctl start miworkbench.target"
    echo ""
    echo "Or start individual services:"
    echo "  sudo systemctl start miworkbench-celery-worker"
    echo "  sudo systemctl start miworkbench-celery-beat"
    echo "  sudo systemctl start miworkbench-backend"
    echo ""
    echo "To view logs:"
    echo "  journalctl -u miworkbench-celery-worker -f"
    echo "  journalctl -u miworkbench-celery-beat -f"
    echo "  journalctl -u miworkbench-backend -f"
    echo ""
    echo -e "${YELLOW}IMPORTANT:${NC} Make sure Docker services (Redis, PostgreSQL) are running first!"
    echo "  cd $BACKEND_DIR && docker compose up -d"
}

uninstall_services() {
    echo_info "Uninstalling miWorkbench systemd services..."

    # Stop services
    echo_info "Stopping services..."
    for service in "${SERVICES[@]}"; do
        systemctl stop "$service" 2>/dev/null || true
    done

    # Disable services
    echo_info "Disabling services..."
    for service in "${SERVICES[@]}"; do
        systemctl disable "$service" 2>/dev/null || true
    done

    # Remove service files
    echo_info "Removing service files..."
    for service in "${SERVICES[@]}"; do
        rm -f "/etc/systemd/system/$service"
    done

    # Reload systemd daemon
    systemctl daemon-reload

    echo_info "Uninstallation complete!"
}

show_status() {
    echo "miWorkbench Services Status"
    echo "========================"
    echo ""

    for service in "${SERVICES[@]}"; do
        if [ -f "/etc/systemd/system/$service" ]; then
            echo -e "${GREEN}[$service]${NC}"
            systemctl status "$service" --no-pager -l 2>/dev/null | head -10 || echo "  Not running"
            echo ""
        else
            echo -e "${YELLOW}[$service]${NC} - Not installed"
            echo ""
        fi
    done
}

show_logs() {
    echo "Following miWorkbench service logs (Ctrl+C to stop)..."
    echo ""
    journalctl -u miworkbench-celery-worker -u miworkbench-celery-beat -u miworkbench-backend -f
}

# Main
case "${1:-}" in
    install)
        install_services
        ;;
    uninstall)
        uninstall_services
        ;;
    status)
        show_status
        ;;
    logs)
        show_logs
        ;;
    *)
        echo "miWorkbench Systemd Services Installer"
        echo ""
        echo "Usage: $0 {install|uninstall|status|logs}"
        echo ""
        echo "Commands:"
        echo "  install   - Install and enable all systemd services"
        echo "  uninstall - Stop, disable, and remove all services"
        echo "  status    - Show status of all services"
        echo "  logs      - Follow logs for all services"
        exit 1
        ;;
esac
