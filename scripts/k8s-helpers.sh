#!/bin/bash
# K8s Helper Functions for miStudio UAT Deployment
# Source this file: source scripts/k8s-helpers.sh

K8S_HOST="192.168.244.61"
K8S_USER="sean"
K8S_PASS="pass"
K8S_NS="mistudio"

# Base SSH command
k8s() {
  ~/.local/bin/sshpass -p "$K8S_PASS" ssh -o StrictHostKeyChecking=no ${K8S_USER}@${K8S_HOST} "$1"
}

# Check DockerHub for new image timestamps
k8s_check() {
  echo "=== DockerHub Image Timestamps ==="
  echo -n "Frontend: " && curl -s "https://hub.docker.com/v2/repositories/hitsai/mistudio-frontend/tags/?page_size=1" | python3 -c "import sys,json; d=json.load(sys.stdin); t=d.get('results',[{}])[0]; print(t.get('last_updated','?'))"
  echo -n "Backend:  " && curl -s "https://hub.docker.com/v2/repositories/hitsai/mistudio-backend/tags/?page_size=1" | python3 -c "import sys,json; d=json.load(sys.stdin); t=d.get('results',[{}])[0]; print(t.get('last_updated','?'))"
  echo "Current:  $(date -u +%Y-%m-%dT%H:%M:%SZ)"
}

# Full deploy: pull + restart + wait + verify
k8s_deploy() {
  echo "=== Pulling images ===" && \
  k8s "docker pull hitsai/mistudio-frontend:latest && docker pull hitsai/mistudio-backend:latest" && \
  echo "=== Restarting deployments ===" && \
  k8s "kubectl rollout restart deployment/mistudio-backend deployment/mistudio-frontend -n $K8S_NS" && \
  echo "=== Waiting for backend rollout ===" && \
  k8s "kubectl rollout status deployment/mistudio-backend -n $K8S_NS --timeout=180s" && \
  echo "=== Waiting for frontend rollout ===" && \
  k8s "kubectl rollout status deployment/mistudio-frontend -n $K8S_NS --timeout=180s" && \
  echo "=== Pod Status ===" && \
  k8s "kubectl get pods -n $K8S_NS"
}

# Quick pod status
k8s_status() {
  k8s "kubectl get pods -n $K8S_NS -o wide"
}

# View backend logs
k8s_logs() {
  local lines=${1:-50}
  k8s "kubectl logs -n $K8S_NS deployment/mistudio-backend -c backend --tail=$lines"
}

# View celery worker logs
k8s_logs_celery() {
  local lines=${1:-50}
  k8s "kubectl logs -n $K8S_NS deployment/mistudio-backend -c celery-worker --tail=$lines"
}

# GPU status
k8s_gpu() {
  k8s "nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv"
}

# Get all resources
k8s_all() {
  k8s "kubectl get all -n $K8S_NS"
}

echo "K8s helpers loaded. Available commands:"
echo "  k8s_check   - Check DockerHub image timestamps"
echo "  k8s_deploy  - Full deploy (pull + restart + wait)"
echo "  k8s_status  - Quick pod status"
echo "  k8s_logs    - Backend logs (k8s_logs [lines])"
echo "  k8s_logs_celery - Celery worker logs"
echo "  k8s_gpu     - GPU status"
echo "  k8s_all     - All k8s resources"
echo "  k8s \"cmd\"   - Run any command on k8s host"
