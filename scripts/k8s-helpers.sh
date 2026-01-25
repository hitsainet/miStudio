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
# Build times: Frontend ~2-3 min, Backend ~9 min (after sync ~1 min)
k8s_check() {
  echo "=== DockerHub Image Timestamps (Frontend ~3min, Backend ~9min after push) ==="
  echo -n "Frontend: " && curl -s "https://hub.docker.com/v2/repositories/hitsai/mistudio-frontend/tags/?page_size=1" | python3 -c "import sys,json; d=json.load(sys.stdin); t=d.get('results',[{}])[0]; print(t.get('last_updated','?'))"
  echo -n "Backend:  " && curl -s "https://hub.docker.com/v2/repositories/hitsai/mistudio-backend/tags/?page_size=1" | python3 -c "import sys,json; d=json.load(sys.stdin); t=d.get('results',[{}])[0]; print(t.get('last_updated','?'))"
  echo "Current:  $(date -u +%Y-%m-%dT%H:%M:%SZ)"
}

# Verify database schema
k8s_schema() {
  echo "=== Verifying Database Schema ==="
  k8s "kubectl exec -n $K8S_NS deployment/mistudio-backend -c backend -- python scripts/verify_schema.py"
}

# Fix missing database tables
k8s_schema_fix() {
  echo "=== Fixing Missing Database Tables ==="
  k8s "kubectl exec -n $K8S_NS deployment/mistudio-backend -c backend -- python scripts/verify_schema.py --fix"
}

# Run database migrations
k8s_migrate() {
  echo "=== Running Database Migrations ==="
  k8s "kubectl exec -n $K8S_NS deployment/mistudio-backend -c backend -- python -m alembic upgrade head"
}

# Manifest location on k8s host
K8S_MANIFEST="/home/sean/app/k8s-mistudio.mcslab.io/mistudio-deployment.yaml"

# Apply the deployment manifest
k8s_apply() {
  echo "=== Applying manifest from $K8S_MANIFEST ==="
  k8s "kubectl apply -f $K8S_MANIFEST"
}

# Full deploy: pull + apply manifest + restart + wait + verify schema
k8s_deploy() {
  echo "=== Pulling images ===" && \
  k8s "docker pull hitsai/mistudio-frontend:latest && docker pull hitsai/mistudio-backend:latest" && \
  echo "=== Applying manifest ===" && \
  k8s "kubectl apply -f $K8S_MANIFEST" && \
  echo "=== Restarting deployments ===" && \
  k8s "kubectl rollout restart deployment/mistudio-backend deployment/mistudio-frontend -n $K8S_NS" && \
  echo "=== Waiting for backend rollout ===" && \
  k8s "kubectl rollout status deployment/mistudio-backend -n $K8S_NS --timeout=180s" && \
  echo "=== Waiting for frontend rollout ===" && \
  k8s "kubectl rollout status deployment/mistudio-frontend -n $K8S_NS --timeout=180s" && \
  echo "=== Pod Status ===" && \
  k8s "kubectl get pods -n $K8S_NS" && \
  echo "" && \
  echo "=== Post-Deploy Schema Verification ===" && \
  k8s "kubectl exec -n $K8S_NS deployment/mistudio-backend -c backend -- python scripts/verify_schema.py" || \
  echo "WARNING: Schema verification failed. Run 'k8s_schema_fix' to attempt auto-fix."
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

# ===========================
# NEURONPEDIA HELPERS
# ===========================
NP_NS="neuronpedia"

# Neuronpedia pod status
np_status() {
  k8s "kubectl get pods -n $NP_NS"
}

# Neuronpedia webapp logs
np_logs() {
  local lines=${1:-50}
  k8s "kubectl logs -n $NP_NS deployment/neuronpedia-webapp --tail=$lines"
}

# Neuronpedia schema check - list expected vs actual columns
np_schema() {
  echo "=== Neuronpedia Schema Check ==="
  k8s "kubectl exec -n $NP_NS neuronpedia-postgres-0 -- psql -U neuronpedia -d neuronpedia -c '
    SELECT table_name, column_name
    FROM information_schema.columns
    WHERE table_schema = '\\''public'\\''
    ORDER BY table_name, ordinal_position
  ' | head -100"
}

# Neuronpedia data summary
np_data() {
  echo "=== Neuronpedia Data Summary ==="
  k8s "kubectl exec -n $NP_NS neuronpedia-postgres-0 -- psql -U neuronpedia -d neuronpedia -c '
    SELECT
      (SELECT COUNT(*) FROM \"Model\") as models,
      (SELECT COUNT(*) FROM \"Neuron\") as neurons,
      (SELECT COUNT(*) FROM \"Activation\") as activations,
      (SELECT COUNT(*) FROM \"Explanation\") as explanations,
      (SELECT COUNT(*) FROM \"Source\") as sources,
      (SELECT COUNT(*) FROM \"SourceSet\") as source_sets
  '"
}

# Neuronpedia purge all model data
np_purge() {
  echo "=== Purging Neuronpedia Model Data ==="
  read -p "Are you sure you want to purge ALL model data? (y/N) " confirm
  if [[ "$confirm" == "y" || "$confirm" == "Y" ]]; then
    k8s "kubectl exec -n $NP_NS neuronpedia-postgres-0 -- psql -U neuronpedia -d neuronpedia -c '
      UPDATE \"Model\" SET \"defaultSourceId\" = NULL, \"defaultSourceSetName\" = NULL;
      DELETE FROM \"Source\";
      DELETE FROM \"Model\";
    '"
    echo "Purge complete."
    np_data
  else
    echo "Purge cancelled."
  fi
}

# Neuronpedia restart webapp
np_restart() {
  echo "=== Restarting Neuronpedia Webapp ==="
  k8s "kubectl rollout restart deployment/neuronpedia-webapp -n $NP_NS && kubectl rollout status deployment/neuronpedia-webapp -n $NP_NS --timeout=120s"
}

# Neuronpedia add missing column (for schema fixes)
# Usage: np_add_column "TableName" "columnName" "TEXT"
np_add_column() {
  local table="$1"
  local column="$2"
  local type="${3:-TEXT}"
  echo "=== Adding column $column to $table ==="
  k8s "kubectl exec -n $NP_NS neuronpedia-postgres-0 -- psql -U neuronpedia -d neuronpedia -c 'ALTER TABLE \"$table\" ADD COLUMN IF NOT EXISTS \"$column\" $type;'"
}

# Full Neuronpedia deploy with schema sync
np_deploy() {
  echo "=== Pulling Neuronpedia image ===" && \
  k8s "docker pull hitsai/neuronpedia-webapp:latest" && \
  echo "=== Restarting deployment ===" && \
  k8s "kubectl rollout restart deployment/neuronpedia-webapp -n $NP_NS" && \
  echo "=== Waiting for rollout ===" && \
  k8s "kubectl rollout status deployment/neuronpedia-webapp -n $NP_NS --timeout=180s" && \
  echo "=== Pod Status ===" && \
  k8s "kubectl get pods -n $NP_NS" && \
  echo "" && \
  echo "=== Data Summary ===" && \
  np_data
}

echo "K8s helpers loaded. Available commands:"
echo ""
echo "  === MISTUDIO ==="
echo "  k8s_check       - Check DockerHub image timestamps"
echo "  k8s_deploy      - Full deploy (pull + apply + restart + verify schema)"
echo "  k8s_apply       - Apply deployment manifest (no restart)"
echo "  k8s_schema      - Verify database schema"
echo "  k8s_schema_fix  - Fix missing database tables"
echo "  k8s_migrate     - Run database migrations"
echo "  k8s_status      - Quick pod status"
echo "  k8s_logs        - Backend logs (k8s_logs [lines])"
echo "  k8s_logs_celery - Celery worker logs"
echo "  k8s_gpu         - GPU status"
echo "  k8s_all         - All k8s resources"
echo "  k8s \"cmd\"       - Run any command on k8s host"
echo ""
echo "  === NEURONPEDIA ==="
echo "  np_status       - Neuronpedia pod status"
echo "  np_logs         - Neuronpedia webapp logs"
echo "  np_schema       - Check database schema"
echo "  np_data         - Show data summary (counts)"
echo "  np_purge        - Purge all model data (with confirmation)"
echo "  np_restart      - Restart webapp"
echo "  np_add_column   - Add missing column: np_add_column Table column TYPE"
echo "  np_deploy       - Full deploy (pull + restart + wait)"
