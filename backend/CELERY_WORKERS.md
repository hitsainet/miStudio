# Celery Worker Configuration Guide

## Overview

miStudio uses a multi-queue Celery architecture to separate different types of workloads:

- **high_priority**: Quick operations and metadata updates
- **datasets**: Dataset downloads and ingestion (I/O-bound)
- **processing**: Tokenization and preprocessing (CPU-bound)
- **training**: SAE training (GPU-bound, low concurrency)
- **extraction**: Feature extraction (GPU-bound, medium concurrency)
- **low_priority**: Maintenance and background tasks

## ⚠️ Important: Queue Configuration

**Workers MUST be started with explicit queue configuration using the `-Q` flag.**

If you start a worker without the `-Q` flag, it will only listen to the default "celery" queue, and tasks routed to other queues (datasets, processing, etc.) will never be picked up!

### ❌ Wrong (will miss tasks)
```bash
celery -A src.core.celery_app worker --loglevel=info
```

### ✅ Correct
```bash
celery -A src.core.celery_app worker -Q high_priority,datasets,processing,training,extraction,low_priority -c 8 --loglevel=info
```

## Quick Start Options

### Option 1: Use the Startup Script (Recommended for Development)

```bash
cd backend
./start-celery-worker.sh
```

This automatically starts a worker listening to all queues with sensible defaults.

### Option 2: Use the Full Scripts (Recommended for Production)

```bash
# Start all workers with proper configuration
./scripts/start-workers.sh minimal

# Available profiles:
#   minimal    - Single worker for all queues (default, good for development)
#   full       - All specialized workers (requires more resources)
#   datasets   - Only dataset-related workers (downloads + processing)
#   training   - Only GPU workers (training + extraction)

# Stop all workers
./scripts/stop-workers.sh
```

### Option 3: Docker Compose (Production Deployment)

```bash
# Start with specialized workers
docker-compose -f docker-compose.dev.yml -f docker-compose.workers.yml up

# Each worker type runs in its own container with optimized concurrency
```

### Option 4: Manual Startup (Advanced)

```bash
cd backend
source venv/bin/activate

# Development: Single worker for all queues
celery -A src.core.celery_app worker \
    -Q high_priority,datasets,processing,training,extraction,low_priority \
    -c 8 \
    --loglevel=info

# Production: Specialized workers (run each in separate terminal/screen/tmux)
celery -A src.core.celery_app worker -Q high_priority -c 8 --loglevel=info --hostname=worker-high-priority@%h
celery -A src.core.celery_app worker -Q datasets -c 4 --loglevel=info --hostname=worker-datasets@%h
celery -A src.core.celery_app worker -Q processing -c 4 --loglevel=info --hostname=worker-processing@%h
celery -A src.core.celery_app worker -Q training -c 1 --loglevel=info --hostname=worker-training@%h --max-memory-per-child=4000000
celery -A src.core.celery_app worker -Q extraction -c 2 --loglevel=info --hostname=worker-extraction@%h --max-memory-per-child=2000000
celery -A src.core.celery_app worker -Q low_priority -c 2 --loglevel=info --hostname=worker-low-priority@%h
```

## Task Routing Reference

| Task | Queue | Priority | Notes |
|------|-------|----------|-------|
| `download_dataset_task` | datasets | 7 | I/O-bound, medium concurrency |
| `tokenize_dataset_task` | datasets | 7 | CPU-bound, uses multiprocessing |
| SAE training tasks | training | 5 | GPU-bound, low concurrency |
| Feature extraction tasks | extraction | 5 | GPU-bound, medium concurrency |
| Quick tasks | high_priority | 10 | Fast operations |
| Maintenance tasks | low_priority | 3 | Background cleanup |

## Monitoring Workers

### Check Active Workers
```bash
celery -A src.core.celery_app inspect active
```

### Check Queue Lengths
```bash
celery -A src.core.celery_app inspect stats
```

### Check Worker Configuration
```bash
celery -A src.core.celery_app inspect active_queues
```

## Troubleshooting

### Problem: Tasks are not being processed

**Symptom**: You submit a tokenization job, but nothing happens. No progress bar, no errors.

**Cause**: Worker is not listening to the queue where the task was routed.

**Solution**:
1. Check worker configuration: `celery -A src.core.celery_app inspect active_queues`
2. Restart workers with correct queue configuration (see Quick Start options above)

### Problem: Tasks are stuck in PENDING state

**Symptom**: Tasks show as PENDING indefinitely in task results.

**Cause**: No worker is consuming from the target queue.

**Solution**: Start a worker listening to all queues (see Option 1 or 2 above)

### Problem: "Connection refused" errors

**Symptom**: Worker can't connect to Redis.

**Cause**: Redis server not running.

**Solution**:
```bash
# Check if Redis is running
redis-cli ping

# Start Redis (if using Docker)
docker-compose -f docker-compose.dev.yml up redis

# Start Redis (if using system service)
sudo systemctl start redis
```

## Configuration Files

- **Backend**: `backend/src/core/celery_app.py` - Main Celery app configuration
- **Docker**: `docker-compose.workers.yml` - Multi-worker Docker Compose configuration
- **Scripts**: `scripts/start-workers.sh` - Production-ready worker startup script
- **Quick Start**: `backend/start-celery-worker.sh` - Simple development startup script

## Development Tips

### Testing Queue Routing

```python
# In Python shell or notebook
from backend.src.core.celery_app import celery_app
from backend.src.workers.dataset_tasks import tokenize_dataset_task

# Check task routing configuration
print(celery_app.conf.task_routes)

# Manually inspect where a task will be routed
result = tokenize_dataset_task.apply_async(args=[...])
print(f"Task ID: {result.id}")
print(f"Task State: {result.state}")
```

### Viewing Logs

```bash
# If using scripts/start-workers.sh
tail -f backend/logs/worker-*.log

# If running manually, logs go to stdout
```

## Best Practices

1. **Development**: Use `./backend/start-celery-worker.sh` or `./scripts/start-workers.sh minimal`
2. **Production**: Use Docker Compose or systemd with specialized workers
3. **Always** specify queues explicitly with `-Q` flag when starting workers manually
4. **Monitor** worker health and queue lengths regularly
5. **Scale** by adding more workers for specific queues (e.g., multiple dataset workers)

## References

- [Celery Documentation - Routing Tasks](https://docs.celeryproject.org/en/stable/userguide/routing.html)
- [miStudio Architecture Documentation](../0xcc/adrs/000_PADR|miStudio.md)
- [Task Configuration](./src/core/celery_app.py)
