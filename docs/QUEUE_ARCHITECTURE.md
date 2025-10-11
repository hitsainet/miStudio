# miStudio Queue Architecture

## Overview

miStudio uses a multi-queue Celery architecture to optimize resource allocation and ensure predictable task execution times. This document describes the queue system, worker configuration, and scaling strategies.

## Queue Design

### Queue Types

| Queue Name | Purpose | Concurrency | Priority | Resource Type |
|-----------|---------|-------------|----------|---------------|
| `high_priority` | Quick metadata operations, user-facing updates | 8 | 10 | CPU (light) |
| `datasets` | Dataset downloads, ingestion from HuggingFace | 4 | 7 | I/O, Network |
| `processing` | Tokenization, preprocessing, data transformations | 4 | 7 | CPU, Memory |
| `training` | SAE training jobs (long-running) | 1 | 5 | GPU, Memory |
| `extraction` | Feature extraction, activation collection | 2 | 5 | GPU, Memory |
| `low_priority` | Cleanup, archival, maintenance tasks | 2 | 3 | CPU (light) |

### Task Routing

Tasks are automatically routed to appropriate queues based on their type:

```python
# In backend/src/core/celery_app.py
task_routes={
    "src.workers.dataset_tasks.download_dataset_task": {"queue": "datasets"},
    "src.workers.dataset_tasks.tokenize_dataset_task": {"queue": "processing"},
    "src.workers.training_tasks.*": {"queue": "training"},
    "src.workers.extraction_tasks.*": {"queue": "extraction"},
    "src.workers.quick_tasks.*": {"queue": "high_priority"},
    "src.workers.maintenance_tasks.*": {"queue": "low_priority"},
}
```

## Worker Configurations

### Development (Single Worker)

For development, run a single worker that handles all queues:

```bash
# Automatic (recommended)
./scripts/start-workers.sh minimal

# Manual
celery -A src.core.celery_app worker \
    -Q high_priority,datasets,processing,training,extraction,low_priority \
    -c 8 \
    --loglevel=info
```

### Production (Specialized Workers)

For production, run multiple specialized workers:

```bash
# Start all workers
./scripts/start-workers.sh full

# Or start individually:
celery -A src.core.celery_app worker -Q high_priority -c 8 --hostname=worker-high-priority@%h
celery -A src.core.celery_app worker -Q datasets -c 4 --hostname=worker-datasets@%h
celery -A src.core.celery_app worker -Q processing -c 4 --hostname=worker-processing@%h
celery -A src.core.celery_app worker -Q training -c 1 --max-memory-per-child=4000000 --hostname=worker-training@%h
celery -A src.core.celery_app worker -Q extraction -c 2 --max-memory-per-child=2000000 --hostname=worker-extraction@%h
celery -A src.core.celery_app worker -Q low_priority -c 2 --hostname=worker-low-priority@%h
```

### Docker Compose (Multi-Worker)

Use Docker Compose for containerized deployment:

```bash
# Start infrastructure + all workers
docker-compose -f docker-compose.dev.yml -f docker-compose.workers.yml up -d

# Scale specific worker types
docker-compose -f docker-compose.dev.yml -f docker-compose.workers.yml up -d --scale worker-datasets=3
```

## Monitoring

### API Endpoints

Check worker and queue status via REST API:

```bash
# Queue lengths (pending tasks)
curl http://localhost:8000/api/v1/workers/queues

# Active tasks
curl http://localhost:8000/api/v1/workers/active

# Worker statistics
curl http://localhost:8000/api/v1/workers/stats

# Overall health
curl http://localhost:8000/api/v1/workers/health
```

### Example Responses

**Queue Lengths:**
```json
{
  "high_priority": 0,
  "datasets": 2,
  "processing": 1,
  "training": 1,
  "extraction": 0,
  "low_priority": 0
}
```

**Health Check:**
```json
{
  "status": "healthy",
  "workers_connected": 6,
  "total_queued": 4,
  "total_active": 2,
  "queues": {
    "high_priority": {"pending": 0, "status": "ok"},
    "datasets": {"pending": 2, "status": "ok"},
    "processing": {"pending": 1, "status": "ok"},
    "training": {"pending": 1, "status": "ok"},
    "extraction": {"pending": 0, "status": "ok"},
    "low_priority": {"pending": 0, "status": "ok"}
  },
  "warnings": []
}
```

## Resource Requirements

### Minimum Resources (Development)

- **CPU**: 4 cores
- **RAM**: 8 GB
- **GPU**: Optional (training/extraction will be slow without it)

### Recommended Resources (Production)

| Worker Type | CPU | RAM | GPU | Notes |
|------------|-----|-----|-----|-------|
| high-priority | 2 cores | 2 GB | No | Light operations |
| datasets | 2 cores | 4 GB | No | I/O bound, network dependent |
| processing | 4 cores | 8 GB | No | CPU intensive |
| training | 2 cores | 16 GB | 1x GPU | Memory intensive, long-running |
| extraction | 2 cores | 8 GB | 1x GPU | GPU intensive |
| low-priority | 1 core | 2 GB | No | Background tasks |

**Total (all workers):** 13 cores, 42 GB RAM, 2 GPUs

### Jetson Deployment (Resource-Constrained)

For NVIDIA Jetson devices with limited resources:

```bash
# Datasets profile: Download and processing only
./scripts/start-workers.sh datasets

# Or minimal: Single worker, lower concurrency
celery -A src.core.celery_app worker \
    -Q high_priority,datasets,processing,training,extraction,low_priority \
    -c 4 \
    --loglevel=info
```

## Scaling Strategies

### Horizontal Scaling

Add more workers to increase throughput:

```bash
# Start additional dataset workers on separate machines
celery -A src.core.celery_app worker -Q datasets -c 4 --hostname=worker-datasets-2@%h
celery -A src.core.celery_app worker -Q datasets -c 4 --hostname=worker-datasets-3@%h
```

### Vertical Scaling

Increase concurrency for specific queues:

```bash
# More concurrent downloads (up to network/disk limits)
celery -A src.core.celery_app worker -Q datasets -c 8

# More concurrent preprocessing
celery -A src.core.celery_app worker -Q processing -c 8
```

### Priority Tuning

Adjust task priorities in `backend/src/core/celery_app.py`:

```python
task_routes={
    "src.workers.urgent_task": {
        "queue": "high_priority",
        "priority": 10,  # 0 (low) to 10 (high)
    },
}
```

## Task Time Limits

Configure time limits to prevent runaway tasks:

```python
# In celery_app.py
task_soft_time_limit=3600,  # 1 hour - send warning
task_time_limit=7200,       # 2 hours - force kill
```

Per-task limits:

```python
@celery_app.task(
    soft_time_limit=1800,  # 30 minutes
    time_limit=3600,       # 1 hour
)
def long_running_task():
    pass
```

## Troubleshooting

### Workers Not Picking Up Tasks

**Symptom:** Tasks stay in queue but don't execute

**Causes:**
1. No worker listening to the queue
2. Workers crashed/not running
3. Queue routing misconfiguration

**Solution:**
```bash
# Check worker status
curl http://localhost:8000/api/v1/workers/stats

# Restart workers
./scripts/stop-workers.sh
./scripts/start-workers.sh full

# Check Redis queues manually
docker exec mistudio-redis redis-cli LLEN datasets
```

### High Memory Usage

**Symptom:** Workers consuming excessive RAM

**Solution:**
```bash
# Limit memory per child process
celery -A src.core.celery_app worker \
    -Q training \
    --max-memory-per-child=4000000  # 4 GB in KB

# Restart workers after N tasks
celery -A src.core.celery_app worker \
    -Q processing \
    --max-tasks-per-child=100
```

### Queue Backlog

**Symptom:** Many tasks pending, long wait times

**Solution:**
```bash
# Add more workers (horizontal scaling)
celery -A src.core.celery_app worker -Q datasets -c 4 --hostname=worker-datasets-2@%h

# Or increase concurrency (vertical scaling)
celery -A src.core.celery_app worker -Q datasets -c 8
```

### Task Failures

**Symptom:** Tasks failing with errors

**Solution:**
1. Check worker logs: `tail -f backend/logs/worker-*.log`
2. Check task result: `curl http://localhost:8000/api/v1/workers/active`
3. Review error in database: Dataset/model `error_message` field
4. Check resource availability (disk space, memory, GPU)

## Best Practices

1. **Development**: Use `minimal` profile (single worker, all queues)
2. **Production**: Use specialized workers per queue
3. **GPU Jobs**: Always use `max-memory-per-child` to prevent memory leaks
4. **Monitoring**: Set up alerts on queue lengths > 100
5. **Scaling**: Add workers before increasing concurrency
6. **Resource Limits**: Always set task time limits
7. **Logging**: Use structured logging with task IDs
8. **Testing**: Test with realistic data sizes
9. **Deployment**: Use systemd or Docker for automatic restarts
10. **Upgrades**: Gracefully stop workers before code updates

## Configuration Summary

### Key Files

- `backend/src/core/celery_app.py` - Queue configuration
- `docker-compose.workers.yml` - Docker worker definitions
- `scripts/start-workers.sh` - Local worker management
- `scripts/stop-workers.sh` - Worker shutdown

### Environment Variables

```bash
CELERY_BROKER_URL=redis://localhost:6379/0
CELERY_RESULT_BACKEND=redis://localhost:6379/0
```

### Worker Naming Convention

- `worker-{queue-name}@{hostname}` - Specialized worker
- `all-queues@{hostname}` - Development worker

## Migration Guide

### From Single Queue to Multi-Queue

Already done! The system is now using multi-queue architecture. If you need to revert:

1. Edit `backend/src/core/celery_app.py`
2. Comment out `task_routes` configuration
3. Set `task_default_queue="celery"`
4. Restart workers

### Adding New Queues

1. Add queue to `task_routes` in `celery_app.py`
2. Update `get_queue_lengths()` queue list
3. Add worker configuration to `docker-compose.workers.yml`
4. Update `scripts/start-workers.sh` profiles
5. Update this documentation

## References

- [Celery Documentation](https://docs.celeryq.dev/)
- [Redis Queue Documentation](https://redis.io/docs/data-types/lists/)
- [Celery Best Practices](https://docs.celeryq.dev/en/stable/userguide/tasks.html#best-practices)
