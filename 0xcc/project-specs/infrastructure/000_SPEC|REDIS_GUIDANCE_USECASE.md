# Redis Usage Guide for MechInterp Studio (miStudio)

**Document Type**: Technical Specification
**Last Updated**: 2025-10-05
**Purpose**: Comprehensive guide to Redis implementation in MechInterp Studio

---

## Table of Contents

1. [Overview](#overview)
2. [Use Case 1: Job Queue Backend](#use-case-1-job-queue-backend)
3. [Use Case 2: WebSocket Pub/Sub](#use-case-2-websocket-pubsub-for-real-time-updates)
4. [Use Case 3: Rate Limiting](#use-case-3-rate-limiting)
5. [Use Case 4: Distributed Coordination](#use-case-4-distributed-coordination)
6. [Use Case 5: Application-Level Caching](#use-case-5-application-level-caching)
7. [Deployment Architecture](#deployment-architecture)
8. [Configuration](#redis-configuration-recommendations)
9. [Performance Specifications](#performance-characteristics)
10. [Implementation Examples](#implementation-examples)

---

## Overview

Redis serves as a critical infrastructure component in MechInterp Studio, providing fast in-memory data storage and messaging capabilities essential for edge AI workloads. This document outlines all Redis use cases and implementation patterns.

**Key Benefits for Edge AI:**
- Ultra-low latency (< 1ms) for real-time ML operations
- Efficient memory usage on resource-constrained hardware
- Simple deployment (single Docker container)
- Built-in persistence for job queue reliability

**Redis Version**: 7.0+ recommended

---

## Use Case 1: Job Queue Backend

### Purpose
Message broker and result backend for asynchronous task processing of long-running ML operations.

### Implementation Stack

**Python (Recommended)**:
```python
# Using Celery with Redis backend
from celery import Celery

app = Celery('mistudio',
             broker='redis://localhost:6379/0',
             backend='redis://localhost:6379/1')

@app.task(bind=True)
def train_sparse_autoencoder(self, training_id, config):
    # Long-running SAE training
    for step in range(config['total_steps']):
        # Update progress via Redis pub/sub
        self.update_state(state='PROGRESS',
                         meta={'step': step, 'loss': loss})
    return {'final_loss': loss, 'status': 'completed'}
```

**Node.js (Alternative)**:
```javascript
// Using BullMQ with Redis
import { Queue, Worker } from 'bullmq';

const trainingQueue = new Queue('training', {
  connection: { host: 'localhost', port: 6379 }
});

const worker = new Worker('training', async (job) => {
  // Process training job
  await trainModel(job.data);
}, { connection: { host: 'localhost', port: 6379 } });
```

### Queued Operations

| Operation | Estimated Duration | Priority | Concurrency Limit |
|-----------|-------------------|----------|-------------------|
| SAE Training | 1-12 hours | High | 1 (GPU limited) |
| Feature Extraction | 30 min - 2 hours | Medium | 1 (GPU limited) |
| Dataset Download | 5-60 minutes | Medium | 2 (network limited) |
| Model Download | 2-30 minutes | Medium | 2 (network limited) |
| Dataset Processing | 10-90 minutes | Low | 1 (CPU/disk limited) |

### Priority Queue Configuration

```python
# Celery priority routing
CELERY_TASK_ROUTES = {
    'tasks.train_sae': {'queue': 'gpu_high', 'priority': 10},
    'tasks.extract_features': {'queue': 'gpu_medium', 'priority': 7},
    'tasks.download_dataset': {'queue': 'io_medium', 'priority': 5},
    'tasks.process_dataset': {'queue': 'cpu_low', 'priority': 3}
}
```

### Redis Data Structures Used

```redis
# Job queue (Celery uses Redis lists)
LPUSH celery:queue:gpu_high '{"task":"train_sae","id":"tr_abc123",...}'
BRPOP celery:queue:gpu_high 1  # Worker blocks waiting for jobs

# Task results
SET celery-task-meta-tr_abc123 '{"status":"SUCCESS","result":{...}}' EX 86400

# Task progress
SET celery-task-progress-tr_abc123 '{"step":1234,"loss":0.342}' EX 3600
```

### Why Redis for Job Queue

- **Fast enqueue/dequeue**: < 1ms for job submission
- **Persistence**: Jobs survive worker restarts (with RDB/AOF)
- **Visibility timeout**: Automatic retry if worker crashes
- **Priority queues**: Critical for GPU resource management
- **Result backend**: Store training results for retrieval

---

## Use Case 2: WebSocket Pub/Sub for Real-Time Updates

### Purpose
Coordinate real-time messages across multiple server instances to enable horizontal scaling of WebSocket connections.

### Architecture

```
┌─────────────┐         ┌──────────┐         ┌────────────────┐
│ GPU Worker  │────────>│  Redis   │────────>│ WebSocket      │
│ (Training)  │ Publish │ Pub/Sub  │Subscribe│ Server(s)      │
└─────────────┘         └──────────┘         └────────────────┘
                             │                      │
                             │                      │ WebSocket
                             │                      ▼
                             │                ┌────────────┐
                             └───────────────>│  Browser   │
                                              │  Client    │
                                              └────────────┘
```

### Implementation

**Publishing (from Celery worker)**:
```python
import redis
import json

redis_client = redis.Redis(host='localhost', port=6379, db=2)

def publish_training_progress(training_id, progress_data):
    channel = f'training:{training_id}'
    message = json.dumps({
        'type': 'training.progress',
        'data': progress_data,
        'timestamp': datetime.utcnow().isoformat()
    })
    redis_client.publish(channel, message)

# During training loop
publish_training_progress('tr_abc123', {
    'step': 1234,
    'total_steps': 10000,
    'progress': 12.34,
    'metrics': {'loss': 0.342, 'sparsity': 12.4},
    'estimated_remaining_seconds': 427
})
```

**Subscribing (from WebSocket server)**:
```python
import asyncio
import websockets
import redis

redis_client = redis.Redis(host='localhost', port=6379, db=2)
pubsub = redis_client.pubsub()

async def handle_websocket(websocket, training_id):
    # Subscribe to training channel
    channel = f'training:{training_id}'
    pubsub.subscribe(channel)

    # Forward messages to WebSocket client
    for message in pubsub.listen():
        if message['type'] == 'message':
            await websocket.send(message['data'])
```

**Socket.IO Integration**:
```python
from socketio import AsyncServer
import redis.asyncio as aioredis

sio = AsyncServer(async_mode='asgi', client_manager=...)

# Use Redis for Socket.IO pub/sub
redis_url = 'redis://localhost:6379/2'
mgr = socketio.AsyncRedisManager(redis_url)
sio = AsyncServer(client_manager=mgr)
```

### Message Types

| Message Type | Channel Pattern | Payload Example |
|--------------|----------------|-----------------|
| Training Progress | `training:{id}` | `{"step": 1234, "loss": 0.342}` |
| Training Complete | `training:{id}` | `{"final_loss": 0.234, "duration": 3600}` |
| Training Error | `training:{id}` | `{"error": "GPU_OOM", "message": "..."}` |
| Feature Extraction | `features:{training_id}` | `{"extracted": 1024, "total": 16384}` |
| System Status | `system:status` | `{"gpu_util": 87, "memory": 4096}` |

### Channel Naming Convention

```
training:{training_id}           # Specific training job updates
features:{training_id}            # Feature extraction for training
dataset:{dataset_id}:download     # Dataset download progress
model:{model_id}:download         # Model download progress
system:status                     # System-wide status broadcasts
system:alerts                     # Critical system alerts
```

### Why Redis Pub/Sub

- **Ultra-low latency**: < 1ms message delivery
- **Fan-out efficiency**: One publish reaches all subscribers instantly
- **Channel isolation**: Each training has its own channel
- **No message persistence**: Perfect for real-time (messages don't need storage)
- **Horizontal scaling**: Multiple WebSocket servers can all subscribe

---

## Use Case 3: Rate Limiting

### Purpose
Protect system resources (GPU, network, disk) from overuse and ensure fair resource allocation.

### Rate Limit Tiers

```yaml
Global API Rate Limits:
  requests_per_minute: 100
  burst_allowance: 10

Training Operations:
  concurrent_training_jobs: 1  # GPU limitation
  training_starts_per_hour: 5

Download Operations:
  downloads_per_hour: 10
  concurrent_downloads: 2

Feature Steering:
  generations_per_hour: 20
  concurrent_generations: 1

Dataset Processing:
  concurrent_processing: 1
  processing_per_hour: 5
```

### Implementation: Sliding Window Algorithm

**Python Implementation**:
```python
import redis
import time

class RateLimiter:
    def __init__(self, redis_client):
        self.redis = redis_client

    def check_rate_limit(self, key, limit, window_seconds):
        """
        Sliding window rate limiter using Redis sorted sets.
        Returns (allowed: bool, remaining: int, reset_at: int)
        """
        now = time.time()
        window_start = now - window_seconds

        pipe = self.redis.pipeline()

        # Remove old entries outside the window
        pipe.zremrangebyscore(key, 0, window_start)

        # Count entries in current window
        pipe.zcard(key)

        # Add current request with timestamp as score
        pipe.zadd(key, {str(now): now})

        # Set expiration
        pipe.expire(key, window_seconds + 1)

        results = pipe.execute()
        current_count = results[1]

        if current_count < limit:
            return (True, limit - current_count - 1, int(now + window_seconds))
        else:
            # Get oldest entry to calculate reset time
            oldest = self.redis.zrange(key, 0, 0, withscores=True)
            if oldest:
                reset_at = int(oldest[0][1] + window_seconds)
            else:
                reset_at = int(now + window_seconds)
            return (False, 0, reset_at)

# Usage in FastAPI endpoint
@app.get("/api/datasets")
async def list_datasets():
    limiter = RateLimiter(redis_client)
    allowed, remaining, reset_at = limiter.check_rate_limit(
        key="ratelimit:global:api",
        limit=100,
        window_seconds=60
    )

    if not allowed:
        raise HTTPException(
            status_code=429,
            headers={
                "X-RateLimit-Limit": "100",
                "X-RateLimit-Remaining": "0",
                "X-RateLimit-Reset": str(reset_at),
                "Retry-After": str(reset_at - int(time.time()))
            }
        )

    # Process request...
    return JSONResponse(
        content={"data": datasets},
        headers={
            "X-RateLimit-Limit": "100",
            "X-RateLimit-Remaining": str(remaining),
            "X-RateLimit-Reset": str(reset_at)
        }
    )
```

### Concurrent Job Limiting

**Training Concurrency Lock**:
```python
def start_training(training_id, config):
    """Ensure only 1 training job runs at a time (GPU constraint)"""
    lock_key = "lock:training:active"

    # Try to acquire lock with 24-hour expiration
    acquired = redis_client.set(
        lock_key,
        training_id,
        nx=True,  # Only set if not exists
        ex=86400  # 24 hour max training time
    )

    if not acquired:
        current_training = redis_client.get(lock_key)
        raise HTTPException(
            status_code=503,
            detail=f"GPU busy with training {current_training.decode()}"
        )

    try:
        # Start training
        result = train_sae.delay(training_id, config)
        return {"job_id": training_id, "status": "queued"}
    except Exception as e:
        # Release lock on error
        redis_client.delete(lock_key)
        raise
```

### Redis Data Structures

```redis
# Sliding window rate limit (sorted set with timestamps)
ZADD ratelimit:global:api 1633024800.123 "1633024800.123"
ZADD ratelimit:global:api 1633024801.456 "1633024801.456"
ZREMRANGEBYSCORE ratelimit:global:api 0 1633024740  # Remove old entries
ZCARD ratelimit:global:api  # Count current entries

# Concurrent training lock
SET lock:training:active "tr_abc123" NX EX 86400

# Download concurrency counter
INCR counter:downloads:active
DECR counter:downloads:active
GET counter:downloads:active
```

### Why Redis for Rate Limiting

- **Atomic operations**: ZADD, ZCARD, etc. prevent race conditions
- **TTL support**: Automatic cleanup of old rate limit data
- **Sorted sets**: Efficient sliding window implementation
- **Distributed**: Works across multiple API servers
- **Fast**: < 1ms latency adds negligible overhead

---

## Use Case 4: Distributed Coordination

### Purpose
Coordinate state and resource allocation across multiple Jetson devices in a cluster deployment.

### Multi-Device Cluster Scenario

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│ Jetson #1    │     │ Jetson #2    │     │ Jetson #3    │
│ (Master)     │     │ (Worker)     │     │ (Worker)     │
│              │     │              │     │              │
│ - API Server │     │ - GPU Worker │     │ - GPU Worker │
│ - Redis      │────>│ - API Server │     │ - API Server │
│ - Postgres   │     └──────────────┘     └──────────────┘
└──────────────┘            │                     │
       │                    │                     │
       └────────────────────┴─────────────────────┘
                     Shared Redis
```

### Distributed Locking

**Acquiring GPU Resource**:
```python
import uuid
from contextlib import contextmanager

@contextmanager
def acquire_gpu_lock(device_id, timeout=300):
    """
    Distributed lock for GPU resource.
    Uses Redlock algorithm for safety.
    """
    lock_key = f"lock:gpu:{device_id}"
    lock_value = str(uuid.uuid4())  # Unique token

    # Try to acquire lock
    acquired = redis_client.set(
        lock_key,
        lock_value,
        nx=True,
        ex=timeout
    )

    if not acquired:
        raise ResourceBusyError(f"GPU {device_id} is busy")

    try:
        yield device_id
    finally:
        # Release lock only if we still own it
        lua_script = """
        if redis.call("get", KEYS[1]) == ARGV[1] then
            return redis.call("del", KEYS[1])
        else
            return 0
        end
        """
        redis_client.eval(lua_script, 1, lock_key, lock_value)

# Usage
with acquire_gpu_lock("jetson_2"):
    train_model()  # GPU safely locked
```

### Worker Health Monitoring

**Worker Heartbeat**:
```python
import threading
import time

def send_heartbeat(worker_id, interval=10):
    """Send heartbeat every 10 seconds"""
    while True:
        redis_client.setex(
            f"worker:{worker_id}:heartbeat",
            30,  # 30 second TTL
            json.dumps({
                "status": "alive",
                "gpu_util": get_gpu_utilization(),
                "memory_free": get_free_memory(),
                "active_jobs": get_active_jobs()
            })
        )
        time.sleep(interval)

# Start heartbeat thread
heartbeat_thread = threading.Thread(
    target=send_heartbeat,
    args=("jetson_2",),
    daemon=True
)
heartbeat_thread.start()
```

**Health Check**:
```python
def get_healthy_workers():
    """Return list of workers with recent heartbeat"""
    workers = []
    for key in redis_client.scan_iter("worker:*:heartbeat"):
        worker_id = key.decode().split(':')[1]
        heartbeat_data = redis_client.get(key)
        if heartbeat_data:
            workers.append({
                "worker_id": worker_id,
                "data": json.loads(heartbeat_data)
            })
    return workers
```

### Job Assignment Tracking

**Assign Job to Worker**:
```python
def assign_job(job_id, worker_id):
    """Track which worker is processing which job"""
    job_key = f"job:{job_id}:assignment"

    redis_client.hset(job_key, mapping={
        "worker_id": worker_id,
        "assigned_at": time.time(),
        "status": "running"
    })
    redis_client.expire(job_key, 86400)  # 24 hour max

    # Add to worker's job set
    redis_client.sadd(f"worker:{worker_id}:jobs", job_id)

def get_worker_jobs(worker_id):
    """Get all jobs assigned to a worker"""
    job_ids = redis_client.smembers(f"worker:{worker_id}:jobs")
    jobs = []
    for job_id in job_ids:
        job_data = redis_client.hgetall(f"job:{job_id.decode()}:assignment")
        jobs.append(job_data)
    return jobs
```

### Leader Election (Optional)

**Simple Leader Election**:
```python
def try_become_leader(node_id, ttl=30):
    """Try to become cluster leader"""
    acquired = redis_client.set(
        "cluster:leader",
        node_id,
        nx=True,
        ex=ttl
    )
    return acquired

def maintain_leadership(node_id):
    """Keep extending leadership while active"""
    while True:
        current_leader = redis_client.get("cluster:leader")
        if current_leader and current_leader.decode() == node_id:
            redis_client.expire("cluster:leader", 30)
        time.sleep(10)
```

### Redis Data Structures

```redis
# Distributed locks
SET lock:gpu:jetson_2 "uuid-12345" NX EX 300

# Worker heartbeats
SETEX worker:jetson_2:heartbeat 30 '{"status":"alive","gpu_util":87}'

# Job assignments
HSET job:tr_abc123:assignment worker_id "jetson_2" assigned_at "1633024800"
SADD worker:jetson_2:jobs "tr_abc123"

# Leader election
SET cluster:leader "jetson_1" NX EX 30
```

### Why Redis for Coordination

- **Atomic operations**: Critical for distributed locking
- **TTL**: Automatic lock release on worker crash
- **Fast**: < 2ms for lock acquire/release
- **Simple**: Easier than ZooKeeper/etcd for this use case
- **Lightweight**: Minimal memory footprint on edge devices

---

## Use Case 5: Application-Level Caching

### Purpose
Cache expensive computation results to improve API response times and reduce redundant processing.

### Cached Data Categories

#### 1. Feature Statistics (Expensive Aggregations)

**Cache Key Pattern**: `cache:features:{training_id}:stats`

```python
def get_feature_statistics(training_id):
    """Get aggregated feature stats with caching"""
    cache_key = f"cache:features:{training_id}:stats"

    # Try cache first
    cached = redis_client.get(cache_key)
    if cached:
        return json.loads(cached)

    # Compute from database (expensive)
    stats = db.query("""
        SELECT
            COUNT(*) as total_features,
            AVG(activation_frequency) as avg_activation,
            AVG(interpretability_score) as avg_interpretability,
            COUNT(CASE WHEN activation_frequency < 0.01 THEN 1 END) as dead_neurons
        FROM features
        WHERE training_id = %s
    """, training_id)

    # Cache for 1 hour
    redis_client.setex(cache_key, 3600, json.dumps(stats))
    return stats
```

**TTL**: 1 hour (updates during training)

#### 2. Model Metadata

**Cache Key Pattern**: `cache:model:{model_id}:metadata`

```python
def get_model_metadata(model_id):
    """Get model architecture info with caching"""
    cache_key = f"cache:model:{model_id}:metadata"

    # Try Redis hash
    cached = redis_client.hgetall(cache_key)
    if cached:
        return {k.decode(): v.decode() for k, v in cached.items()}

    # Load from local storage config.json (slower I/O)
    metadata = load_model_config_from_disk(model_id)

    # Cache as hash for 24 hours
    redis_client.hset(cache_key, mapping=metadata)
    redis_client.expire(cache_key, 86400)
    return metadata
```

**TTL**: 24 hours (rarely changes)

#### 3. Dataset Statistics

**Cache Key Pattern**: `cache:dataset:{dataset_id}:stats`

```python
def get_dataset_statistics(dataset_id):
    """Get dataset stats with caching"""
    cache_key = f"cache:dataset:{dataset_id}:stats"

    cached = redis_client.get(cache_key)
    if cached:
        return json.loads(cached)

    # Compute from parquet files (expensive)
    stats = {
        "total_tokens": count_tokens(dataset_id),
        "vocab_size": get_vocab_size(dataset_id),
        "samples": count_samples(dataset_id),
        "avg_seq_length": get_avg_sequence_length(dataset_id)
    }

    # Cache for 24 hours
    redis_client.setex(cache_key, 86400, json.dumps(stats))
    return stats
```

**TTL**: 24 hours (static after processing)

#### 4. Feature Correlation Matrices

**Cache Key Pattern**: `cache:correlations:{training_id}:matrix`

```python
def get_feature_correlations(training_id, top_k=100):
    """Get feature correlation matrix (computationally expensive)"""
    cache_key = f"cache:correlations:{training_id}:top{top_k}"

    cached = redis_client.get(cache_key)
    if cached:
        return json.loads(cached)

    # Compute correlations (very expensive: O(n²) with n=16384)
    correlations = compute_correlation_matrix(training_id, top_k)

    # Cache for 2 hours
    redis_client.setex(cache_key, 7200, json.dumps(correlations))
    return correlations
```

**TTL**: 2 hours (expensive to recompute)

#### 5. Checkpoint Lists

**Cache Key Pattern**: `cache:checkpoints:{training_id}:list`

```python
def list_checkpoints(training_id):
    """List available checkpoints with caching"""
    cache_key = f"cache:checkpoints:{training_id}:list"

    cached = redis_client.get(cache_key)
    if cached:
        return json.loads(cached)

    # Query from database and local storage
    checkpoints = get_checkpoints_from_storage(training_id)

    # Cache for 5 minutes (updates frequently during training)
    redis_client.setex(cache_key, 300, json.dumps(checkpoints))
    return checkpoints
```

**TTL**: 5 minutes (frequent updates during training)

### Cache Invalidation Strategies

**Manual Invalidation on Updates**:
```python
def save_checkpoint(training_id, checkpoint_data):
    """Save checkpoint and invalidate cache"""
    # Save to storage
    save_to_s3(checkpoint_data)

    # Invalidate checkpoint list cache
    redis_client.delete(f"cache:checkpoints:{training_id}:list")

    # Invalidate feature stats cache (checkpoint might affect stats)
    redis_client.delete(f"cache:features:{training_id}:stats")
```

**Time-Based TTL Strategy**:

| Data Type | TTL | Rationale |
|-----------|-----|-----------|
| Model metadata | 24 hours | Static after download |
| Dataset stats | 24 hours | Static after processing |
| Feature stats | 1 hour | Updates during training |
| Correlations | 2 hours | Expensive to compute |
| Checkpoint lists | 5 minutes | Frequent updates |
| API responses | 60 seconds | Balance freshness/performance |

**Cache-Aside Pattern**:
```python
def get_with_cache(key, compute_fn, ttl=3600):
    """Generic cache-aside helper"""
    # Try cache
    cached = redis_client.get(key)
    if cached:
        return json.loads(cached)

    # Compute
    value = compute_fn()

    # Store in cache
    redis_client.setex(key, ttl, json.dumps(value))
    return value
```

### Memory Management

**Eviction Policy**: `allkeys-lru` (Least Recently Used)

```conf
maxmemory 2gb
maxmemory-policy allkeys-lru
```

When Redis reaches memory limit:
1. Evicts least recently accessed keys first
2. Preserves job queue and rate limit data (accessed frequently)
3. Removes old cached statistics automatically

### Redis Data Structures

```redis
# Feature statistics (JSON string)
SET cache:features:tr_abc123:stats '{"total":16384,"avg":0.023}' EX 3600

# Model metadata (Hash)
HSET cache:model:gpt2_q4 architecture "GPT2" params "124M" layers "12"
EXPIRE cache:model:gpt2_q4 86400

# Dataset statistics (JSON string)
SET cache:dataset:openwebtext:stats '{"tokens":8038044935}' EX 86400

# Correlation matrix (compressed JSON)
SET cache:correlations:tr_abc123:top100 '[...]' EX 7200

# Checkpoint list (JSON array)
SET cache:checkpoints:tr_abc123:list '[{...},{...}]' EX 300
```

### Cache Hit Rate Monitoring

```python
def track_cache_stats(key, hit):
    """Track cache hit/miss rates"""
    stat_key = f"cache:stats:{key.split(':')[1]}"  # Extract cache type

    if hit:
        redis_client.hincrby(stat_key, "hits", 1)
    else:
        redis_client.hincrby(stat_key, "misses", 1)

    redis_client.expire(stat_key, 86400)

def get_cache_hit_rate(cache_type):
    """Calculate cache hit rate"""
    stat_key = f"cache:stats:{cache_type}"
    stats = redis_client.hgetall(stat_key)

    if not stats:
        return 0.0

    hits = int(stats.get(b'hits', 0))
    misses = int(stats.get(b'misses', 0))
    total = hits + misses

    return (hits / total * 100) if total > 0 else 0.0
```

### Why Redis for Caching

- **Sub-millisecond reads**: 100-1000x faster than database/disk I/O
- **Flexible TTL**: Per-key expiration
- **Rich data types**: Strings, hashes, sets for different use cases
- **LRU eviction**: Automatic memory management
- **Simple**: No complex cache invalidation logic needed

---

## Deployment Architecture

### Single Device Deployment (Jetson Orin Nano)

```yaml
services:
  redis:
    image: redis:7.2-alpine
    container_name: mistudio-redis
    ports:
      - "6379:6379"
    volumes:
      - ./redis-data:/data
      - ./redis.conf:/usr/local/etc/redis/redis.conf
    command: redis-server /usr/local/etc/redis/redis.conf
    restart: unless-stopped
    mem_limit: 512m
    networks:
      - mistudio-network

  api:
    depends_on:
      - redis
    environment:
      - REDIS_URL=redis://redis:6379/0
      - CELERY_BROKER_URL=redis://redis:6379/0
      - CELERY_RESULT_BACKEND=redis://redis:6379/1

  celery-worker:
    depends_on:
      - redis
    environment:
      - CELERY_BROKER_URL=redis://redis:6379/0
      - CELERY_RESULT_BACKEND=redis://redis:6379/1
```

**Database Allocation**:
- DB 0: Celery job queue (broker)
- DB 1: Celery results backend
- DB 2: Pub/sub channels
- DB 3: Rate limiting
- DB 4: Application cache
- DB 5: Distributed locks/coordination

### Multi-Device Cluster Deployment

```yaml
# On master node (Jetson #1)
services:
  redis:
    image: redis:7.2-alpine
    container_name: mistudio-redis-cluster
    ports:
      - "6379:6379"
    volumes:
      - ./redis-data:/data
      - ./redis-cluster.conf:/usr/local/etc/redis/redis.conf
    command: redis-server /usr/local/etc/redis/redis.conf
    restart: unless-stopped
    mem_limit: 2g
    networks:
      - cluster-network

# On worker nodes (Jetson #2, #3)
environment:
  - REDIS_URL=redis://jetson-1:6379/0
```

**Network Configuration**:
```yaml
networks:
  cluster-network:
    driver: bridge
    ipam:
      config:
        - subnet: 172.20.0.0/16
```

---

## Redis Configuration Recommendations

### Production Configuration (`redis.conf`)

```conf
# ==============================================================================
# REDIS CONFIGURATION FOR MECHINTERN STUDIO
# ==============================================================================

# NETWORK
bind 0.0.0.0
protected-mode yes
port 6379
tcp-backlog 511
timeout 0
tcp-keepalive 300

# SECURITY
requirepass your_strong_redis_password_here
# maxclients 10000

# MEMORY MANAGEMENT
maxmemory 2gb
maxmemory-policy allkeys-lru
maxmemory-samples 5

# PERSISTENCE - RDB Snapshots
save 900 1       # Save if 1 key changed in 15 minutes
save 300 10      # Save if 10 keys changed in 5 minutes
save 60 10000    # Save if 10000 keys changed in 1 minute
stop-writes-on-bgsave-error yes
rdbcompression yes
rdbchecksum yes
dbfilename dump.rdb
dir /data

# PERSISTENCE - Append Only File (AOF)
appendonly yes
appendfilename "appendonly.aof"
appendfsync everysec
no-appendfsync-on-rewrite no
auto-aof-rewrite-percentage 100
auto-aof-rewrite-min-size 64mb

# LAZY FREEING (better performance)
lazyfree-lazy-eviction yes
lazyfree-lazy-expire yes
lazyfree-lazy-server-del yes
replica-lazy-flush yes

# ADVANCED
databases 16
loglevel notice
logfile ""

# PUB/SUB
client-output-buffer-limit normal 0 0 0
client-output-buffer-limit replica 256mb 64mb 60
client-output-buffer-limit pubsub 32mb 8mb 60

# SLOW LOG (for debugging)
slowlog-log-slower-than 10000  # 10ms
slowlog-max-len 128
```

### Docker Compose Production Example

```yaml
version: '3.8'

services:
  redis:
    image: redis:7.2-alpine
    container_name: mistudio-redis
    restart: unless-stopped
    ports:
      - "6379:6379"
    volumes:
      - redis-data:/data
      - ./config/redis.conf:/usr/local/etc/redis/redis.conf:ro
    command: redis-server /usr/local/etc/redis/redis.conf
    environment:
      - REDIS_PASSWORD=${REDIS_PASSWORD}
    mem_limit: 2g
    mem_reservation: 512m
    cpus: 2
    healthcheck:
      test: ["CMD", "redis-cli", "-a", "${REDIS_PASSWORD}", "ping"]
      interval: 10s
      timeout: 3s
      retries: 3
      start_period: 10s
    networks:
      - mistudio-internal
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"

volumes:
  redis-data:
    driver: local

networks:
  mistudio-internal:
    driver: bridge
```

### Environment Variables

```bash
# .env file
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=your_secure_password_here
REDIS_DB_QUEUE=0
REDIS_DB_RESULTS=1
REDIS_DB_PUBSUB=2
REDIS_DB_RATELIMIT=3
REDIS_DB_CACHE=4
REDIS_DB_LOCKS=5
```

---

## Performance Characteristics

### Expected Load by Use Case

| Use Case | Operations/Second | Avg Latency | P99 Latency | Memory Usage |
|----------|------------------|-------------|-------------|--------------|
| Job Queue Enqueue | 10 | 0.5ms | 2ms | ~1MB/100 jobs |
| Job Queue Dequeue | 10 | 0.8ms | 3ms | - |
| Pub/Sub Publish | 50 | 0.3ms | 1ms | Negligible |
| Pub/Sub Receive | 50 | 0.3ms | 1ms | ~10KB/channel |
| Rate Limit Check | 100 | 0.5ms | 2ms | ~10KB/limit |
| Cache Read (Hit) | 50 | 0.3ms | 1ms | ~100MB total |
| Cache Write | 20 | 0.5ms | 2ms | - |
| Distributed Lock | 5 | 1ms | 5ms | ~1KB/lock |
| Worker Heartbeat | 0.1 (every 10s) | 0.5ms | 2ms | ~1KB/worker |

### Total System Load

**Expected aggregate Redis load**:
- **QPS**: ~300 operations/second (peak)
- **Memory**: 200MB typical, 500MB peak
- **Network**: < 1 MB/sec
- **CPU**: < 5% of single core

### Capacity Planning

**Single Jetson Orin Nano**:
```
Redis Allocation:
- Memory: 512MB reserved, 1GB max
- CPU: 0.5 cores max
- Disk: 100MB for persistence
```

**Multi-Device Cluster (3 Jetson devices)**:
```
Redis Allocation:
- Memory: 1GB reserved, 2GB max
- CPU: 1 core max
- Disk: 500MB for persistence
- Network: Gigabit Ethernet required
```

### Monitoring Metrics

```python
# Key metrics to track
def get_redis_metrics():
    info = redis_client.info()

    return {
        # Memory
        "used_memory_mb": info['used_memory'] / 1024 / 1024,
        "used_memory_peak_mb": info['used_memory_peak'] / 1024 / 1024,
        "mem_fragmentation_ratio": info['mem_fragmentation_ratio'],

        # Performance
        "total_commands_processed": info['total_commands_processed'],
        "instantaneous_ops_per_sec": info['instantaneous_ops_per_sec'],

        # Connections
        "connected_clients": info['connected_clients'],
        "blocked_clients": info['blocked_clients'],

        # Persistence
        "rdb_last_save_time": info['rdb_last_save_time'],
        "rdb_changes_since_last_save": info['rdb_changes_since_last_save'],

        # Replication
        "connected_slaves": info['connected_slaves'],

        # Stats
        "keyspace_hits": info['keyspace_hits'],
        "keyspace_misses": info['keyspace_misses'],
        "evicted_keys": info['evicted_keys'],
    }
```

### Alerting Thresholds

```yaml
alerts:
  memory_usage:
    warning: 70%  # 1.4GB of 2GB
    critical: 85%  # 1.7GB of 2GB

  cpu_usage:
    warning: 60%
    critical: 80%

  evicted_keys:
    warning: 100/min
    critical: 500/min

  blocked_clients:
    warning: 5
    critical: 10

  replication_lag:
    warning: 5s
    critical: 30s
```

---

## Implementation Examples

### Complete FastAPI Integration

```python
# app/redis_client.py
import redis.asyncio as aioredis
from typing import Optional
import json

class RedisClient:
    def __init__(self):
        self.redis: Optional[aioredis.Redis] = None

    async def connect(self, url: str):
        """Connect to Redis"""
        self.redis = await aioredis.from_url(
            url,
            encoding="utf-8",
            decode_responses=True
        )

    async def disconnect(self):
        """Disconnect from Redis"""
        if self.redis:
            await self.redis.close()

    # Cache operations
    async def cache_get(self, key: str):
        """Get from cache"""
        value = await self.redis.get(key)
        return json.loads(value) if value else None

    async def cache_set(self, key: str, value: dict, ttl: int = 3600):
        """Set cache with TTL"""
        await self.redis.setex(key, ttl, json.dumps(value))

    async def cache_delete(self, key: str):
        """Delete from cache"""
        await self.redis.delete(key)

    # Rate limiting
    async def check_rate_limit(self, key: str, limit: int, window: int):
        """Check rate limit using sliding window"""
        now = time.time()
        window_start = now - window

        pipe = self.redis.pipeline()
        pipe.zremrangebyscore(key, 0, window_start)
        pipe.zcard(key)
        pipe.zadd(key, {str(now): now})
        pipe.expire(key, window + 1)

        results = await pipe.execute()
        current_count = results[1]

        return current_count < limit

    # Pub/Sub
    async def publish(self, channel: str, message: dict):
        """Publish message to channel"""
        await self.redis.publish(channel, json.dumps(message))

    async def subscribe(self, channel: str):
        """Subscribe to channel"""
        pubsub = self.redis.pubsub()
        await pubsub.subscribe(channel)
        return pubsub

# app/main.py
from fastapi import FastAPI, HTTPException, Depends
from contextlib import asynccontextmanager

redis_client = RedisClient()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await redis_client.connect(os.getenv("REDIS_URL"))
    yield
    # Shutdown
    await redis_client.disconnect()

app = FastAPI(lifespan=lifespan)

@app.get("/api/features/{training_id}/stats")
async def get_feature_stats(training_id: str):
    """Get feature statistics with caching"""
    cache_key = f"cache:features:{training_id}:stats"

    # Try cache
    cached = await redis_client.cache_get(cache_key)
    if cached:
        return {"data": cached, "cached": True}

    # Compute from database
    stats = await compute_feature_statistics(training_id)

    # Cache for 1 hour
    await redis_client.cache_set(cache_key, stats, ttl=3600)

    return {"data": stats, "cached": False}

@app.post("/api/trainings")
async def start_training(config: TrainingConfig):
    """Start training with rate limiting"""
    # Check rate limit
    allowed = await redis_client.check_rate_limit(
        "ratelimit:training:starts",
        limit=5,
        window=3600
    )

    if not allowed:
        raise HTTPException(status_code=429, detail="Too many training requests")

    # Queue training job
    task = train_sae.delay(config.dict())

    return {"job_id": task.id, "status": "queued"}
```

### Celery Worker with Redis

```python
# app/celery_app.py
from celery import Celery
import redis

# Configure Celery with Redis
celery_app = Celery(
    'mistudio',
    broker=os.getenv('CELERY_BROKER_URL', 'redis://localhost:6379/0'),
    backend=os.getenv('CELERY_RESULT_BACKEND', 'redis://localhost:6379/1')
)

celery_app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    task_track_started=True,
    task_time_limit=86400,  # 24 hour max
    worker_prefetch_multiplier=1,  # One job at a time
    worker_max_tasks_per_child=10,  # Restart worker after 10 jobs
)

# Redis client for pub/sub
redis_client = redis.Redis(
    host=os.getenv('REDIS_HOST', 'localhost'),
    port=int(os.getenv('REDIS_PORT', 6379)),
    db=2,  # Pub/sub database
    decode_responses=True
)

@celery_app.task(bind=True)
def train_sparse_autoencoder(self, training_id: str, config: dict):
    """Train SAE with progress updates via Redis pub/sub"""

    try:
        # Initialize training
        model = initialize_sae_model(config)
        dataset = load_dataset(config['dataset_id'])

        total_steps = config['total_steps']

        for step in range(total_steps):
            # Training step
            loss, sparsity = train_step(model, dataset)

            # Publish progress via Redis pub/sub
            if step % 10 == 0:
                redis_client.publish(
                    f'training:{training_id}',
                    json.dumps({
                        'type': 'training.progress',
                        'data': {
                            'step': step,
                            'total_steps': total_steps,
                            'progress': (step / total_steps) * 100,
                            'metrics': {
                                'loss': float(loss),
                                'sparsity': float(sparsity)
                            }
                        }
                    })
                )

            # Update Celery task state
            self.update_state(
                state='PROGRESS',
                meta={'step': step, 'loss': loss, 'sparsity': sparsity}
            )

        # Training complete
        redis_client.publish(
            f'training:{training_id}',
            json.dumps({
                'type': 'training.completed',
                'data': {'final_loss': float(loss)}
            })
        )

        return {'status': 'completed', 'final_loss': float(loss)}

    except Exception as e:
        # Error handling
        redis_client.publish(
            f'training:{training_id}',
            json.dumps({
                'type': 'training.error',
                'error': {'message': str(e)}
            })
        )
        raise
```

### WebSocket Server with Redis Pub/Sub

```python
# app/websocket.py
from fastapi import WebSocket, WebSocketDisconnect
import asyncio
import redis.asyncio as aioredis
import json

class TrainingProgressWebSocket:
    def __init__(self, redis_url: str):
        self.redis_url = redis_url

    async def handle_connection(self, websocket: WebSocket, training_id: str):
        """Handle WebSocket connection for training progress"""
        await websocket.accept()

        # Connect to Redis
        redis = await aioredis.from_url(self.redis_url, decode_responses=True)
        pubsub = redis.pubsub()

        try:
            # Subscribe to training channel
            await pubsub.subscribe(f'training:{training_id}')

            # Listen for messages
            async for message in pubsub.listen():
                if message['type'] == 'message':
                    # Forward to WebSocket client
                    await websocket.send_text(message['data'])

        except WebSocketDisconnect:
            print(f"Client disconnected from training {training_id}")
        finally:
            await pubsub.unsubscribe(f'training:{training_id}')
            await redis.close()

# In main.py
ws_handler = TrainingProgressWebSocket(redis_url=os.getenv("REDIS_URL"))

@app.websocket("/ws/training/{training_id}")
async def training_websocket(websocket: WebSocket, training_id: str):
    await ws_handler.handle_connection(websocket, training_id)
```

---

## Troubleshooting Guide

### Common Issues

#### 1. Memory Errors

**Symptom**: `OOM command not allowed when used memory > 'maxmemory'`

**Solutions**:
```bash
# Check current memory usage
redis-cli INFO memory

# Increase maxmemory in redis.conf
maxmemory 2gb

# Change eviction policy
maxmemory-policy allkeys-lru

# Clear specific cache namespace
redis-cli --scan --pattern 'cache:*' | xargs redis-cli DEL
```

#### 2. Connection Refused

**Symptom**: `Connection refused` or `Could not connect to Redis`

**Solutions**:
```bash
# Check if Redis is running
docker ps | grep redis

# Check Redis logs
docker logs mistudio-redis

# Verify network connectivity
ping redis-container-name

# Check bind address in redis.conf
bind 0.0.0.0  # Allow remote connections
```

#### 3. Slow Performance

**Symptom**: Redis operations taking > 10ms

**Solutions**:
```bash
# Check slow log
redis-cli SLOWLOG GET 10

# Monitor real-time commands
redis-cli MONITOR

# Check if persistence is blocking
redis-cli INFO persistence

# Disable AOF temporarily
redis-cli CONFIG SET appendonly no
```

#### 4. High CPU Usage

**Symptom**: Redis using > 50% CPU

**Solutions**:
```bash
# Check for KEYS command (use SCAN instead)
redis-cli SLOWLOG GET 100 | grep KEYS

# Monitor command stats
redis-cli INFO commandstats

# Check for long-running Lua scripts
```

---

## Best Practices Summary

### Do's ✅

1. **Use appropriate TTLs** for all cached data
2. **Use SCAN instead of KEYS** for production code
3. **Implement connection pooling** in your application
4. **Monitor memory usage** and set appropriate limits
5. **Enable persistence** (RDB + AOF) for job queue data
6. **Use pipelining** for bulk operations
7. **Set up health checks** for Redis container
8. **Use Redis password** even in internal networks
9. **Separate databases** for different use cases (0-15)
10. **Log slow queries** and optimize them

### Don'ts ❌

1. **Don't use KEYS** in production (use SCAN)
2. **Don't store large objects** (> 1MB) in Redis
3. **Don't use Redis as primary database** (use PostgreSQL)
4. **Don't forget to set TTLs** on cache keys
5. **Don't use blocking commands** in async code
6. **Don't ignore memory warnings**
7. **Don't run without persistence** for critical data
8. **Don't expose Redis port** to public internet
9. **Don't use default password** in production
10. **Don't mix different data patterns** in same database

---

## Related Documentation

- [MechInterp Studio Technical Specification](./miStudio_Specification.md)
- [Backend Implementation Guide](./Mock-embedded-interp-ui.tsx) (lines 4-313)
- [OpenAPI Specification](./openapi.yaml)
- [Code Quality Review](./CODE_QUALITY_REVIEW.md)

---

**Document Version**: 1.0
**Last Updated**: 2025-10-05
**Maintained By**: MechInterp Studio Team
