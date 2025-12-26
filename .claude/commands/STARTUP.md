# MechInterp Studio - Startup Guide (project)

Use this guide to start, stop, and troubleshoot MechInterp Studio services.

## Quick Start

### First Time Setup

1. **Add domain to /etc/hosts** (one-time setup):
```bash
sudo bash -c 'echo "127.0.0.1  mistudio.mcslab.io" >> /etc/hosts'
```

2. **Start all services**:
```bash
cd /home/x-sean/app/miStudio
./start-mistudio.sh
```

3. **Access the application**:
- Main URL: http://mistudio.mcslab.io

### Subsequent Starts

```bash
cd /home/x-sean/app/miStudio
./start-mistudio.sh
```

### Stop Services

```bash
./stop-mistudio.sh
```

### Manual Shutdown (if script fails)

If `stop-mistudio.sh` doesn't fully stop all services, use these commands in order:

**1. Stop Native Processes First (reverse startup order):**
```bash
# Stop Frontend (Vite)
pkill -f "vite" 2>/dev/null || true

# Stop Backend (FastAPI/uvicorn)
pkill -f "uvicorn src.main:app" 2>/dev/null || true

# Stop Celery Beat
pkill -f "celery.*beat.*src.core.celery_app" 2>/dev/null || true

# Stop Celery Worker
pkill -f "celery.*worker.*src.core.celery_app" 2>/dev/null || true
```

**2. Stop Docker Containers:**
```bash
# Stop Ollama (standalone docker run)
docker stop mistudio-ollama 2>/dev/null || true

# Stop docker-compose services
cd /home/x-sean/app/miStudio
docker-compose -f docker-compose.dev.yml stop
```

**3. Verify All Stopped:**
```bash
# Check for remaining processes
lsof -i :8000  # Backend should be empty
lsof -i :3000  # Frontend should be empty
pgrep -f celery  # Should return nothing
docker ps | grep mistudio  # Should show no running containers
```

### Emergency Shutdown (Zombie Processes)

If processes become unresponsive or zombie:

```bash
# Force kill all Python processes (CAUTION: kills ALL Python)
pkill -9 -f python 2>/dev/null || true

# Force kill specific ports
fuser -k 8000/tcp 2>/dev/null || true
fuser -k 3000/tcp 2>/dev/null || true

# Check for zombie processes
ps aux | grep defunct

# Clear GPU memory (if stuck)
# Note: May require reboot if GPU processes are zombies
nvidia-smi --gpu-reset  # Only works if no active clients
```

**If GPU is held by zombie process:** A system reboot is required. Zombie processes cannot be killed - they must be reaped by their parent, and if the parent is PID 1, only a reboot will clear them.

## Service Architecture

MechInterp Studio uses a **hybrid architecture** with both Docker-managed containers and native processes:

### Docker-Managed Services (via docker-compose)
| Service | Container Name | Port | Description |
|---------|---------------|------|-------------|
| PostgreSQL | mistudio-postgres | 5432 | Database |
| Redis | mistudio-redis | 6379 | Message broker for Celery |
| Nginx | mistudio-nginx | 80 | Reverse proxy |

### Docker-Managed Services (via docker run)
| Service | Container Name | Port | Description |
|---------|---------------|------|-------------|
| Ollama | mistudio-ollama | 11434 | LLM inference with GPU |

### Native Processes (host system)
| Service | Process | Port | Log File |
|---------|---------|------|----------|
| Backend | uvicorn | 8000 | `/tmp/backend.log` |
| Frontend | vite | 3000 | `/tmp/frontend.log` |
| Celery Worker | celery worker | - | `/tmp/celery-worker.log` |
| Celery Beat | celery beat | - | `/tmp/celery-beat.log` |

## What Gets Started

The startup script automatically starts all required services in the correct order:

### 1. Docker Services
- **PostgreSQL** (port 5432) - Database (via docker-compose)
- **Redis** (port 6379) - Message broker for Celery (via docker-compose)
- **Nginx** (port 80) - Reverse proxy (via docker-compose)
- **Ollama** (port 11434) - LLM inference server with GPU support (via docker run)

### 2. Celery Worker
- Background task processor
- Handles model downloads, tokenization, training jobs
- Log: `/tmp/celery-worker.log`

### 2b. Celery Beat
- Periodic task scheduler
- Triggers system monitoring every 2 seconds
- Triggers cleanup tasks every 10 minutes
- Log: `/tmp/celery-beat.log`

### 3. Backend (FastAPI)
- Python API server on port 8000
- Serves REST API at `/api/`
- WebSocket endpoint at `/ws/`
- Log: `/tmp/backend.log`

### 4. Frontend (Vite)
- React development server on port 3000
- Hot module reloading enabled
- Log: `/tmp/frontend.log`

## Access URLs

| Service | URL | Description |
|---------|-----|-------------|
| **Main App** | http://mistudio.mcslab.io | Primary access point (via nginx) |
| Frontend Direct | http://localhost:3000 | Direct Vite dev server |
| Backend Direct | http://localhost:8000 | Direct FastAPI server |
| API Docs | http://localhost:8000/docs | Swagger/OpenAPI docs |
| PostgreSQL | localhost:5432 | Database (user: postgres, db: mistudio) |
| Redis | localhost:6379 | Message broker |
| Ollama API | http://localhost:11434 | LLM inference (direct) |
| Ollama via Nginx | http://mistudio.mcslab.io/ollama/v1 | LLM inference (proxied) |

## Checking Service Status

### All Services at Once
```bash
./start-mistudio.sh
# The script will show a status check at the end
```

### Individual Service Checks

**Docker services:**
```bash
docker ps
# Should show: mistudio-postgres, mistudio-redis, mistudio-nginx, mistudio-ollama
```

**Backend:**
```bash
lsof -i :8000
curl http://localhost:8000/api/v1/datasets
```

**Frontend:**
```bash
lsof -i :3000
curl http://localhost:3000
```

**Celery Worker:**
```bash
pgrep -f "celery.*worker.*src.core.celery_app"
```

**Celery Beat:**
```bash
pgrep -f "celery.*beat.*src.core.celery_app"
```

**Nginx:**
```bash
docker exec mistudio-nginx nginx -t
```

**Ollama:**
```bash
docker exec mistudio-ollama ollama list
curl http://localhost:11434/api/tags
curl http://mistudio.mcslab.io/ollama/v1/models
```

## Viewing Logs

```bash
# Backend logs
tail -f /tmp/backend.log

# Frontend logs
tail -f /tmp/frontend.log

# Celery Worker logs
tail -f /tmp/celery-worker.log

# Celery Beat logs
tail -f /tmp/celery-beat.log

# Docker service logs
docker logs -f mistudio-postgres
docker logs -f mistudio-redis
docker logs -f mistudio-nginx
docker logs -f mistudio-ollama
```

## Troubleshooting

### Port Already in Use

**Port 8000 (Backend):**
```bash
lsof -i :8000
pkill -f "uvicorn src.main:app"
```

**Port 3000 (Frontend):**
```bash
lsof -i :3000
pkill -f "vite"
```

**Port 80 (Nginx):**
```bash
docker restart mistudio-nginx
```

**Port 11434 (Ollama):**
```bash
docker restart mistudio-ollama
```

### Domain Not Resolving

Check /etc/hosts:
```bash
grep mistudio /etc/hosts
```

Should show:
```
127.0.0.1  mistudio.mcslab.io
```

If missing, add it:
```bash
sudo bash -c 'echo "127.0.0.1  mistudio.mcslab.io" >> /etc/hosts'
```

### Services Won't Start

**Docker services:**
```bash
cd /home/x-sean/app/miStudio
docker-compose -f docker-compose.dev.yml up -d
docker ps  # Check status
```

**Backend (check environment):**
```bash
cd /home/x-sean/app/miStudio/backend
source venv/bin/activate
cat .env  # Verify configuration
uvicorn src.main:app --host 0.0.0.0 --port 8000
```

**Frontend (check dependencies):**
```bash
cd /home/x-sean/app/miStudio/frontend
npm install  # Reinstall if needed
npm run dev
```

### Database Connection Issues

Check PostgreSQL is running and accessible:
```bash
docker exec mistudio-postgres psql -U postgres -d mistudio -c "SELECT 1;"
```

### Redis Connection Issues

Check Redis is running:
```bash
docker exec mistudio-redis redis-cli ping
# Should return: PONG
```

### Ollama Connection Issues

Check Ollama is running and models are available:
```bash
# Check container status
docker ps | grep ollama

# Check Ollama API
curl http://localhost:11434/api/tags

# Check models list
docker exec mistudio-ollama ollama list

# Check via Nginx proxy
curl http://mistudio.mcslab.io/ollama/v1/models

# Pull a model if none available
docker exec mistudio-ollama ollama pull gemma2:2b
```

### Ollama GPU Support

Ollama is configured with GPU acceleration for faster inference:

**Check GPU is accessible:**
```bash
# Check nvidia-smi inside container
docker exec mistudio-ollama nvidia-smi

# Should show RTX 3090 with VRAM usage
```

**Check model is loaded in VRAM:**
```bash
# Run inference to load model
docker exec mistudio-ollama ollama run gemma2:2b "test"

# Check GPU memory usage (should show ~2.8GB for gemma2:2b)
nvidia-smi --query-gpu=memory.used --format=csv,noheader
```

**Technical Details:**
- Ollama runs via `docker run --gpus all` (not docker-compose due to ContainerConfig bug)
- GPU support requires nvidia-container-toolkit installed
- Models automatically load into VRAM on first inference
- Performance: ~50x faster than CPU-only mode
- gemma2:2b uses ~2.8GB VRAM with Q4_0 quantization

## Complete Reset

If you need to completely reset everything:

```bash
# Stop all services
./stop-mistudio.sh

# === NATIVE PROCESSES ===
# Kill any remaining native processes
pkill -f "uvicorn src.main:app" 2>/dev/null || true
pkill -f "vite" 2>/dev/null || true
pkill -f "celery.*src.core.celery_app" 2>/dev/null || true

# === DOCKER CONTAINERS ===
# Remove docker-compose containers and volumes (PostgreSQL, Redis, Nginx)
cd /home/x-sean/app/miStudio
docker-compose -f docker-compose.dev.yml down -v

# Remove Ollama container and volumes (standalone docker run)
docker stop mistudio-ollama 2>/dev/null || true
docker rm mistudio-ollama 2>/dev/null || true
docker volume rm ollama_data 2>/dev/null || true  # WARNING: Deletes downloaded models!

# === VERIFY CLEAN STATE ===
# Check nothing is running
lsof -i :8000 2>/dev/null || echo "Port 8000: Clear"
lsof -i :3000 2>/dev/null || echo "Port 3000: Clear"
pgrep -f celery || echo "Celery: Clear"
docker ps | grep mistudio || echo "Docker: Clear"
nvidia-smi --query-compute-apps=pid,name --format=csv,noheader || echo "GPU: Clear"

# Restart everything
./start-mistudio.sh
```

**Note:** Removing `ollama_data` volume will delete all downloaded models (~1.6GB for gemma2:2b). They will need to be re-downloaded on next start.

### Reset Database Only

To reset just the database without affecting other services:

```bash
# Stop backend and celery (they connect to DB)
pkill -f "uvicorn src.main:app" 2>/dev/null || true
pkill -f "celery.*src.core.celery_app" 2>/dev/null || true

# Reset PostgreSQL
docker-compose -f docker-compose.dev.yml down postgres
docker volume rm mistudio_postgres_data 2>/dev/null || true
docker-compose -f docker-compose.dev.yml up -d postgres

# Run migrations
cd /home/x-sean/app/miStudio/backend
source venv/bin/activate
alembic upgrade head

# Restart backend and celery
./start-mistudio.sh  # or manually start backend/celery
```

## Development Workflow

### Normal Development Session

1. Start services: `./start-mistudio.sh`
2. Access app at http://mistudio.mcslab.io
3. Make code changes (backend/frontend auto-reload)
4. View logs: `tail -f /tmp/backend.log` or `/tmp/frontend.log`
5. Stop services when done: `./stop-mistudio.sh`

### Backend Changes

Backend has auto-reload enabled via `--reload` flag:
- Edit Python files in `backend/src/`
- Save file → uvicorn auto-reloads
- Check `/tmp/backend.log` for any errors

### Frontend Changes

Frontend has HMR (Hot Module Replacement):
- Edit React/TypeScript files in `frontend/src/`
- Save file → Vite auto-updates browser
- Check browser console or `/tmp/frontend.log` for errors

### Database Migrations

When you change models:
```bash
cd /home/x-sean/app/miStudio/backend
source venv/bin/activate
alembic revision --autogenerate -m "description"
alembic upgrade head
```

## Architecture Overview

```
┌──────────────────────────────────────────────────────────┐
│  http://mistudio.mcslab.io (Port 80)                    │
│                    ↓                                     │
│  ┌──────────────────────────────────────────────────┐   │
│  │  Nginx (Docker)                                  │   │
│  │  - Proxies / → Frontend (port 3000)             │   │
│  │  - Proxies /api/ → Backend (port 8000)          │   │
│  │  - Proxies /ws/ → Backend WebSocket             │   │
│  │  - Proxies /ollama/ → Ollama (port 11434)       │   │
│  └──────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────┘
           ↓                          ↓               ↓
  ┌────────────────┐        ┌─────────────────┐  ┌──────────┐
  │ Frontend       │        │ Backend         │  │ Ollama   │
  │ Vite Dev       │        │ FastAPI         │  │ (Docker) │
  │ Port 3000      │        │ Port 8000       │  │ Port     │
  │ (React + TS)   │        │ (Python)        │  │ 11434    │
  └────────────────┘        └─────────────────┘  └──────────┘
                                     ↓
                    ┌────────────────┼────────────────┐
                    ↓                ↓                ↓
            ┌─────────────┐  ┌──────────┐  ┌──────────────┐
            │ PostgreSQL  │  │  Redis   │  │ Celery Worker│
            │ (Docker)    │  │ (Docker) │  │ + Beat       │
            │ Port 5432   │  │ Port 6379│  │ (Background) │
            └─────────────┘  └──────────┘  └──────────────┘
```

## Environment Variables

Backend configuration is in `backend/.env`:
```bash
# Database
DATABASE_URL=postgresql+asyncpg://postgres:devpassword@localhost:5432/mistudio

# Redis
REDIS_URL=redis://localhost:6379/0

# Celery
CELERY_BROKER_URL=redis://localhost:6379/0
CELERY_RESULT_BACKEND=redis://localhost:6379/1

# HuggingFace (optional)
HF_TOKEN=your_token_here
```

## Next Steps

- Access the app at http://mistudio.mcslab.io
- Check out API docs at http://localhost:8000/docs
- View logs in `/tmp/` for debugging
- Read CLAUDE.md for development workflow
