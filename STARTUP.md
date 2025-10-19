# MechInterp Studio - Startup Guide

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

## What Gets Started

The startup script automatically starts all required services in the correct order:

### 1. Docker Services (via docker-compose.dev.yml)
- **PostgreSQL** (port 5432) - Database
- **Redis** (port 6379) - Message broker for Celery
- **Nginx** (port 80) - Reverse proxy

### 2. Celery Worker
- Background task processor
- Handles model downloads, tokenization, training jobs
- Log: `/tmp/celery-worker.log`

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
# Should show: mistudio-postgres, mistudio-redis, mistudio-nginx
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

**Celery:**
```bash
pgrep -f "celery.*src.core.celery_app"
```

**Nginx:**
```bash
docker exec mistudio-nginx nginx -t
```

## Viewing Logs

```bash
# Backend logs
tail -f /tmp/backend.log

# Frontend logs
tail -f /tmp/frontend.log

# Celery logs
tail -f /tmp/celery-worker.log

# Docker service logs
docker logs -f mistudio-postgres
docker logs -f mistudio-redis
docker logs -f mistudio-nginx
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

## Complete Reset

If you need to completely reset everything:

```bash
# Stop all services
./stop-mistudio.sh

# Remove Docker containers and volumes
cd /home/x-sean/app/miStudio
docker-compose -f docker-compose.dev.yml down -v

# Kill any remaining processes
pkill -f uvicorn
pkill -f vite
pkill -f celery

# Restart everything
./start-mistudio.sh
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
┌─────────────────────────────────────────────────────┐
│  http://mistudio.mcslab.io (Port 80)               │
│                    ↓                                │
│  ┌──────────────────────────────────────────────┐  │
│  │  Nginx (Docker)                              │  │
│  │  - Proxies / → Frontend (port 3000)         │  │
│  │  - Proxies /api/ → Backend (port 8000)      │  │
│  │  - Proxies /ws/ → Backend WebSocket         │  │
│  └──────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────┘
           ↓                          ↓
  ┌────────────────┐        ┌─────────────────┐
  │ Frontend       │        │ Backend         │
  │ Vite Dev       │        │ FastAPI         │
  │ Port 3000      │        │ Port 8000       │
  │ (React + TS)   │        │ (Python)        │
  └────────────────┘        └─────────────────┘
                                     ↓
                    ┌────────────────┼────────────────┐
                    ↓                ↓                ↓
            ┌─────────────┐  ┌──────────┐  ┌──────────────┐
            │ PostgreSQL  │  │  Redis   │  │ Celery Worker│
            │ (Docker)    │  │ (Docker) │  │ (Background) │
            │ Port 5432   │  │ Port 6379│  │              │
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
