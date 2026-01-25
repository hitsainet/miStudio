# MechInterp Studio - GCP Docker Deployment

Deploy MechInterp Studio on a Google Cloud Platform instance with NVIDIA T4 GPU.

**URL**: http://gcp-dkr-mistudio.hitsai.net

## Prerequisites

1. **GCP Instance**: n1-standard-4 (or larger) with NVIDIA T4 GPU
2. **OS**: Ubuntu 22.04 LTS
3. **NVIDIA Drivers**: Installed via GCP's Deep Learning VM or manually
4. **DNS**: `gcp-dkr-mistudio.hitsai.net` pointing to the instance IP
5. **Firewall**: Port 80 open
6. **DNS**: `gcp-dkr-neuron.hitsai.net` pointing to the instance IP (for Neuronpedia)

## Quick Start

```bash
# 1. Clone the repository (or copy the docker/ directory)
git clone https://github.com/hitsainet/miStudio.git
cd miStudio/docker

# 2. Run the setup script
chmod +x setup-gcp.sh
./setup-gcp.sh
```

## Manual Deployment

If you prefer manual deployment:

```bash
# 1. Install Docker
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER

# 2. Install Docker Compose plugin
sudo apt-get update
sudo apt-get install -y docker-compose-plugin

# 3. Install NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# 4. Deploy
mkdir -p /opt/mistudio
cp docker-compose.gcp.yml /opt/mistudio/docker-compose.yml
cp nginx.gcp.conf /opt/mistudio/nginx.gcp.conf
cd /opt/mistudio
docker compose pull
docker compose up -d
```

## Services

| Service | Container | Port | Description |
|---------|-----------|------|-------------|
| Frontend | mistudio-frontend | 3000 | React UI (proxied via nginx) |
| Backend | mistudio-backend | 8000 | FastAPI + WebSocket |
| Celery Worker | mistudio-celery-worker | - | Background tasks (GPU) |
| Celery Beat | mistudio-celery-beat | - | Scheduled tasks |
| PostgreSQL | mistudio-postgres | 5432 | MiStudio database |
| PostgreSQL | neuronpedia-postgres | 5432 | Neuronpedia database |
| Redis | mistudio-redis | 6379 | Celery broker |
| Nginx | mistudio-nginx | 80 | Reverse proxy |
| Neuronpedia | neuronpedia-webapp | 3000 (internal) | Feature visualization |

## Access URLs

- **MechInterp Studio**: http://gcp-dkr-mistudio.hitsai.net
- **Neuronpedia**: http://gcp-dkr-neuron.hitsai.net
- **API Documentation**: http://gcp-dkr-mistudio.hitsai.net/api/docs

## Common Commands

```bash
# View all containers
docker compose ps

# View logs
docker compose logs -f                    # All services
docker compose logs -f backend            # Backend only
docker compose logs -f celery-worker      # Celery worker only

# Restart services
docker compose restart
docker compose restart backend            # Restart specific service

# Stop all services
docker compose down

# Update to latest images
docker compose pull
docker compose up -d

# Check GPU usage
docker exec mistudio-backend nvidia-smi
docker exec mistudio-celery-worker nvidia-smi

# Database access
docker exec -it mistudio-postgres psql -U mistudio -d mistudio
docker exec -it neuronpedia-postgres psql -U neuronpedia -d neuronpedia

# Run database migrations
docker exec mistudio-backend python -m alembic upgrade head
```

## Data Persistence

Data is stored in Docker volumes:
- `mistudio_postgres_data` - MiStudio PostgreSQL data
- `neuronpedia_postgres_data` - Neuronpedia PostgreSQL data
- `mistudio_redis_data` - Redis data
- `mistudio_data` - Model checkpoints, datasets, SAEs

To backup:
```bash
docker run --rm -v mistudio_data:/data -v $(pwd):/backup alpine tar cvf /backup/mistudio_data_backup.tar /data
```

## Troubleshooting

### GPU not available in containers
```bash
# Verify GPU is accessible
docker run --rm --gpus all nvidia/cuda:12.0-base nvidia-smi

# If not, reconfigure nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

### Services not starting
```bash
# Check logs for errors
docker compose logs backend
docker compose logs celery-worker

# Check if databases are healthy
docker compose ps
```

### WebSocket connection issues
- Ensure nginx is properly proxying WebSocket connections
- Check browser console for connection errors
- Verify port 80 is accessible

### Database connection errors
```bash
# Check PostgreSQL is running
docker compose logs postgres
docker compose logs neuronpedia-postgres

# Test connection
docker exec mistudio-postgres pg_isready -U mistudio -d mistudio
```

## Security Notes

For production deployment:
1. Change all `*_secure_pw_2026` passwords in docker-compose.yml
2. Change `SECRET_KEY` and `NEXTAUTH_SECRET` values
3. Consider adding HTTPS with Let's Encrypt
4. Restrict firewall rules to only necessary IPs
5. Enable GCP IAM authentication if needed

## GCP Firewall Rules

Create firewall rules to allow HTTP traffic:

```bash
gcloud compute firewall-rules create allow-mistudio-http \
    --allow tcp:80 \
    --target-tags=mistudio \
    --description="Allow HTTP access to MechInterp Studio"
```

Then add the `mistudio` network tag to your instance.

**DNS Configuration:**
Both domains must point to your GCP instance IP:
- `gcp-dkr-mistudio.hitsai.net` → MechInterp Studio
- `gcp-dkr-neuron.hitsai.net` → Neuronpedia
