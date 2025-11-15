# Installing GPU Support for Ollama

## Current Status
- Ollama is running in CPU-only mode (2 MiB VRAM used)
- GPU acceleration requires nvidia-container-toolkit installation
- All other services are running correctly

## Installation Steps

### Step 1: Install NVIDIA Container Toolkit

Run these commands (requires sudo):

```bash
# Add NVIDIA GPG key and repository
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# Update package list
sudo apt-get update

# Install nvidia-container-toolkit
sudo apt-get install -y nvidia-container-toolkit

# Configure Docker to use NVIDIA runtime
sudo nvidia-ctk runtime configure --runtime=docker

# Restart Docker daemon
sudo systemctl restart docker
```

Or simply run the pre-made script:
```bash
cd /home/x-sean/app/miStudio
./install-nvidia-container-toolkit.sh
```

### Step 2: Enable GPU in Docker Compose

After toolkit installation, uncomment lines 62-68 in `docker-compose.dev.yml`:

**Change from:**
```yaml
    # Then uncomment the following lines:
    # deploy:
    #   resources:
    #     reservations:
    #       devices:
    #         - driver: nvidia
    #           count: all
    #           capabilities: [gpu]
```

**To:**
```yaml
    # GPU support enabled
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
```

### Step 3: Restart the Application

```bash
./stop-mistudio.sh
./start-mistudio.sh
```

### Step 4: Verify GPU Access

Check that Ollama can see the GPU:

```bash
# Check nvidia-smi inside container
docker exec mistudio-ollama nvidia-smi

# Should show RTX 3090 with 24GB total memory
```

### Step 5: Test Model Loading

```bash
# Pull a model (if not already present)
docker exec mistudio-ollama ollama pull gemma2:2b

# Run inference to load model into VRAM
docker exec mistudio-ollama ollama run gemma2:2b "Hello, test"

# Check GPU memory usage (should show ~2GB for gemma2:2b)
nvidia-smi --query-gpu=memory.used --format=csv,noheader
```

## Expected Results

After GPU support is enabled:

- **Before**: 2 MiB VRAM used (CPU-only mode)
- **After**: ~2000 MiB VRAM used (GPU acceleration active)
- **Performance**: 10-50x faster inference speed
- **Model**: gemma2:2b Q4_0 quantization loads into RTX 3090

## Troubleshooting

### If GPU still not accessible after installation:

1. Verify toolkit is installed:
   ```bash
   dpkg -l | grep nvidia-container-toolkit
   ```

2. Check Docker runtime configuration:
   ```bash
   docker info | grep -i runtime
   # Should show: nvidia-container-runtime or nvidia
   ```

3. Verify Docker daemon restarted:
   ```bash
   sudo systemctl status docker
   ```

4. Check Docker can access GPU:
   ```bash
   docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi
   ```

### If container fails to start with GPU error:

1. Check error message:
   ```bash
   docker logs mistudio-ollama
   ```

2. Verify nvidia-docker-runtime is registered:
   ```bash
   cat /etc/docker/daemon.json
   ```

3. Re-run configuration:
   ```bash
   sudo nvidia-ctk runtime configure --runtime=docker
   sudo systemctl restart docker
   ```

## Architecture Notes

- Ollama container uses Docker Compose GPU device reservation
- GPU access requires nvidia-container-toolkit for device passthrough
- Model files stored in Docker volume: `ollama_data`
- CORS configured for remote browser access via Nginx proxy
- Endpoint: http://mistudio.mcslab.io/ollama/v1

## Current Configuration Files

- `docker-compose.dev.yml` - Service definitions with GPU config
- `nginx/nginx.conf` - Reverse proxy with `/ollama/` location
- `frontend/src/components/labeling/StartLabelingButton.tsx` - Default endpoint
- `backend/src/services/openai_labeling_service.py` - JSON parsing for Ollama responses
