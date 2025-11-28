# Ollama Integration Guide

## Overview
Ollama has been integrated into miStudio to provide local LLM inference for feature labeling. This document covers setup, configuration, and usage.

## Current Status
- **Ollama Service**: PID 1955052 (running as system service)
- **Available Model**: `gemma3:4b` (4.3B parameters, Q4_K_M quantization, 3.3GB)
- **API Endpoint**: `http://localhost:11434` (or `http://ollama.mcslab.io:11434`)

## Setup Options

### Option 1: Use Existing Native Ollama (Recommended - Already Running)
Your Ollama is already installed and running as a system service. This is the simplest option.

**Pros**:
- Already configured and running
- Models already downloaded
- Better GPU access
- Easier to manage with systemctl

**To use it**: No changes needed! Your backend is already configured to use `http://localhost:11434`.

**Management commands**:
```bash
# Check status
systemctl status ollama

# Stop service
sudo systemctl stop ollama

# Start service
sudo systemctl start ollama

# Restart service
sudo systemctl restart ollama

# Check logs
journalctl -u ollama -f
```

### Option 2: Use Dockerized Ollama
If you prefer containerization, follow these steps:

**Step 1: Stop Native Ollama**
```bash
sudo systemctl stop ollama
sudo systemctl disable ollama  # Prevent auto-start
```

**Step 2: Start Ollama Container**
```bash
cd /home/x-sean/app/miStudio
docker-compose -f docker-compose.dev.yml up -d ollama
```

**Note**: The container is configured with `OLLAMA_ORIGINS` to allow CORS requests from the frontend (http://mistudio.mcslab.io and http://localhost:3000).

**Step 3: Verify Container**
```bash
docker ps | grep mistudio-ollama
curl -s http://localhost:11434/api/tags
curl -s http://localhost:11434/v1/models  # OpenAI-compatible endpoint
```

**Step 4: Pull Models** (if not using shared volume)
```bash
docker exec mistudio-ollama ollama pull gemma3:4b
```

**Management commands**:
```bash
# Check logs
docker logs mistudio-ollama -f

# Restart container
docker restart mistudio-ollama

# Execute commands inside container
docker exec mistudio-ollama ollama list
docker exec mistudio-ollama ollama pull <model-name>
```

## GPU Support (Optional)

### For Native Ollama
GPU support is automatic if NVIDIA drivers are installed.

### For Dockerized Ollama
To enable GPU support, edit `docker-compose.dev.yml` and uncomment these lines:

```yaml
  ollama:
    image: ollama/ollama:latest
    container_name: mistudio-ollama
    ports:
      - "11434:11434"
    volumes:
      - ollama_data:/root/.ollama
    restart: unless-stopped
    deploy:                           # ← Uncomment these lines
      resources:                      # ← Uncomment these lines
        reservations:                 # ← Uncomment these lines
          devices:                    # ← Uncomment these lines
            - driver: nvidia          # ← Uncomment these lines
              count: all              # ← Uncomment these lines
              capabilities: [gpu]     # ← Uncomment these lines
```

Then restart: `docker-compose -f docker-compose.dev.yml up -d ollama`

**Prerequisites**:
- NVIDIA Docker Runtime installed
- NVIDIA drivers installed
- `nvidia-container-toolkit` package installed

## Model Management

### List Available Models
```bash
# Native Ollama
ollama list

# Dockerized Ollama
docker exec mistudio-ollama ollama list

# Via API
curl -s http://localhost:11434/api/tags | python3 -m json.tool
```

### Pull New Models
```bash
# Native Ollama
ollama pull llama3.2  # Example: pull Llama 3.2

# Dockerized Ollama
docker exec mistudio-ollama ollama pull llama3.2
```

### Recommended Models for Feature Labeling

| Model | Size | Memory | Speed | Quality |
|-------|------|--------|-------|---------|
| `gemma3:4b` | 3.3GB | 6GB RAM | Fast | Good |
| `llama3.2:3b` | 2GB | 4GB RAM | Very Fast | Good |
| `llama3.2:1b` | 1.3GB | 2GB RAM | Very Fast | Moderate |
| `phi3:3.8b` | 2.2GB | 4GB RAM | Fast | Good |
| `qwen2.5:3b` | 1.9GB | 4GB RAM | Very Fast | Good |

**Recommendation**: Use `gemma3:4b` (already installed) or `llama3.2:3b` for best balance of speed and quality.

## Using Ollama for Feature Labeling

### Backend Configuration

The backend is already configured to use Ollama via OpenAI-compatible API:

**In `backend/src/schemas/labeling.py`**:
```python
local_model: Optional[str] = Field(
    default="meta-llama/Llama-3.2-1B",
    description="Local model to use for labeling when labeling_method='local'"
)
```

**In `backend/src/services/openai_labeling_service.py`**:
- The service uses OpenAI-compatible client
- Ollama endpoint: `http://localhost:11434/v1` (OpenAI API compatible)
- Model name format: Just use the model name (e.g., `gemma3:4b`, `llama3.2`)

### Creating a Labeling Job

When creating a labeling job in the frontend UI (from any remote computer):

1. Navigate to **Extractions** panel
2. Find a completed extraction and click **"Label Features"**
3. **Select Labeling Method**: Choose "OpenAI-Compatible (Ollama, vLLM, etc.)"
4. **Configure Endpoint**:
   - Base URL: `http://mistudio.mcslab.io/ollama/v1` (default, proxied through Nginx)
   - Click **"Fetch Models"** to discover available models from Ollama
   - Select `gemma2:2b` from the dropdown (or any other model you've pulled)
5. **Select Prompt Template**: Choose a template or use the default
6. Click **"Start Labeling"**

**Architecture Notes**:
- All traffic goes through Nginx proxy at `http://mistudio.mcslab.io`
- `/ollama/*` routes are proxied to the Ollama container
- CORS is configured at both Nginx and Ollama levels
- Works from any remote computer accessing the application

### Testing Ollama

Test the API directly:
```bash
curl http://localhost:11434/api/generate -d '{
  "model": "gemma3:4b",
  "prompt": "Why is the sky blue?",
  "stream": false
}'
```

Test OpenAI-compatible endpoint:
```bash
curl http://localhost:11434/v1/chat/completions -H "Content-Type: application/json" -d '{
  "model": "gemma3:4b",
  "messages": [{"role": "user", "content": "Why is the sky blue?"}]
}'
```

## Token Filtering

Token filtering is now working correctly (81.2% filtering rate)! This means:
- Stopwords like "the", "and", "in" are removed
- Single characters are filtered out
- Only meaningful content words are sent to the LLM
- Results in better, more semantic labels

See `/tmp/filtering_verification_summary.md` for test results.

## Monitoring

### Check Ollama Status
```bash
# Native Ollama
systemctl status ollama

# Dockerized Ollama
docker ps | grep mistudio-ollama
docker logs mistudio-ollama --tail 50
```

### Check API Availability
```bash
curl -s http://localhost:11434/api/tags
```

### Monitor Labeling Jobs
```bash
# Check Celery logs
tail -f /tmp/celery-worker.log | grep -E "label|filter|junk"

# Check backend logs
tail -f /tmp/backend.log | grep -E "ollama|local"
```

## Troubleshooting

### Problem: Port 11434 already in use
```bash
# Find what's using the port
lsof -i :11434
pgrep -a ollama

# Stop native Ollama
sudo systemctl stop ollama

# Or kill specific process
sudo kill <PID>
```

### Problem: Model not found
```bash
# List available models
ollama list  # or docker exec mistudio-ollama ollama list

# Pull the model
ollama pull gemma3:4b  # or docker exec mistudio-ollama ollama pull gemma3:4b
```

### Problem: Slow inference
- Use a smaller model (e.g., `llama3.2:1b`)
- Enable GPU support (see GPU Support section)
- Check available memory: `free -h`
- Check GPU usage (if applicable): `nvidia-smi`

### Problem: Out of memory
- Use a smaller model
- Close other applications
- Check memory usage: `free -h`
- For Docker: Increase Docker memory limit

## Performance Tips

1. **Model Selection**: Start with `gemma3:4b` or `llama3.2:3b` for best quality/speed balance
2. **Batch Size**: Keep batch size around 10 for parallel labeling
3. **GPU**: Enable GPU support for 2-5x speedup
4. **Memory**: Ensure at least 2GB free RAM per 1B parameters (e.g., 8GB for 4B model)
5. **Disk Space**: Ensure adequate space for model storage (~2-5GB per model)

## Integration Status

✅ **Completed**:
- Docker Compose configuration with Ollama service
- Start script updated to check Ollama status
- Token filtering (81.2% rate, removes stopwords/junk)
- OpenAI-compatible API support

📋 **Documentation Created**:
- This guide (OLLAMA_INTEGRATION.md)
- Token filtering verification (/tmp/filtering_verification_summary.md)

🔄 **Current State**:
- Native Ollama running on PID 1955052
- `gemma3:4b` model available
- Ready to create labeling jobs!

## Next Steps

1. **Choose setup option** (native or Docker - recommend keeping native)
2. **Test labeling** with a small extraction job
3. **Verify descriptions** are clean (no stopwords)
4. **Adjust model** if needed (try `llama3.2:3b` for faster inference)
5. **Monitor performance** and optimize batch size

## Additional Resources

- **Ollama Documentation**: https://github.com/ollama/ollama
- **Ollama Models Library**: https://ollama.com/library
- **OpenAI API Compatibility**: https://github.com/ollama/ollama/blob/main/docs/openai.md
- **miStudio Token Filtering**: /tmp/filtering_verification_summary.md
