# OpenAI API Request Testing Guide

## Overview

When running labeling jobs with "Save API requests for testing" enabled, the system automatically saves every OpenAI API request to organized folders for testing and debugging. This allows you to:
- Review exact prompts being sent to the LLM
- Test requests directly in Postman
- Experiment with prompt variations using cURL
- Debug labeling issues by inspecting the full request/response cycle

## File Organization

API requests are saved in the application root under `tmp_api/` with the following structure:

```
miStudio/
└── tmp_api/
    ├── 20251117_143022_label_extr_..._20251117_103044/
    │   ├── 20251117_143022_label_extr_..._20251117_103044_neuron_0.json
    │   ├── 20251117_143022_label_extr_..._20251117_103044_neuron_0.sh
    │   ├── 20251117_143022_label_extr_..._20251117_103044_neuron_0_postman.json
    │   ├── 20251117_143022_label_extr_..._20251117_103044_neuron_1.json
    │   ├── 20251117_143022_label_extr_..._20251117_103044_neuron_1.sh
    │   ├── 20251117_143022_label_extr_..._20251117_103044_neuron_1_postman.json
    │   └── ... (all neurons for this job)
    └── 20251117_150530_label_extr_..._20251117_145000/
        └── ... (all neurons for another job)
```

**Folder naming format:** `YYYYMMDD_HHMMSS_{labeling_job_id}/`
- **ONE folder per labeling job** - All neurons for a job are saved in the same folder
- Cross-platform compatible (Windows, Mac, Linux)
- Timestamp allows chronological sorting
- Job ID identifies which labeling job the requests belong to

**File naming format:** `{folder_name}_neuron_{idx}.{ext}`
- Files are prefixed with their folder name for self-identification
- Even if moved elsewhere, files remain tied to their labeling job and SAE
- Makes it easy to identify which extraction and training run produced each file

## Generated Files

For each feature being labeled, three files are created in the job's folder:

### 1. JSON Payload (`{folder_name}_neuron_{idx}.json`)
The raw request body in JSON format. Contains:
- `model`: Model identifier (e.g., "gpt-4o-mini")
- `messages`: Array with system and user messages
- `temperature`: Sampling temperature
- `max_tokens`: Maximum response tokens
- `top_p`: Nucleus sampling parameter

**Example:**
```json
{
  "model": "gpt-4o-mini",
  "messages": [
    {
      "role": "system",
      "content": "You are an expert in mechanistic interpretability..."
    },
    {
      "role": "user",
      "content": "Analyze this feature based on token frequencies:\n'example'   → 15 times\n'police'    → 12 times\n..."
    }
  ],
  "temperature": 0.3,
  "max_tokens": 50,
  "top_p": 0.9
}
```

### 2. cURL Script (`{folder_name}_neuron_{idx}.sh`)
Executable shell script with a ready-to-run cURL command.

**Usage:**
```bash
# Run from within the job's folder
cd tmp_api/20251117_143022_label_extr_.../
bash 20251117_143022_label_extr_..._neuron_964.sh

# Or make it executable and run
chmod +x 20251117_143022_label_extr_..._neuron_964.sh
./20251117_143022_label_extr_..._neuron_964.sh
```

**Example script:**
```bash
#!/bin/bash
# OpenAI API Request - Generated 20251117_143022
# Labeling Job ID: label_extr_20251116_201719_train_36_20251117_103044
# Neuron Index: 964
# Base URL: https://api.openai.com/v1
# Model: gpt-4o-mini
# Folder: 20251117_143022_label_extr_20251116_201719_train_36_20251117_103044

curl -X POST 'https://api.openai.com/v1/chat/completions' \
  -H 'Authorization: Bearer sk-...' \
  -H 'Content-Type: application/json' \
  -d @20251117_143022_label_extr_..._neuron_964.json
```

### 3. Postman Collection (`{folder_name}_neuron_{idx}_postman.json`)
A complete Postman collection file ready to import.

**Import steps:**
1. Open Postman
2. Click "Import" button
3. Select the `*_postman.json` file
4. The request will appear in your collections
5. Click "Send" to test

## Finding Your Files

List recent labeling job folders:
```bash
ls -lt tmp_api/ | head -20
```

List files in a specific labeling job:
```bash
ls -lh tmp_api/20251117_143022_label_extr_.../
```

Find files for a specific neuron across all jobs:
```bash
ls -lh tmp_api/*/*_neuron_964*
```

View the JSON payload:
```bash
cat tmp_api/20251117_143022_label_extr_.../20251117_143022_label_extr_..._neuron_964.json | jq '.'
```

## Testing Workflow

### Option 1: Quick Test with cURL
```bash
# Navigate to the job folder and execute the script
cd tmp_api/20251117_143022_label_extr_.../
bash 20251117_143022_label_extr_..._neuron_964.sh | jq '.'
```

### Option 2: Modify and Test with cURL
```bash
# Navigate to the job folder
cd tmp_api/20251117_143022_label_extr_.../

# Edit the JSON payload
nano 20251117_143022_label_extr_..._neuron_964.json

# Run the cURL script (it references the JSON file in same folder)
bash 20251117_143022_label_extr_..._neuron_964.sh
```

### Option 3: Use Postman
1. Import the `*_postman.json` file into Postman
2. Modify the request body as needed
3. Click "Send" to test
4. View formatted response
5. Save variations as separate requests

### Option 4: Manual cURL with Inline JSON
```bash
# Copy the JSON payload
cat tmp_api/20251117_143022_label_extr_.../20251117_143022_label_extr_..._neuron_964.json

# Construct manual cURL command
curl -X POST 'https://api.openai.com/v1/chat/completions' \
  -H 'Authorization: Bearer YOUR_API_KEY' \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "gpt-4o-mini",
    "messages": [...],
    "temperature": 0.3,
    "max_tokens": 50,
    "top_p": 0.9
  }'
```

## Use Cases

### 1. Debugging Failed Labels
When a feature gets a poor label:
```bash
# Find the labeling job folder
ls -lt tmp_api/ | head -10

# Find the request for that neuron
ls tmp_api/*/*_neuron_964*

# Review the prompt
cat tmp_api/20251117_143022_label_extr_.../20251117_143022_label_extr_..._neuron_964.json | jq '.messages[1].content'

# Test with the same request
cd tmp_api/20251117_143022_label_extr_.../
bash 20251117_143022_label_extr_..._neuron_964.sh
```

### 2. Experimenting with Prompts
```bash
# Navigate to job folder
cd tmp_api/20251117_143022_label_extr_.../

# Copy the JSON file
cp 20251117_143022_label_extr_..._neuron_964.json experiment_prompt.json

# Edit the user message
nano experiment_prompt.json

# Test with modified prompt
curl -X POST 'https://api.openai.com/v1/chat/completions' \
  -H 'Authorization: Bearer sk-...' \
  -H 'Content-Type: application/json' \
  -d @experiment_prompt.json | jq '.'
```

### 3. Testing Different Models
```bash
# Navigate to job folder
cd tmp_api/20251117_143022_label_extr_.../

# Edit the model parameter
jq '.model = "gpt-4o"' 20251117_143022_label_extr_..._neuron_964.json > test_gpt4o.json

# Test with GPT-4o
curl -X POST 'https://api.openai.com/v1/chat/completions' \
  -H 'Authorization: Bearer sk-...' \
  -H 'Content-Type: application/json' \
  -d @test_gpt4o.json | jq '.'
```

### 4. Testing OpenAI-Compatible Endpoints (Ollama, vLLM)
For local endpoints without authentication:
```bash
# Navigate to job folder
cd tmp_api/20251117_143022_label_extr_.../

# Test with local endpoint
curl -X POST 'http://mistudio.mcslab.io/ollama/v1/chat/completions' \
  -H 'Content-Type: application/json' \
  -d @20251117_143022_label_extr_..._neuron_964.json | jq '.'
```

## Log Output

When labeling jobs run with "Save API requests" enabled, the backend logs show the saved file locations:

```
INFO: 📁 Created API request folder for labeling job: tmp_api/20251117_143022_label_extr_20251116_201719_train_36_20251117_103044/
INFO: 💾 Saved API request for testing:
INFO:    Folder: tmp_api/20251117_143022_label_extr_20251116_201719_train_36_20251117_103044/
INFO:    JSON Payload: /path/to/miStudio/tmp_api/20251117_143022_label_extr_.../20251117_143022_label_extr_..._neuron_964.json
INFO:    cURL Script: /path/to/miStudio/tmp_api/20251117_143022_label_extr_.../20251117_143022_label_extr_..._neuron_964.sh
INFO:    Postman Collection: /path/to/miStudio/tmp_api/20251117_143022_label_extr_.../20251117_143022_label_extr_..._neuron_964_postman.json
INFO:
INFO:    Quick test: cd tmp_api/20251117_143022_label_extr_.../ && bash 20251117_143022_label_extr_..._neuron_964.sh
INFO:    Or import into Postman
```

View logs in real-time:
```bash
tail -f /tmp/backend.log | grep "💾 Saved API request"
```

## Cleanup

Remove old request folders to save space:
```bash
# Remove folders older than 7 days
find tmp_api/ -type d -mtime +7 -exec rm -rf {} +

# Remove specific labeling job folder
rm -rf tmp_api/20251117_143022_label_extr_.../

# Remove all saved requests (careful!)
rm -rf tmp_api/
```

## Security Note

⚠️ **Important**: The saved files contain your API key (if using OpenAI).
- Files are saved to `tmp_api/` in the application root
- `tmp_api/` is automatically excluded from git via `.gitignore`
- Do not share these files publicly
- Clean up after testing (folders persist across reboots)

## Troubleshooting

### "Authorization failed" error
- Check the API key in the cURL script or Postman collection
- Verify the key hasn't expired
- For OpenAI-compatible endpoints (Ollama), remove the Authorization header

### "Model not found" error
- Check the model name is correct
- For OpenAI-compatible endpoints, ensure the model is loaded
- List available models: `curl http://mistudio.mcslab.io/ollama/v1/models`

### "Connection refused" error
- Verify the endpoint URL is accessible
- Check if the service is running: `docker ps | grep ollama`
- Test endpoint: `curl http://mistudio.mcslab.io/ollama/v1/models`

## Next Steps

Once you've tested and refined your prompts:
1. Create a new prompt template in the Labeling Prompt Templates panel
2. Copy your refined system message and user prompt template
3. Use the new template for future labeling jobs
4. Compare results between different prompt templates
