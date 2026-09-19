---
sidebar_position: 101
title: "FAQ"
description: "Frequently asked questions"
---

# Frequently Asked Questions

### Which models can I use?

Any HuggingFace transformer. miStudio has no architecture whitelist — layer discovery introspects each model dynamically, so standard architectures (GPT-2, Llama, Gemma, Qwen, Phi…) and unusual ones (hybrid models like LFM2) all work. Models that ship custom code need the **trust_remote_code** checkbox at download time.

### How much VRAM do I need?

It depends on model size and SAE width, not on miStudio itself. Practical reference points:

- **Getting started** (TinyStories + a ~1B model, 8× SAE): comfortable on 8 GB
- **Serious work** (2–7B models, 8–32× SAEs): 24 GB (e.g., RTX 3090/4090) is the sweet spot
- **Bigger models**: quantize to Q8/Q4 for extraction, and train SAEs from [cached activations](/advanced/multi-dataset) so the base model doesn't occupy VRAM during training

See the [OOM sizing rules](/troubleshooting#out-of-memory-oom-sizing) for the arithmetic.

### Do I need an OpenAI account?

No. Labeling works with any **OpenAI-compatible endpoint** — vLLM, Ollama, LM Studio, or another local server. OpenAI's API is supported as an alternative; pick per-pipeline in [Settings](/advanced/settings-reference). Everything else in miStudio (training, extraction, steering) runs fully locally.

### Can I use SAEs I trained elsewhere? Can I take miStudio SAEs elsewhere?

Both directions work through the **SAELens community standard** format (`cfg.json` + `sae_weights.safetensors`):

- **Inbound:** download pre-trained SAEs (Gemma Scope and others) from HuggingFace, or import from disk — see [SAE Management](/advanced/external-saes)
- **Outbound:** every Neuronpedia export includes a `saelens/` folder that loads directly in SAELens/TransformerLens; trained SAEs can also be uploaded to HuggingFace

### Where does my data live?

Everything is local: model weights, datasets, activations, and SAE weights on the filesystem (`/data/` by default); metadata, features, and labels in PostgreSQL. Nothing leaves your machine unless you explicitly push to Neuronpedia, upload to HuggingFace, or use the OpenAI labeling backend.

### What's the difference between "activation extraction" and "feature extraction"?

Two different pipeline stages that unfortunately share a word:

1. **Activation extraction** (Models panel) runs the *base model* over a dataset and caches its raw internal activations
2. **Feature extraction** (SAEs panel) runs a *trained SAE* over activations to find each feature's top activating examples

See the [extraction pipeline concept page](/concepts/extraction-pipeline) for the full picture.

### Why are my job progress bars not moving?

Progress streams over WebSocket with polling fallback — see [Real-time Updates & Progress](/troubleshooting#real-time-updates--progress). Nine times out of ten a page refresh resolves it; behind a reverse proxy, check WebSocket upgrade forwarding.

### How do I reset everything?

Stop the stack, drop the PostgreSQL database, and clear the data directory (`/data/`). On next startup the backend recreates the schema via migrations. To reset only the Settings PIN, use the `MISTUDIO_BYPASS_PIN=true` recovery flow described in the [Settings Reference](/advanced/settings-reference#pin-protection).

### Is there an API I can script against?

Yes — the full REST API (~100 endpoints) plus WebSocket channels are documented in the [Reference section](/reference/api/overview). The interactive Swagger UI is also available at `/docs` on the backend port.
