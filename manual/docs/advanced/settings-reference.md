---
sidebar_position: 6
title: "Settings Reference"
description: "Endpoints, encrypted API keys, labeling defaults, and PIN protection"
---

# Settings Reference

The **Settings** panel configures the services and defaults the rest of the app depends on. Settings are stored in the database (not browser storage), so they apply to every session and survive restarts. Sensitive values are encrypted at rest.

## Endpoints Tab

Where labeling LLMs live.

| Setting | Purpose |
|---------|---------|
| **OpenAI-compatible endpoint** | Base URL of any OpenAI-compatible server (vLLM, Ollama's OpenAI shim, miLLM, LM Studio…) used for labeling |
| **OpenAI-compatible model** | Model name to request from that endpoint |
| **Ollama / LLM Service URL** | Base URL for a local LLM service. Overrides the server's environment variable. |

The **Fetch Models** button queries the configured endpoint's `/models` list, so you can pick from what the server actually serves instead of typing model names blind.

## API Keys Tab

| Key | Used for |
|-----|----------|
| **OpenAI API Key** (`sk-proj-…`) | Bulk + enhanced labeling against api.openai.com |
| **HuggingFace Token** (`hf_…`) | Gated model/dataset downloads, SAE uploads to the Hub |

Keys are encrypted at rest with **AES-256-GCM** and never displayed in full after saving — the UI shows a masked placeholder. Re-entering a key replaces the stored ciphertext.

## Labeling Tab

Defaults for the labeling pipelines (individual jobs can still override them):

| Setting | Default | Description |
|---------|---------|-------------|
| **Default Batch Size** | 10 | Features per LLM request in bulk labeling |
| **Default Max Examples per Feature** | 25 | Activation examples shown to the LLM per feature |
| **Labeling Method** (enhanced) | OpenAI-compatible | Backend for [enhanced labeling](/core-workflow/enhanced-labeling): `openai` (api.openai.com) or `openai_compatible` (your endpoint) |
| **OpenAI Model** (enhanced) | `gpt-4o-mini` | Model for enhanced labeling when method = OpenAI. Fetch Models lists your account's available models. |
| **Max Parallel Workers (Pass 1)** | 8 | Concurrency for enhanced labeling's per-example summarization pass |

![Settings — Enhanced labeling method selection](/img/miStudio_Settings_Labeling-Enhanced_Method_Dropdown.jpg)

:::tip Reasoning models
Reasoning-class OpenAI models (gpt-5, o1/o3/o4 families) are auto-detected — miStudio switches to `max_completion_tokens` and strips `<think>` blocks from responses, so they work in labeling without special configuration.
:::

## Display Tab

Display preferences (theme, sidebar collapse) are controlled from the header and stored locally in your browser — they don't sync across devices.

## PIN Protection

Because Settings holds API keys and arms irreversible checkpoint deletion, **two tabs** can be
locked behind a PIN:

- Set a PIN in the **API Keys** tab. From then on, opening **API Keys** or **Storage** prompts for
  it once per session.
- The PIN is stored as a **PBKDF2-SHA256** hash — it is not recoverable, only resettable.

:::warning What the PIN is, and is not
The PIN is a **UI affordance**, not authentication.

- It gates **two tabs**, not the panel. Endpoints, Labeling and Display open without it, and the
  tab bar always renders. This page previously said *"the panel can be locked"*, which is not what
  the code does (MIS-E2E-160) — and until that was fixed the **Storage** tab, which arms
  irreversible checkpoint deletion, was the one destructive surface left ungated.
- **It does not gate the API.** Every setting reachable through the UI is reachable over HTTP
  without it. miStudio has no per-user authentication by design — see PADR IDL-47 for the posture,
  its boundary and what invalidates it.
- Treat it as "stop me clicking the wrong thing", never as access control. If you need real access
  control, the network boundary is where it lives.
:::

**Forgot your PIN?** Set the environment variable on the backend and restart:

```bash
MISTUDIO_BYPASS_PIN=true
```

The Settings panel opens without a PIN and shows a persistent warning banner. Reset your PIN in the API Keys tab, then **remove the flag and restart** — bypass mode is for recovery, not day-to-day use.
