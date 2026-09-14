---
sidebar_position: 2
title: "Dataset Management"
description: "Downloading, tokenizing, and inspecting training data"
---

# Dataset Management

Everything starts with data. The **Datasets** panel is where you download text corpora from HuggingFace, tokenize them for specific models, and inspect what's actually inside them before committing GPU-hours to training.

## Downloading a Dataset

Click **+ Download** and enter a HuggingFace dataset repo ID — for example:

- `roneneldan/TinyStories` — small, fast, great for first experiments
- `monology/pile-uncopyrighted` — diverse general text
- `HuggingFaceFW/fineweb` — large-scale web text (pick a subset!)

You can select a **split** and **subset/configuration** where the repository offers them. Download progress streams to the dataset card in real time over WebSocket, with automatic fallback to polling if the connection drops.

**Cancel** is available while a download is in flight. Cancelling during the download phase removes the partial files; cancelling during post-download processing keeps the raw files so you can retry without re-downloading.

## Tokenization: One Dataset, Many Tokenizations

Models don't read text — they read integer **tokens**, and every model family has its own tokenizer. miStudio therefore stores tokenizations **per model**: a single downloaded dataset can carry multiple tokenizations, one for each `(model, max_length)` combination you need.

This matters in practice: if you download TinyStories once, you can tokenize it for `gemma-2-2b` at 1024 tokens *and* for `TinyLlama` at 512 tokens, and both training pipelines use the same raw data on disk.

### Tokenization parameters

| Parameter | Default | What it does |
|-----------|---------|--------------|
| **Model (tokenizer)** | — | Which model's tokenizer to use. Must be a downloaded model. |
| **Max Length** | 1024 | Context window per sequence. Longer = more context per sample, more VRAM during extraction/training. |
| **Token filtering** | off | Optional junk filtering (see below) |

Each tokenization is tracked as its own job with live progress, and can be cancelled or deleted independently. The unique key is `(dataset, model, max_length)` — requesting the same combination again reuses the existing tokenization instead of duplicating work.

### Token filtering

Web-scraped corpora contain junk — boilerplate, encoding artifacts, repeated punctuation. The dataset-level filter settings let you:

- Enable/disable filtering per dataset
- Choose a filter mode and a **junk-ratio threshold** (sequences above the threshold are dropped)
- Optionally strip all punctuation or a custom character set per tokenization

Use the **Tokenize Preview** action to see exactly how a sample will be tokenized and filtered *before* running the full job.

:::tip Why filter?
SAE training amplifies whatever is common in the data. If 5% of your sequences are HTML boilerplate, you will spend SAE capacity on features that detect HTML boilerplate. Filtering junk up front yields more of your feature budget for interesting semantics.
:::

## Inspecting a Dataset

Open a dataset card to view:

- **Samples browser** — paginated raw samples, so you can verify what the text actually looks like
- **Tokenization list** — every tokenization with its model, max length, token count, vocab size, and average sequence length
- **Statistics** — total tokens, sequence counts, and size on disk

:::info Bytes in exotic datasets
Some HuggingFace datasets (e.g., The Pile) embed raw bytes in metadata fields. miStudio sanitizes these automatically when displaying samples — you'll see decoded strings, not errors.
:::

## Managing Datasets

- **Delete** removes the dataset and cascades to all of its tokenizations.
- Deleting a single **tokenization** leaves the raw dataset (and its other tokenizations) intact.
- A dataset that failed mid-download shows its error message on the card; **retry** restarts the operation.

## How This Feeds the Pipeline

Training and activation extraction jobs select a **tokenization**, not a raw dataset. When you configure a training job for model *M*, only tokenizations created with *M*'s tokenizer are offered — this is what guarantees the token IDs flowing into the model are valid.

Next step: [Model Management](/core-workflow/data-model-management) — downloading and preparing the models these tokenizations reference.
