# Future Feature: Training Recipes & Dataset Curation

**Status:** Planned
**Priority:** Medium
**Added:** 2026-01-07

## Overview

Enable users to create targeted training data mixes by concatenating multiple datasets from HuggingFace and user-uploaded sources.

## Motivation

- SAE quality depends heavily on training data diversity and relevance
- Users want to create specialized SAEs (e.g., "code + math", "safety-relevant conversations")
- Currently limited to single dataset per training job

## Proposed Design

```
Training Recipe
├── Dataset Sources (ordered list)
│   ├── HuggingFace: lmsys/lmsys-chat-1m (split: train, samples: 100k)
│   ├── HuggingFace: ola13/small-the_pile (split: train, samples: 50k)
│   └── Local Upload: my_custom_examples.jsonl (samples: 5k)
├── Mixing Strategy
│   ├── Sequential (A then B then C)
│   ├── Interleaved (round-robin)
│   └── Weighted random (e.g., 60% A, 30% B, 10% C)
├── Preprocessing
│   └── Text column mapping, filtering, dedup
└── Output: Combined tokenized dataset
```

## Key Features

1. **Multi-source datasets**
   - HuggingFace public datasets
   - HuggingFace gated datasets (with token)
   - Local file uploads (JSONL, CSV, Parquet)
   - User-curated example collections

2. **Mixing strategies**
   - Sequential: Train on A, then B, then C
   - Interleaved: Round-robin sampling
   - Weighted: Proportional sampling (e.g., 60/30/10)
   - Curriculum: Start with simple, progress to complex

3. **Sample limits per source**
   - Cap samples from each dataset
   - Useful for balancing large vs small datasets

4. **Recipe saving & sharing**
   - Save recipes as templates
   - Export/import recipe configurations
   - Community recipe sharing (future)

## Implementation Notes

- Leverage HuggingFace `datasets.concatenate_datasets()` and `datasets.interleave_datasets()`
- Store recipes in database with JSON schema for sources
- UI: Recipe builder with drag-drop ordering
- Backend: New Celery task for recipe compilation

## Related

- Current single-dataset training: `backend/src/workers/training_tasks.py`
- Dataset download: `backend/src/workers/dataset_tasks.py`
- Tokenization service: `backend/src/services/tokenization_service.py`
