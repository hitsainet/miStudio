---
sidebar_position: 10
title: "Templates & Settings API"
description: "The four template routers and application settings"
---

# Templates & Settings API

## Template routers

Four template types share one CRUD shape. UI: [Templates & Presets](/advanced/templates).

| Prefix | Template type |
|--------|--------------|
| `/api/v1/training-templates` | SAE training configurations |
| `/api/v1/extraction-templates` | Feature-extraction configurations |
| `/api/v1/labeling-prompt-templates` | Labeling prompt/system templates |
| `/api/v1/prompt-templates` | Steering prompt sets (multi-prompt) |

Common endpoints (all four routers):

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `` | Create (201) |
| `GET` | `` | List (search + pagination) |
| `GET` | `/{id}` | Get one |
| `PATCH` | `/{id}` | Update |
| `DELETE` | `/{id}` | Delete |
| `POST` | `/export` | Export selected templates as JSON |
| `POST` | `/import` | Import templates from JSON |

Router-specific additions:

- **training-, extraction-, prompt-templates:** `GET /favorites`, `POST /{id}/favorite`
- **prompt-templates:** `POST /{id}/duplicate`
- **labeling-prompt-templates:** `GET /default`, `POST /{id}/set-default`, `POST /{id}/clone`, `GET /{id}/usage-count`

## Settings — prefix `/api/v1/settings`

DB-backed application settings. UI: [Settings Reference](/advanced/settings-reference).

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `` | List settings (sensitive values returned masked) |
| `GET` | `/{key}` | Get one setting |
| `PUT` | `` | Upsert a setting — body: `{key, value, is_sensitive, category}`. Sensitive values are AES-256-GCM encrypted at rest |
| `PUT` | `/bulk` | Upsert several settings at once |
| `DELETE` | `/{key}` | Remove a setting (204) |

### PIN protection

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/pin/status` | `{configured, bypass_active}` — whether a PIN is set and whether `MISTUDIO_BYPASS_PIN` is active |
| `POST` | `/pin/verify` | Verify an entered PIN (PBKDF2-SHA256) |
| `POST` | `/pin/set` | Set or reset the PIN |
