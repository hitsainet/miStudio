---
sidebar_position: 7
title: "MCP Server"
description: "Agentic access — let Claude Code and other MCP clients drive miStudio"
---

# MCP Server: Agentic Access

miStudio ships an optional **MCP (Model Context Protocol) server** that exposes the post-extraction workflow — feature analysis, [cross-feature clustering](/core-workflow/feature-groups), circuit discovery, steering, calibration, and label write-back — as tools for agentic AI clients like Claude Code. An agent can run the full *analyze → group → steer → relabel* loop autonomously, or the deeper *capture → discover → validate → calibrate → promote → export* circuit pipeline, with every action flowing through the same REST API and appearing in the UI like any other work.

The server also carries an optional **cross-product plane**: when pointed at a running [miLLM](https://github.com/Onegaishimas/miLLM) runtime, a family of `millm_*` tools lets the same agent move a tuned circuit or cluster into production and drive live serving. miStudio **discovers and calibrates** (it runs the model to learn); miLLM **serves** (it runs the model behind an OpenAI-compatible API). The boundary between the two is a portable document, not a code dependency.

## Enabling the Server

The MCP server is **off by default** and runs as its own container.

**Docker Compose:**

```bash
# 1. Set a token in .env (required — the port is LAN-reachable)
echo "MCP_AUTH_TOKEN=$(openssl rand -hex 32)" >> .env

# 2. Start with the mcp profile
docker compose --profile mcp up -d
```

**Kubernetes:** add `mcp-auth-token` to the `mistudio-secrets` Secret and apply the `mistudio-mcp` Deployment/Service/Ingress in `k8s/base/mcp.yaml`. The shipped Ingress (streaming-friendly nginx annotations: `proxy-buffering off`, 3600s read timeout) serves three routes:

- `http://mcp-mistudio.hitsai.local/mcp` and `http://mcp-mistudio.hitsai.net/mcp` — dedicated MCP hosts with full path space (`/health` works too). These resolve **only where you make them resolve** (hosts file or local DNS pointing at the ingress IP); the `.net` name is not published in public DNS unless you do so yourself.
- `http://k8s-mistudio.hitsai.local/mcp` — path route on the shared miStudio host, kept for compatibility.

Point a client machine at the ingress with a hosts-file entry, e.g. `192.168.244.61  mcp-mistudio.hitsai.local mcp-mistudio.hitsai.net`. No Ingress at all? `kubectl port-forward svc/mistudio-mcp 8765:8765` works too.

**Verify:** `curl http://<host>:8765/health` returns the enabled tool categories.

:::warning Network exposure
The server binds `0.0.0.0:8765` so agents on other LAN machines can connect. A bearer token is **always** required on this transport — and that is now enforced, not just stated: `MCP_ALLOW_ANONYMOUS` is honoured on **stdio only**, and the server refuses to start if it is set over HTTP.

Until this was fixed, the flag alone satisfied the startup guard on HTTP, so following the troubleshooting remedy below produced a LAN-reachable **unauthenticated** server — on the page that told you a token was always required.

If your agents run on the same host only, firewall port 8765.
:::

## Connecting a Client

**Claude Code:**

```bash
# Docker Compose (direct port):
claude mcp add --transport http mistudio http://<host>:8765/mcp \
  --header "Authorization: Bearer $MCP_AUTH_TOKEN"

# Kubernetes (via the shipped ingress, dedicated host):
claude mcp add --transport http mistudio http://mcp-mistudio.hitsai.local/mcp \
  --header "Authorization: Bearer $MCP_AUTH_TOKEN"
```

**Any MCP client:** streamable-HTTP transport, URL `http://<host>:8765/mcp`, header `Authorization: Bearer <token>`. A stdio mode exists for local development: `python -m src.mcp_server --stdio` (inside the backend container/venv).

:::tip Give your agent the operating manual
After connecting, paste the [**MCP Agent Instructions**](/advanced/mcp-agent-instructions) into the agent's context (or tell it to fetch that page). It covers the tool catalog semantics, the analyze→group→steer→relabel recipes, guardrail reactions, and the evidence-notes convention — agents perform dramatically better with it.
:::

## Configuration

| Env variable | Default | Purpose |
|--------------|---------|---------|
| `MCP_AUTH_TOKEN` | *(required)* | Bearer token; startup refused if empty |
| `MCP_TOOL_CATEGORIES` | `read,groups,steering,labeling,experiments,profiles,jobs,circuits` | Which tool categories are exposed; add `admin` to enable destructive deletes |
| `MILLM_API_URL` | *(unset)* | Base URL of a miLLM runtime. Set it to enable the four cross-product `millm_*` categories (`millm_circuits`, `millm_clusters`, `millm_runtime`, `millm_sensing`); they never appear unless this is set |
| `MCP_STEERING_MAX_CONCURRENT` | `2` | Max in-flight agent steering tasks |
| `MCP_STEERING_MAX_NEW_TOKENS` | `512` | Ceiling on generation length for agent steering |
| `MCP_STEERING_APPROVAL` | `false` | Route agent steering through operator approval (see below) |

Disabled categories simply don't appear in the agent's tool list. Disabling the whole server (omit the compose profile / scale the deployment to 0) never affects the frontend or REST API.

The `circuits` category is **on by default** as of the circuits arc. The four `millm_*` categories are the **cross-product plane** — miStudio discovers and calibrates, miLLM serves — and are gated separately: they require `MILLM_API_URL` and are never enabled by default, even if you list them in `MCP_TOOL_CATEGORIES`.

## Tool Catalog

The authoritative inventory is [`docs/mcp-contract.md`](https://github.com/Onegaishimas/miStudio/blob/main/docs/mcp-contract.md) in the repo — an **auto-generated, diff-tested** file derived from the live tool registry. The table below mirrors it; the contract wins if they ever drift.

:::tip Agents: call `mistudio_howto` first
`mistudio_howto` is the **"call me first"** tool. A flat table can't carry ordering constraints, GPU-lock contention, id namespaces, or the failure modes that mislead — that tool holds the workflows (feature analysis, the circuit-discovery evidence ladder, the cross-plane document hand-off) and the guardrail reactions. Point your agent at it before it starts circuit or steering work.
:::

| Category | Tools |
|----------|-------|
| **read** (12) | `list_extractions`, `get_extraction_summary`, `list_trainings`, `search_features`, `get_feature`, `get_feature_examples`, `get_feature_token_analysis`, `get_feature_logit_lens`, `get_feature_correlations`, `get_feature_ablation`, `get_feature_nlp_analysis`, `mistudio_howto` |
| **groups** (6) | `compute_feature_groups`, `get_grouping_status`, `get_feature_groups`, `get_feature_group_members`, `find_features_by_token`, `find_related_features` |
| **steering** (10) | `steering_status`, `get_steering_mode`, `enter_steering_mode`, `exit_steering_mode`, `steer_compare`, `steer_sweep`, `steer_combined`, `compute_cluster_allocation`, `get_steering_result`, `cancel_steering_task` |
| **circuits** (24) *(default-on)* | `start_circuit_capture`, `list_circuit_captures`, `run_circuit_discovery`, `get_discovery_results`, `run_attribution_pass`, `validate_circuit_edges`, `create_circuit`, `build_circuit_from_discovery`, `update_circuit`, `get_circuit`, `list_circuits`, `delete_circuit` ⚠️, `run_circuit_faithfulness`, `calibrate_circuit_strength`, `reproduce_calibration`, `promote_circuit`, `export_circuit_definition`, `export_circuit_slices`, `import_circuit_definition`, `record_steering_samples`, `get_steering_samples`, `list_validation_manifests`, `get_validation_manifest`, `reproduce_validation` |
| **experiments** (3) | `save_experiment`, `list_experiments`, `get_experiment` |
| **profiles** (4) | `list_cluster_profiles`, `get_cluster_profile`, `save_cluster_profile`, `export_cluster_definition` — durable cluster profiles + portable `mistudio.cluster-definition/v1` export |
| **labeling** (3) | `update_feature_label`, `run_enhanced_labeling`, `get_enhanced_label` |
| **jobs** (1) | `get_task_status` |
| **admin** (2) *(off by default)* | `delete_experiment` ⚠️, `delete_extraction` ⚠️ — destructive |

**Cross-product plane** — the four `millm_*` categories drive a live miLLM runtime and only appear when `MILLM_API_URL` is set:

| Category | Tools |
|----------|-------|
| **millm_circuits** (16) | `millm_import_circuit`, `millm_list_circuits`, `millm_export_circuit`, `millm_delete_circuit` ⚠️, `millm_activate_circuit`, `millm_deactivate_circuit`, `millm_circuit_status`, `millm_set_circuit_intensity`, `millm_circuit_claims`, `millm_release_circuit_claims`, `millm_circuit_sensing_enable`, `millm_circuit_sensing_disable`, `millm_circuit_sensing_status`, `millm_circuit_sensing_events`, `millm_circuit_sensing_event`, `millm_circuit_sensing_clear` ⚠️ |
| **millm_clusters** (6) | `millm_import_cluster`, `millm_list_clusters`, `millm_export_cluster`, `millm_activate_cluster`, `millm_deactivate_cluster`, `millm_hub_search` |
| **millm_runtime** (5) | `millm_status`, `millm_list_profiles`, `millm_activate_profile`, `millm_deactivate_profile`, `millm_set_intensity` |
| **millm_sensing** (5) | `millm_sensing_enable`, `millm_sensing_disable`, `millm_sensing_status`, `millm_sensing_events`, `millm_sensing_config` |

⚠️ = destructive or irreversible.

Long-running operations follow the platform's async pattern: the start tool returns a task id, and the agent polls `get_task_status` / `get_steering_result`. Task ids are durable database records, so they survive agent reconnects.

### Circuits: discovery, calibration, and recording

The `circuits` category exposes the full mechanistic-interpretability pipeline. Two tools from the current arc are worth calling out:

- **Calibration** (`calibrate_circuit_strength` → `reproduce_calibration`): a **two-detector usable-band search**. It finds the ONSET where steering starts to bite (output-drift, no judge needed) and the CORRECTNESS CLIFF where the model starts producing fluent-but-false output (an LLM judge scores generated neutral-topic falsifiable probes), bisects between them, and clamps the served dial to `[onset, cliff]`. It is a **badge, not a gate** — it annotates the circuit, it does not block serving. A judge too weak to grade its own probes reports `judge_unreliable` rather than a false `no_band`.
- **Steered Transcript Recorder** (`record_steering_samples` → `get_steering_samples`): an **instrument, not a judge**. It records `(dial, prompt, unsteered, steered)` transcripts for a circuit, a cluster, or an ad-hoc feature set, so a stronger model can analyze the run *afterwards*. It never scores anything itself.

Both write **manifests** — self-contained, reproducible records — and both have a `reproduce_*` tool that re-runs from the manifest and reports a delta verdict, so a claim is reproducible rather than a one-off.

## Guardrails & Provenance

- **Label provenance:** agent label edits carry `label_source: mcp_agent`, so you can always tell agent work from human work.
- **Protected labels:** aqua-starred features (completed enhanced labels) return a 409 when an agent tries to edit their name/category/description; the agent must pass an explicit `override_protected` flag. Notes remain appendable — the documented convention is `[MCP <date>] evidence: experiment <id> — <summary>`.
- **Steering limits:** concurrency cap and generation-length ceiling are enforced before the GPU is touched.
- **Operator-approval mode:** with `MCP_STEERING_APPROVAL=true`, agent steering calls become pending requests instead of running. They appear as a banner in the **Steering panel** with Approve/Deny buttons; on approval the backend submits the stored request itself, and the agent's poll picks up the resulting task id.
- **Audit:** every tool call is logged (tool name, argument digest, status, duration).

## Worked Example: the Analyze → Group → Steer → Relabel Loop

A Claude Code session pointed at the server, instructed with natural language only:

1. *"List extractions and build feature clusters for the newest one"* → `list_extractions`, `compute_feature_groups`, `get_task_status` until complete
2. *"Show me the biggest groups"* → `get_feature_groups(sort_by="size")` — say it finds a 12-member `"love"` group with cohesion 0.81
3. *"What do the members have in common?"* → `get_feature_group_members`, then `get_feature_examples` + `get_feature_logit_lens` per member → agent hypothesizes "expressions of affection"
4. *"Validate that by steering"* → `enter_steering_mode`, `steer_sweep(feature_idx=4821, strength_values=[0, 10, 30])`, `get_steering_result` → steered outputs turn affectionate; baseline doesn't
5. *"Save the evidence and update the labels"* → `save_experiment`, then `update_feature_label(name="expressions of affection", notes="[MCP 2026-07-12] evidence: experiment exp_… — sweep shows dose-dependent affection shift")`
6. Every record — the group index, the experiment, the relabeled features with `mcp_agent` provenance — is immediately visible in the miStudio UI.

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| Server exits at startup: "MCP_AUTH_TOKEN is required" | Set `MCP_AUTH_TOKEN` in `.env`. **`MCP_ALLOW_ANONYMOUS` will not help here** — it is honoured on the **stdio** transport only, and the server refuses to start over HTTP with it set. That is deliberate: the HTTP port is LAN-reachable and exposes `delete_circuit`, GPU steering and label write-back. |
| 401 from every call | Client isn't sending `Authorization: Bearer <token>`, or tokens don't match |
| Tools error "backend unreachable" | The `mcp-server` container can't reach `backend:8000` — check both are on the same network and the backend is healthy |
| `steer_*` returns a guardrail message | Concurrency cap hit — poll or cancel existing tasks, or raise `MCP_STEERING_MAX_CONCURRENT` |
| Steering tools return `pending_approval` | Approval mode is on — approve the request in the Steering panel |
