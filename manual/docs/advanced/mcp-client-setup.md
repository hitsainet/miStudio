---
title: Configure Codex and Claude Code
description: Connect Codex and Claude Code to the suite MCP servers.
---

# Connect miStudio MCP to Codex and Claude Code

This guide configures Codex and Claude Code to use an existing miStudio MCP server. Both clients can connect to the same server. Configuration examples use placeholders and contain no credentials.

## 1. Choose the endpoint

miStudio runs a separate MCP process, normally on port **8765**, using **Streamable HTTP** at **`/mcp`**. Use the MCP address, rather than the frontend URL or the backend REST API URL.

| Deployment | MCP URL |
| --- | --- |
| Docker Compose, on the client machine | `http://localhost:8765/mcp` |
| Docker Compose, on another machine | `http://<server-host>:8765/mcp` |
| Kubernetes ingress | `https://<your-mcp-host>/mcp` |
| Kubernetes port-forward | `http://localhost:8765/mcp` |

For a port-forward, leave this running in another terminal:

```bash
kubectl -n mistudio port-forward svc/mistudio-mcp 8765:8765
```

`localhost` means the machine or container running the client. An ingress hostname needs working DNS and an appropriate TLS certificate. Use HTTPS when traffic leaves a trusted network.

### One server or several?

The suite has **three independent MCP servers**. The configurations below use the verified public endpoints of the hosted suite. For a self-hosted installation, replace them with your own ingress hostnames. Health checks do not verify authenticated tool calls.

| Client entry | Hosted MCP endpoint | Coverage |
| --- | --- | --- |
| `mistudio` | `https://mcp-studio.hitsai.net/mcp` | miStudio plus optional miLLM and curated miDataworks proxies |
| `midataworks` | `https://mcp-dataworks.hitsai.net/mcp` | miDataworks' full native tool surface |
| `miforge` | `https://mcp-forge.hitsai.net/mcp` | miForge training, packs, checkpoints, adapters, chains, and settings |

The public HTTPS proxy forwards each hostname to its matching MCP ingress and service on port 8765. The client URL keeps the `/mcp` path and uses the default HTTPS port; do not append `:8765` to these public URLs. Each public `/health` route returned its matching service identity with HTTP 200, and `/mcp` returned HTTP 401 without a bearer token when checked on 2026-10-25 (America/New_York).

miLLM's MCP tools are hosted by miStudio; no separate miLLM MCP listener was found in its application or deployment configuration. Multiple ingress host aliases for one service do not constitute additional servers.

Use all three entries for direct suite coverage. The miStudio server can optionally proxy miLLM (`millm_*`) and miDataworks (`dataworks_*`) tools, so a separate miDataworks connection is optional if its curated proxy covers your needs. **miForge control requires its own connection**: miStudio's `MIFORGE_API_URL` only lets `gpu_preflight` report miForge's active runs.

miForge's native tools include run validation, budget estimation, run creation and cancellation. GPU starts, attachment, deletion, and settings changes pass its operator approval gates. Call `forge_howto` first; for native miDataworks call `dataworks_howto` first. See the sibling repositories' `miForge/docs/mcp-contract.md` and `miDataworks/docs/mcp-contract.md` for the full inventories.

The operator must enable the relevant categories and configure the upstream URLs in the **MCP server process**:

| Capability | Server environment |
| --- | --- |
| miStudio REST API | `MISTUDIO_API_URL` (default `http://localhost:8000`) |
| miLLM proxy | `MILLM_API_URL` plus the desired `millm_*` categories in `MCP_TOOL_CATEGORIES` |
| miDataworks proxy | `MIDATAWORKS_API_URL` plus `dataworks` in `MCP_TOOL_CATEGORIES` |

Setting these variables in the client shell does not reconfigure an already running remote server. Explicit category lists in Compose or Kubernetes override the code defaults. Consult [the server manual](./mcp-server.md) and [the generated tool contract](https://github.com/hitsainet/miStudio/blob/main/docs/mcp-contract.md) for categories and tool names.

If you use multiple miStudio deployments, register each under a distinct name, such as `mistudio-dev` and `mistudio-prod`, with its own URL and token variable.

### Configure all three servers

Obtain each server's token separately and export `MISTUDIO_MCP_TOKEN`, `MIDATAWORKS_MCP_TOKEN`, and `MIFORGE_MCP_TOKEN` in the client's launch environment. Do not assume their values are identical. All three processes read `MCP_AUTH_TOKEN`, but the Kubernetes secret sources differ:

| Namespace | Secret | Key |
| --- | --- | --- |
| `mistudio` | `mistudio-secrets` | `mcp-auth-token` |
| `midataworks` | `midataworks-secrets` | `MCP_BEARER_TOKEN` |
| `miforge` | `miforge-secrets` | `MCP_BEARER_TOKEN` |

For Codex, use these tables in place of the single-server example below:

```toml
[mcp_servers.mistudio]
url = "https://mcp-studio.hitsai.net/mcp"
bearer_token_env_var = "MISTUDIO_MCP_TOKEN"

[mcp_servers.midataworks]
url = "https://mcp-dataworks.hitsai.net/mcp"
bearer_token_env_var = "MIDATAWORKS_MCP_TOKEN"

[mcp_servers.miforge]
url = "https://mcp-forge.hitsai.net/mcp"
bearer_token_env_var = "MIFORGE_MCP_TOKEN"
```

For Claude Code, use these entries in `.mcp.json`:

```json
{
  "mcpServers": {
    "mistudio": {
      "type": "http",
      "url": "https://mcp-studio.hitsai.net/mcp",
      "headers": { "Authorization": "Bearer ${MISTUDIO_MCP_TOKEN}" }
    },
    "midataworks": {
      "type": "http",
      "url": "https://mcp-dataworks.hitsai.net/mcp",
      "headers": { "Authorization": "Bearer ${MIDATAWORKS_MCP_TOKEN}" }
    },
    "miforge": {
      "type": "http",
      "url": "https://mcp-forge.hitsai.net/mcp",
      "headers": { "Authorization": "Bearer ${MIFORGE_MCP_TOKEN}" }
    }
  }
}
```

For local deployments, all three servers default to container port 8765. Give each Compose service a different host port, or each Kubernetes port-forward a different local port, and adjust the URLs accordingly.

## 2. Obtain the bearer token

Ask the deployment operator for the existing miStudio MCP token through your normal secret-sharing channel. It must match the server's `MCP_AUTH_TOKEN`; it is separate from OpenAI and Anthropic API keys. miStudio HTTP MCP uses a static bearer token, so an OAuth login command is unnecessary.

In Bash, load it without putting the literal token in shell history:

```bash
read -rsp 'miStudio MCP token: ' MISTUDIO_MCP_TOKEN
printf '\n'
export MISTUDIO_MCP_TOKEN
```

Use this same shell to launch the client. A GUI or IDE must receive the variable through its own launch environment; an export in an unrelated terminal will not update a running application. Client tools do not automatically read the deployment's `.env` file.

An authorized Kubernetes operator can load the existing token directly:

```bash
export MISTUDIO_MCP_TOKEN="$(kubectl -n mistudio get secret mistudio-secrets \
  -o jsonpath='{.data.mcp-auth-token}' | base64 -d)"
```

Do not generate or rotate a server token just to add another client. Both clients may use the existing token. Keep literal tokens out of shared configuration and version control.

## 3. Configure Codex

Merge this table into **`~/.codex/config.toml`** for your user, or **`.codex/config.toml`** for the project. Replace the example URL with the endpoint selected above. Project configuration is subject to Codex workspace trust.

```toml
[mcp_servers.mistudio]
url = "https://mcp-studio.hitsai.net/mcp"
bearer_token_env_var = "MISTUDIO_MCP_TOKEN"
```

`bearer_token_env_var` takes the **variable name**, without `$` or `${…}`. Codex reads the token from its environment and constructs the Authorization header. Merge with existing settings; do not overwrite other server entries or define the same TOML table twice.

Start a new Codex session with the variable exported. Check registration:

```bash
codex mcp list
```

Then run `/mcp` inside Codex to inspect the active connection. A saved configuration alone does not prove tools can reach the backend. These settings follow [OpenAI's MCP configuration documentation](https://developers.openai.com/codex/mcp).

## 4. Configure Claude Code

### Project configuration with an environment variable

Merge this entry into **`.mcp.json`** at the root of the project where you run Claude Code:

```json
{
  "mcpServers": {
    "mistudio": {
      "type": "http",
      "url": "https://mcp-studio.hitsai.net/mcp",
      "headers": {
        "Authorization": "Bearer ${MISTUDIO_MCP_TOKEN}"
      }
    }
  }
}
```

Replace the URL, preserve existing server entries, and launch `claude` from the shell containing the exported token. Claude Code expands the variable in the header. Approve the project server when prompted. This file can be shared with the project because it contains a variable reference instead of a literal credential.

### CLI alternative

For a private registration available in every project:

```bash
claude mcp add --scope user --transport http mistudio https://mcp-studio.hitsai.net/mcp \
  --header "Authorization: Bearer $MISTUDIO_MCP_TOKEN"
```

This command expands the token immediately and saves the header value in user configuration (`~/.claude.json`); it is not a live environment-variable reference. Treat that file as a credential file. Prefer the project JSON example if you want the token resolved at launch. Without `--scope user`, the CLI defaults to private local configuration for the current project; `--scope project` writes shared `.mcp.json` configuration.

Check registration and connection:

```bash
claude mcp list
claude mcp get mistudio
```

Run `/mcp` inside Claude Code to inspect the connection and available tools. See [Claude Code's MCP documentation](https://code.claude.com/docs/en/mcp) for scopes, headers, and environment expansion.

## 5. Verify with read-only tool calls

First check the HTTP process, using the same host and scheme as your MCP URL:

```bash
curl --fail --silent --show-error https://mcp-studio.hitsai.net/health
```

`/health` is unauthenticated and reports registered categories and product status. A successful health check does **not** verify the bearer token or every backend tool.

In either client, ask:

> Use the mistudio MCP server. Call mistudio_howto first, then get_environment_settings and list_datasets. Report whether the connection works and any backend errors. Do not start jobs or change data.

Successful discovery and read calls verify more than a health response. See [MCP Agent Instructions](./mcp-agent-instructions.md) for operating workflows. For ongoing use, add an instruction to the project's `AGENTS.md` (Codex) or `CLAUDE.md` (Claude Code) to call `mistudio_howto` before miStudio work.

Client tool permission prompts and miStudio operator approvals are separate. Some GPU operations return an approval request; follow the tool's instructions and inspect `get_approval_status` before retrying. Adding a client does not bypass server approvals.

## Troubleshooting

| Symptom | Check |
| --- | --- |
| Connection refused or timeout | Server enabled and running; port-forward still open; hostname, port, firewall, and client network reachability. |
| HTML response or 404 | Use the MCP host and `/mcp`, rather than the frontend or REST API route. |
| 401 Unauthorized | Client process has a nonempty token matching server `MCP_AUTH_TOKEN`; restart the client after updating its environment. |
| `/health` works but connection fails | Health is unauthenticated; check the bearer header and `/mcp` URL. |
| Tools connect but report backend unavailable | Check server-side `MISTUDIO_API_URL` and upstream reachability from the MCP process. |
| Missing `millm_*` or `dataworks_*` tools | Check server-side categories and upstream URL variables, plus `/health` product status. |
| Missing newly added categories | Inspect `/health` for `unknown_categories`; verify deployed image and category settings agree. |
| Claude uses an unexpected endpoint | Inspect duplicate `mistudio` definitions across scopes with `claude mcp get mistudio` and `/mcp`. |
| HTTP server refuses anonymous mode | Supply `MCP_AUTH_TOKEN` and unset `MCP_ALLOW_ANONYMOUS`; anonymous mode is not supported over HTTP. |

For a new deployment, enable the Compose `mcp` profile and set its `MCP_AUTH_TOKEN`, or configure the Kubernetes `mistudio-mcp` deployment and its `mistudio-secrets` / `mcp-auth-token` key. Follow [the server manual](./mcp-server.md) for deployment steps. This guide configures clients; it does not start or change a deployment.

## Local development over stdio

HTTP is the usual shared deployment setup. For local development, clients can instead launch `python -m src.mcp_server --stdio`. Set the working directory to miStudio's `backend` directory, use an absolute Python executable from an environment with the backend dependencies installed, and provide `MISTUDIO_API_URL` for the running backend. This creates a separate MCP process per client, rather than connecting to the shared HTTP service. See the server manual for local development details.

