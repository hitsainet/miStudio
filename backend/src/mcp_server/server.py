"""
miWorkbench MCP server assembly: MCPServer instance, bearer auth, category gating.

Auth model (Feature 010, BR-5.1): the streamable-HTTP transport requires
``Authorization: Bearer <MCP_AUTH_TOKEN>``, compared with
``hmac.compare_digest`` (same constant-time pattern as the backend's
X-Internal-Token check). The port binds 0.0.0.0 by default — LAN-reachable by
design (product decision); the token is therefore mandatory: startup is
refused with an empty token unless MCP_ALLOW_ANONYMOUS=true (stdio/local dev).
"""

import functools
import hashlib
import hmac
import json
import logging
import re
import time
from typing import Any

from mcp.server.mcpserver import MCPServer
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

from .client import MiWorkbenchClient
from .config import MCPSettings
from .context import ExternalContext
from .tools import CATEGORY_MODULES, EXTERNAL_PRODUCTS, all_categories
from .tools.miserver_batches import MAX_BATCH_BYTES

#: The Streamable HTTP request-body limit. mcp 1.29 added one, default 4 MiB, answered 413 before
#: any tool runs - but `millm_upload_batch_file` takes up to MAX_BATCH_BYTES (16 MiB) of rows as
#: its JSON arguments, so the SDK default silently cut the tool's documented cap to a quarter.
#: Twice the cap leaves room for the JSON-RPC envelope and for a client that \u-escapes non-ASCII,
#: and it is still a bound, which is what the SDK change is for. Pinned through the real ASGI app
#: by tests/unit/test_mcp_transport_limits.py.
MCP_MAX_REQUEST_BODY_BYTES = 2 * MAX_BATCH_BYTES

logger = logging.getLogger(__name__)

SERVER_INSTRUCTIONS = """\
miWorkbench MCP server — SAE feature analysis, circuit discovery, and steering.

**CALL `miworkbench_howto` FIRST** for circuit or steering work. Up to {tool_count} tools
across {category_count} categories is more than a description can carry; that tool holds the
workflows, the document shapes that cross between miWorkbench and miServer, and the
failure modes that look like something else.

Renamed in the 2026-11 rebrand: this server is `miworkbench`, every tool that calls miServer
starts `miserver_`, and the guide tool is `miworkbench_howto`. A tool name saved before the
rename that is no longer listed has the same arguments and answers under the new prefix.

Two planes: miWorkbench DISCOVERS and CALIBRATES (it runs the model to learn);
miServer SERVES (it runs the model to serve, behind an OpenAI-compatible API).
The boundary is a portable document, not a code dependency.

Feature analysis: list_extractions → search_features → get_feature →
get_feature_examples/token-analysis/logit-lens → update_feature_label.

Circuit discovery (each stage EARNS an evidence rung — see
`miworkbench_howto(\"discovery_pipeline\")`): start_circuit_capture →
run_circuit_discovery (rung 0) → run_attribution_pass (rung 1) →
validate_circuit_edges (rung 2, a real intervention) → create_circuit →
run_circuit_faithfulness → promote_circuit → export_circuit_definition.

Steering: steer_sweep / steer_compare / steer_combined auto-start the GPU
worker, one FRESH worker per task that exits after it, so the model is
loaded per task and released with the worker. enter_steering_mode is
OPTIONAL and loads nothing; exit_steering_mode stops an idle worker.
Poll get_steering_result: it also releases a concurrency slot.

Probe monitors: list_datasets (download_dataset brings one in) →
register_probe_dataset (train/eval/calibration
/everyday_text/nonsense_text views; create_nonsense_text_dataset makes and
registers a nonsense set) → recommend_probe_rules (which pooling rules fit the
training view, from its span column; a run omitting one is refused unless
config.acknowledge_rule_choice says why) → forecast_probe_training_run (host
memory, CPU only) → start_probe_training_run → list_probe_monitors(run_id) →
evaluate_probe_monitor (how rung 2 is reached: every out-of-distribution set
clears 0.5 AND catches ≥ half its positives at the shipped bar, plus an
everyday_text and a nonsense_text set with few false alarms) → build_probe_definition. The
two GPU starts wait for operator approval: the first call returns
{"approval_request_id", "status": "pending_approval"}; re-call with the same
arguments plus that id once approved.

Long-running operations return task ids — poll get_task_status /
get_steering_result. A task that seems never to start: read `worker_watch` in get_gpu_status.

Cross-product (when miserver_* tools are present): move a tuned artifact into
production with export_cluster_definition → miserver_import_cluster, or
export_circuit_definition → miserver_import_circuit → miserver_activate_circuit.
A circuit below rung 2 is REFUSED unless you pass acknowledge_unvalidated.
miserver_status answers "what is steering right now" in one call. When miServer is
down its tools return {"unavailable": "miserver", "reason": …} — report the
reason, don't retry in a loop. miserver_* scoring and generation tools never swap
the served model: they answer {"refused": "model_not_resident", "next":
"miserver_load_model"} instead. miServer decides its own requests (ENH-008): a
model load, unload or delete, a probe-limit change, a capacity measurement,
an import of operator code or a telemetry-settings change (miserver_set_telemetry)
answers miServer's {"approval_id", "status":
"pending", …, "decide_in": "miserver"} — the operator decides in miServer's
Approvals page and miServer runs it once; poll miserver_get_approval_status, never
re-submit.

miDataworks (when dataworks_* tools are present): find data, read a version's
handoff manifest, label, publish, send a detector set. miDataworks holds every
approval: a gated action answers {"approval_id", "status": "pending", …} —
poll dataworks_get_approval_status, never re-submit. When miDataworks is down
its tools return {"unavailable": "dataworks", "reason": …}.
"""


class BearerAuthMiddleware(BaseHTTPMiddleware):
    """Constant-time bearer-token check on every HTTP request except /health."""

    def __init__(self, app: Any, token: str):
        super().__init__(app)
        self._token = token

    async def dispatch(self, request: Request, call_next):
        if request.url.path == "/health":
            return await call_next(request)
        header = request.headers.get("authorization", "")
        provided = header[7:] if header.lower().startswith("bearer ") else ""
        # MIS-E2E-117: `hmac.compare_digest` raises TypeError when either str
        # argument contains a non-ASCII character, so a token like "tökén"
        # produced a 500 — from an unauthenticated, LAN-reachable port. A 500
        # on a bad credential also tells the caller their input reached
        # something, which a 401 does not. Compare as UTF-8 bytes: every input
        # encodes, the comparison stays constant-time, and a wrong token is a
        # 401 whatever alphabet it is written in.
        if not provided or not hmac.compare_digest(
            provided.encode("utf-8"), self._token.encode("utf-8")
        ):
            return JSONResponse({"detail": "Invalid or missing bearer token"}, status_code=401)
        return await call_next(request)


class AuditToolLogger:
    """Structured one-line audit log per tool call (BR-5.4)."""

    @staticmethod
    def log(tool: str, args: dict, status: str, duration_ms: float) -> None:
        digest = hashlib.sha256(
            json.dumps(args, sort_keys=True, default=str).encode()
        ).hexdigest()[:12]
        logger.info(
            "mcp_tool_call tool=%s args_digest=%s status=%s duration_ms=%.0f",
            tool, digest, status, duration_ms,
        )


def _server_instructions() -> str:
    """`SERVER_INSTRUCTIONS` with the tool and category counts filled in.

    MIS-E2E-161. The counts were hand-written and had drifted three ways at
    once: this string said 92 tools / 13 categories, `manual/docs/advanced/
    mcp-server.md` said 97 / 13, and the generated contract said 116 / 14. The
    contract's is right because it is DERIVED; the other two were maintained by
    memory.

    An agent reading these instructions is told how big the surface is in order
    to decide whether to call `miworkbench_howto`. A number 24 tools short
    understates exactly the thing it exists to warn about.
    """
    # COUNTED BY AST, because the obvious sources both fail here:
    #   * summing `CATEGORY_MODULES` counts MODULES (16), not tools;
    #   * `_all_tools()` BUILDS A SERVER, and build_server calls this — infinite
    #     recursion, which is how I found out.
    # So walk the same `@mcp.tool()` decorators `contract.py` walks. No imports
    # beyond the tool modules, which are being imported anyway.
    import ast as _ast
    import inspect as _inspect

    categories = all_categories()
    tool_count = 0
    for modules in categories.values():
        for module in modules:
            try:
                tree = _ast.parse(_inspect.getsource(module))
            except (OSError, TypeError, SyntaxError):
                continue
            for node in _ast.walk(tree):
                if not isinstance(node, (_ast.AsyncFunctionDef, _ast.FunctionDef)):
                    continue
                for dec in node.decorator_list:
                    target = dec.func if isinstance(dec, _ast.Call) else dec
                    if isinstance(target, _ast.Attribute) and target.attr == "tool":
                        tool_count += 1
                        break
    # Plain replacement, not `.format` — the instructions contain literal
    # braces (a JSON example: `{"unavailable": "miserver", …}`) that `.format`
    # reads as fields and raises on.
    return (
        SERVER_INSTRUCTIONS
        .replace("{tool_count}", str(tool_count or "many"))
        .replace("{category_count}", str(len(categories)))
    )


def _client_for(spec, url: str) -> Any:
    """Construct a product's client from its row's dotted `client_path`.

    Imported lazily so a product whose categories are not requested costs nothing. A client
    that records "who" takes the row's identity (Feature 034, FR-12); one that does not, does
    not — read from its signature rather than from the product's name.
    """
    import importlib
    import inspect as _inspect

    module_path, _, class_name = spec.client_path.partition(":")
    cls = getattr(importlib.import_module(module_path), class_name)
    if "identity" in _inspect.signature(cls).parameters:
        return cls(url, identity=spec.identity)
    return cls(url)


def build_server(settings: MCPSettings, stdio: bool = False) -> tuple[MCPServer, MiWorkbenchClient]:
    """Create the MCPServer server with only the enabled tool categories registered."""
    # `MCP_ALLOW_ANONYMOUS` IS STDIO-ONLY, AS EVERYTHING ALREADY CLAIMED
    # (MIS-E2E-150).
    #
    # This read `not (settings.allow_anonymous or stdio)`, so the flag ALONE
    # satisfied the guard on the HTTP transport. `__main__.py` then adds
    # `BearerAuthMiddleware` only when a token is set — otherwise it logs a
    # warning and serves on `settings.host`, default 0.0.0.0.
    #
    # So an operator hitting this very error, following the remedy this very
    # message gives, got a LAN-reachable UNAUTHENTICATED MCP server exposing
    # `delete_circuit`, GPU steering and label write-back. The manual says a
    # bearer token is "always required" and the message says "for local stdio
    # development only": three statements of a rule nothing enforced.
    #
    # `and stdio` is the enforcement.
    if not settings.auth_token and not (settings.allow_anonymous and stdio) and not stdio:
        raise SystemExit(
            "MCP_AUTH_TOKEN is required — the MCP port is LAN-reachable by default. "
            "Set a token, or set MCP_ALLOW_ANONYMOUS=true for local stdio development only."
        )

    if settings.allow_anonymous and not stdio:
        raise SystemExit(
            "MCP_ALLOW_ANONYMOUS is only honoured on the stdio transport. Over "
            "HTTP the port is LAN-reachable and exposes destructive tools "
            "(delete_circuit, GPU steering, label write-back), so anonymous "
            "access is refused. Set MCP_AUTH_TOKEN instead."
        )

    categories = settings.enabled_categories()
    # mcp 2.x: the transport options (host, stateless, JSON responses, body limit) are
    # arguments of `streamable_http_app`, not of the server — see `build_http_app`.
    mcp = MCPServer(
        "miworkbench",
        instructions=_server_instructions(),
    )
    client = MiWorkbenchClient(settings.api_url)

    # 042 FR-042.9 (CF-5 M6): this pod's old-name uses wait in `push_state` and reach the backend through
    # the `/health`-driven push below. Removing this line fails test_legacy_names_push.py.
    from src.core import legacy_names
    from .legacy_names_push import PushState

    push_state = PushState()
    legacy_names.set_sink(push_state.sink, "mcp")
    mcp.legacy_names_push = push_state  # type: ignore[attr-defined]

    registered = []
    for category, modules in CATEGORY_MODULES.items():
        if category not in categories:
            continue
        for module in modules:
            module.register(mcp, client, settings)
        registered.append(category)

    # External products (Unified MCP, Feature 9; one loop since Feature 034):
    # each product's categories are wired against that product's client and the
    # shared HealthGate. An unset URL skips them at REGISTRATION (one warning per
    # product) — a running-but-unreachable product instead keeps its tools
    # registered and the gate returns a structured unavailable per call
    # (contract §3).
    from .health_gate import OWN_PRODUCT, HealthGate

    gate = HealthGate(
        workbench_url=settings.api_url,
        product_urls={p.spec.name: getattr(settings, p.spec.url_attr)
                      for p in EXTERNAL_PRODUCTS},
    )

    product_clients: list[Any] = []
    requested_products: set[str] = set()
    for product in EXTERNAL_PRODUCTS:
        requested = [c for c in product.categories if c in categories]
        if not requested:
            continue
        requested_products.add(product.spec.name)
        url = getattr(settings, product.spec.url_attr)
        if not url:
            logger.warning(
                "%s categories requested but %s is unset — skipping: %s",
                product.spec.name, product.spec.url_env, sorted(requested),
            )
            continue
        product_client = _client_for(product.spec, url)
        product_clients.append(product_client)
        ctx = ExternalContext(settings=settings, mistudio=client,
                              identity=product.spec.identity)
        for category in requested:
            for module in product.categories[category]:
                module.register(mcp, product_client, gate, ctx)
            registered.append(category)
    logger.info(f"MCP tool categories enabled: {sorted(registered)}")

    @mcp.custom_route("/health", methods=["GET"])
    async def health(request: Request) -> JSONResponse:
        # Per-product availability (US-9.4) — NEVER blocks on live probes:
        # this route answers orchestrator readiness/liveness checks (default
        # timeout 1 s), and a hung backend must degrade its TOOLS, not take
        # the MCP server out of rotation or restart-loop it (009 R2).
        # snapshot() serves the last known state and refreshes in the
        # background. Reasons are coarse categories — the route is
        # unauthenticated and detailed reasons carry internal URLs.
        products = {}
        product_names = [OWN_PRODUCT] + [
            p.spec.name for p in EXTERNAL_PRODUCTS
            if getattr(settings, p.spec.url_attr) or p.spec.name in requested_products
        ]
        for product in product_names:
            available, reason = gate.snapshot(product)
            products[product] = {
                "available": available,
                "reason": gate.public_reason(reason),
            }
        # `unknown_categories` is normally absent. It appears when
        # MCP_TOOL_CATEGORIES names something this build does not recognise —
        # dropped, never honoured. Expected transiently while a manifest change
        # leads its image (ArgoCD syncs in minutes, the image takes about
        # nine); persisting, it is a typo and those tools are OFF. Reported
        # here because the alternative to a startup crash must not be silence.
        body = {"status": "ok", "service": "miworkbench-mcp",
                "categories": sorted(registered), "products": products}
        unknown = sorted(settings.unknown_categories())
        if unknown:
            body["unknown_categories"] = unknown
        # 042 FR-042.9: this process's BR-023 block and its meta (`error` always present, CF-24), plus
        # the push state. A due push is scheduled in the background — with or without uses (the
        # heartbeat) — and never awaited here (the 1-second probe budget above).
        body["legacy_names"] = legacy_names.snapshot()
        body["legacy_names_meta"] = {**legacy_names.meta(), "push": push_state.as_dict()}
        push_state.schedule(client)
        return JSONResponse(body)

    # Shutdown hooks (009 R2: aclose existed with zero callers). __main__
    # and tests can close everything via mcp — build_server's 2-tuple
    # return shape is load-bearing for existing callers.
    closables = [client, gate, *product_clients]

    async def _close_backend_clients() -> None:
        for closable in closables:
            try:
                close = getattr(closable, "aclose", None) or getattr(
                    closable, "close", None)
                if close is not None:
                    await close()
            except Exception:  # noqa: BLE001 — best-effort shutdown
                pass

    mcp.close_backend_clients = _close_backend_clients  # type: ignore[attr-defined]

    # L5 / F-6 (2026-10-21): every tool that returns a JSON list answers with the list as ONE
    # structured value too, and an empty list is never zero content blocks.
    normalise_list_results(mcp)
    # MCP gap 2026-11-10: an argument the tool does not declare is refused, by name, before anything runs.
    refuse_unknown_arguments(mcp)

    return mcp, client


def list_result(items: Any) -> Any:
    """The answer for a tool that returned a list (L5 + F-6, 2026-10-21).

    MCPServer turns a returned list into ONE CONTENT BLOCK PER ITEM, so:

    * L5 — an EMPTY list became ZERO content blocks (`content: []`), which a client cannot tell
      from a dropped response; an end-to-end probe run's own client crashed on it (2026-10-20).
    * F-6 — a client had to parse and stitch N blocks back into the list, and a client that
      truncated its output cut the JSON mid-block.

    LEAST BREAKING FIX: `content` keeps one block per item, exactly as before, so every client
    that iterates blocks still works; an empty list is ONE block, the text `[]`; and the whole list
    is ALSO returned as `structuredContent = {"result": [...]}`, the MCP spec's channel for one
    JSON value (the same `{"result": …}` wrapping MCPServer uses for a tool with a list output
    schema). A client reads `structuredContent.result` and never stitches. Changing `content` to
    a single array block was rejected: it would silently break every client that already reads
    one item per block.
    """
    import pydantic_core
    from mcp.server.mcpserver.utilities.func_metadata import _convert_to_content
    from mcp.types import CallToolResult, TextContent

    values = list(items)
    content = _convert_to_content(values) if values else [TextContent(type="text", text="[]")]
    return CallToolResult(
        content=list(content),
        structured_content={"result": pydantic_core.to_jsonable_python(values, fallback=str)},
        is_error=False,
    )


def _is_plain_list(result: Any) -> bool:
    """A JSON list a tool returned — not a list of MCP content objects, which stay as they are."""
    from mcp.server.mcpserver.utilities.types import Audio, Image
    from mcp.types import AudioContent, EmbeddedResource, ImageContent, ResourceLink, TextContent

    content_types = (TextContent, ImageContent, AudioContent, ResourceLink, EmbeddedResource,
                     Image, Audio)
    return isinstance(result, (list, tuple)) and not any(
        isinstance(item, content_types) for item in result
    )


def normalise_list_results(mcp: MCPServer) -> int:
    """Wrap every registered tool without an output schema so a list result goes through
    `list_result`. Returns how many tools were wrapped. Called by `build_server` — the one place
    every production tool passes through — so no tool module has to remember it."""
    import inspect as _inspect

    manager = mcp._tool_manager  # noqa: SLF001 — no public hook for this in the SDK
    wrapped = 0
    for tool in manager.list_tools():
        if tool.fn_metadata.output_schema is not None:
            continue  # MCPServer already returns structured content for these
        original = tool.fn
        if _inspect.iscoroutinefunction(original):
            @functools.wraps(original)
            async def wrapper(*args, _list_result_original=original, **kwargs):
                result = await _list_result_original(*args, **kwargs)
                return list_result(result) if _is_plain_list(result) else result
        else:
            @functools.wraps(original)
            def wrapper(*args, _list_result_original=original, **kwargs):  # type: ignore[misc]
                result = _list_result_original(*args, **kwargs)
                return list_result(result) if _is_plain_list(result) else result
        # `functools.wraps` keeps the signature visible (`inspect.signature` follows __wrapped__),
        # which the paging and argument guards read; the marker says the wrapper is applied.
        wrapper.normalises_list_results = True  # type: ignore[attr-defined]
        tool.fn = wrapper
        wrapped += 1
    return wrapped


_WORD = re.compile(r"[a-z][a-z0-9_]*")


def _resolve(schema: Any, defs: dict) -> list[dict]:
    """The object schemas `schema` stands for: through `$ref`, `anyOf`/`oneOf`/`allOf`."""
    if not isinstance(schema, dict):
        return []
    ref = schema.get("$ref")
    if isinstance(ref, str) and ref.startswith("#/$defs/"):
        return _resolve(defs.get(ref.split("/")[-1]), defs)
    out = [schema]
    for key in ("anyOf", "oneOf", "allOf"):
        for sub in schema.get(key) or []:
            out += _resolve(sub, defs)
    return out


def unknown_arguments_message(tool_name: str, schema: dict, unknown: list[str]) -> str:
    """What an agent reads when it sent names the tool does not declare (MCP gap, 2026-11-10)."""
    properties = schema.get("properties") or {}
    defs = schema.get("$defs") or {}
    noun = "argument" if len(unknown) == 1 else "arguments"
    lines = [f"Error executing tool {tool_name}: unknown {noun}: {', '.join(unknown)}. Nothing was run."]
    if properties:
        lines.append(f"{tool_name} accepts: " + ", ".join(f"`{n}`" for n in sorted(properties)) + ".")
    else:
        lines.append(f"{tool_name} takes no arguments.")
    # An unknown name that is a field of an object parameter — declared, or named in its description (a
    # free-form object such as start_probe_training_run's `config`) — probably belongs inside it.
    hints = []
    for name in unknown:
        for param, sub in sorted(properties.items()):
            schemas = _resolve(sub, defs)
            if not any(s.get("type") == "object" or "properties" in s for s in schemas):
                continue
            fields = {f for s in schemas for f in (s.get("properties") or {})}
            words = {w for s in [sub, *schemas] for w in _WORD.findall(str(s.get("description") or ""))}
            if name in fields or name in words:
                hints.append(f"`{param}.{name}`")
                break
    if hints:
        lines.append("Inside an object parameter they would be: " + ", ".join(hints) + ".")
    return " ".join(lines)


def refuse_unknown_arguments(mcp: MCPServer) -> int:
    """Refuse, server-wide, a `tools/call` whose arguments name something the tool does not declare.

    The SDK validates arguments with a model that IGNORES names it does not know, so a misplaced or
    misspelt argument silently became its default — a GPU run started with default settings while the agent
    believed it had set them (2026-11-10). Every call reaches a tool through the manager's `call_tool`
    (`MCPServer.call_tool` and the transports alike), so the check sits there, once, before validation and
    before the tool runs, against the tool's ADVERTISED `inputSchema` properties. Called by `build_server`;
    removing that call fails tests/unit/test_mcp_unknown_arguments.py. Returns 1 when it wrapped, 0 when the
    manager was already wrapped.
    """
    manager = mcp._tool_manager  # noqa: SLF001 — no public hook for this in the SDK
    original = manager.call_tool
    if getattr(original, "refuses_unknown_arguments", False):
        return 0
    from mcp.server.mcpserver.exceptions import ToolError

    async def call_tool(name: str, arguments: dict[str, Any], context: Any = None,
                        convert_result: bool = False) -> Any:
        tool = manager.get_tool(name)
        if tool is not None:
            schema = tool.parameters or {}
            unknown = sorted(set(arguments or {}) - set(schema.get("properties") or {}))
            if unknown:
                raise ToolError(unknown_arguments_message(name, schema, unknown))
        return await original(name, arguments, context, convert_result=convert_result)

    call_tool.refuses_unknown_arguments = True  # type: ignore[attr-defined]
    manager.call_tool = call_tool  # type: ignore[method-assign]
    return 1


def wrap_tool_with_audit(mcp: MCPServer) -> None:
    """Wrap registered tools with the audit logger."""
    # MCPServer stores tools in the tool manager; wrap their run for audit logging.
    manager = mcp._tool_manager  # noqa: SLF001 — no public hook for this in the SDK
    for tool in manager.list_tools():
        original = tool.fn

        def make_wrapper(fn, tool_name):
            async def wrapper(*args, **kwargs):
                start = time.monotonic()
                try:
                    result = await fn(*args, **kwargs)
                    AuditToolLogger.log(tool_name, kwargs, "ok", (time.monotonic() - start) * 1000)
                    return result
                except Exception as e:
                    AuditToolLogger.log(tool_name, kwargs, f"error:{type(e).__name__}",
                                        (time.monotonic() - start) * 1000)
                    raise
            return wrapper

        tool.fn = make_wrapper(original, tool.name)


def streamable_http_options(settings: MCPSettings) -> dict[str, Any]:
    """The streamable-HTTP transport's options, applied by `build_http_app`.

    mcp 1.x took these on the `FastMCP` constructor; mcp 2.x (`MCPServer`) takes them on
    `streamable_http_app`, so they moved here unchanged:

    * `stateless_http` — no sessions are kept, which is also why GHSA-84m7-p3x7-pcfv (unbounded
      session accumulation, fixed in mcp 1.30.0) never applied here. Pinned by
      test_mcp_transport_limits.py::test_the_server_issues_no_session_ids.
    * `json_response` — one JSON body per request, no SSE stream.
    * `max_request_body_size` — see `MCP_MAX_REQUEST_BODY_BYTES`.
    * `host` — the SDK turns on DNS-rebinding protection (Host must be localhost) when the bind
      host is a loopback address, exactly as 1.x did from the constructor's `host`; on the
      default 0.0.0.0 it stays off, as before.
    """
    return {
        "stateless_http": True,
        "json_response": True,
        "max_request_body_size": MCP_MAX_REQUEST_BODY_BYTES,
        "host": settings.host,
    }


def build_http_app(mcp: MCPServer, settings: MCPSettings) -> Any:
    """The ASGI app `__main__` serves: the streamable-HTTP app behind `BearerAuthMiddleware`.

    G1 (2026-10-21). Before an Ingress puts this on a public name, the claim "unauthenticated
    requests are refused" has to hold for the app as ASSEMBLED, not for the middleware on a toy
    app — so the assembly lives here, `__main__` calls it, and
    `tests/unit/test_mcp_http_auth.py` drives THIS app: no token / a wrong token → 401 on
    `/mcp`, the right token → served, `/health` open. `build_server` already refuses to start
    the HTTP transport without a token (MIS-E2E-150), so the middleware is always added here.
    """
    if not settings.auth_token:
        raise SystemExit(
            "MCP_AUTH_TOKEN is required for the HTTP transport — refusing to serve without it."
        )
    app = mcp.streamable_http_app(**streamable_http_options(settings))
    app.add_middleware(BearerAuthMiddleware, token=settings.auth_token)
    # 042 FR-042.9 (TI-9): added AFTER the bearer middleware, so it wraps it — a request refused for its
    # token on an old host is still recorded (it arrived: evidence of use). Removing this line fails
    # test_old_hosts_recorded.py::test_mcp_app_records_old_host_even_when_refused.
    from src.core.legacy_host_middleware import LegacyHostMiddleware

    app.add_middleware(LegacyHostMiddleware)
    return app
