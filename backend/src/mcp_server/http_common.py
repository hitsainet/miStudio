"""Response handling shared by every product client (Feature 034, FTDD §6.5).

Two defects in this server were the same defect in two clients: a timeout message that said
nothing (F20 R3-18) and a 2xx HTML page read as an empty success (F20 R3-17). Each was fixed in
`MiLLMClient` and never carried to its siblings. Three clients now need both behaviours, so they
live here once, and each client calls them (asserted by AST in `test_millm_client_extensions.py`).
"""

import json
from typing import Any

import httpx


def timeout_detail(product_label: str, exc: Exception, method: str, path: str) -> str:
    """An agent-readable timeout message that says WHICH phase timed out.

    `str(exc)` on an httpx timeout is frequently empty. The phase is what matters: a connect
    timeout means the product was never reached, while a READ timeout on a POST means the request
    may already have committed server-side — an agent that retries it can double-apply.
    """
    detail = f"{product_label} request timed out ({type(exc).__name__}) on {method} {path}"
    if str(exc):
        detail += f": {exc}"
    if isinstance(exc, httpx.ReadTimeout):
        detail += (
            " — a read timeout means the request may already have been "
            "applied; check state before retrying."
        )
    return detail


def parse_body(response: httpx.Response) -> tuple[Any, bool]:
    """`(body, unparseable)`.

    `unparseable` separates "no body" from "a body that is not JSON". Collapsing the two is how a
    200 HTML splash page from a misrouted ingress reached an agent as `{}`: a successful, empty
    answer to "what is steering right now?".
    """
    if not response.content:
        return None, False
    try:
        return response.json(), False
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None, True


def non_json_2xx_detail(product_label: str, path: str) -> str:
    """The refusal text for a 2xx whose body is not JSON. Never an empty success."""
    return (
        f"{product_label} returned non-JSON content from {path} — the response did not come "
        "from the API (a misrouted ingress or a proxy error page). This is NOT an empty "
        "result: do not read it as 'nothing is configured'."
    )
