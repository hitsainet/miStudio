"""The external products the MCP server can reach, as data (Feature 034, FR-1).

ONE ROW PER PRODUCT. Before this table the miLLM product was spelled out by hand at ten sites —
the server's registration block, its `/health` list, the health gate's paths and its
not-configured message, the contract generator, `mistudio_howto`'s enumeration and four test
guards — each as `{**CATEGORY_MODULES, **MILLM_CATEGORY_MODULES}` or a literal `"millm"`. Adding
a second product meant finding all ten, and the guards that were meant to catch a missed one were
among the ten.

A LEAF MODULE, deliberately: `health_gate.py` needs the health paths and every tool module imports
`health_gate`, so this file imports nothing from the package. The category maps live in
`tools/__init__.py`, which joins them to these rows (`EXTERNAL_PRODUCTS`).
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ProductSpec:
    """What the MCP server needs to know to reach one external product."""

    #: Health-gate product key and the label in `{"unavailable": <name>, …}`.
    name: str
    #: Health route relative to the product's base URL.
    health_path: str
    #: The bare environment variable an operator sets (not `MCP_`-prefixed).
    url_env: str
    #: The `MCPSettings` property that resolves the URL (override first, then env).
    url_attr: str
    #: The `MCPSettings` field that overrides the URL for one settings object.
    override_field: str
    #: `"module:Class"` of the HTTP client, imported lazily by `server.py`.
    client_path: str
    #: Who holds approvals for the product's gated actions: `"mistudio"` (miStudio's own
    #: `/mcp/approvals` queue, used for miLLM model loads, FR-20) or `"product"` (the product's
    #: REST API holds and enforces them; the proxy passes its answer through, FR-11).
    approval_owner: str
    #: The identity this MCP path presents where the product records "who" (FR-12, C5).
    identity: str


MILLM = ProductSpec(
    name="millm",
    health_path="/api/health",
    url_env="MILLM_API_URL",
    url_attr="millm_api_url",
    override_field="millm_api_url_override",
    client_path="src.mcp_server.millm_client:MiLLMClient",
    approval_owner="mistudio",
    identity="agent:mistudio-mcp",
)

DATAWORKS = ProductSpec(
    name="dataworks",
    # miDataworks ADR-013; confirmed against its served routes at the FR-9 precondition read.
    health_path="/api/health",
    url_env="MIDATAWORKS_API_URL",
    url_attr="midataworks_api_url",
    override_field="midataworks_api_url_override",
    client_path="src.mcp_server.dataworks_client:DataworksClient",
    approval_owner="product",
    identity="agent:mistudio-mcp",
)

PRODUCTS: tuple[ProductSpec, ...] = (MILLM, DATAWORKS)


def spec_for(name: str) -> ProductSpec:
    """The row for a product name; `KeyError` for a name no row carries."""
    for spec in PRODUCTS:
        if spec.name == name:
            return spec
    raise KeyError(name)
