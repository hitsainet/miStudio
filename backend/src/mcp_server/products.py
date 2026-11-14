"""The external products the MCP server can reach, as data (Feature 034, FR-1).

ONE ROW PER PRODUCT. Before this table the miServer product was spelled out by hand at ten sites —
the server's registration block, its `/health` list, the health gate's paths and its
not-configured message, the contract generator, `miworkbench_howto`'s enumeration and four test
guards — each as `{**CATEGORY_MODULES, **MISERVER_CATEGORY_MODULES}` or a literal `"millm"`. Adding
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
    #: Who holds approvals for the product's gated actions: `"miworkbench"` (this app's own
    #: `/mcp/approvals` queue; no row uses it since ENH-008) or `"product"` (the product's
    #: REST API holds and enforces them; the proxy passes its answer through, FR-11, and a relayed pending
    #: answer carries `decide_in: <name>`).
    approval_owner: str
    #: The identity this MCP path presents where the product records "who" (FR-12, C5).
    identity: str


#: miServer (miServer until its rename). `approval_owner="product"` since ENH-008 (miServer decides its own requests)
#: and READ since the P3 cut (042 TPD-8): `miserver_approvals.relayed()` adds `decide_in: <name>` only for a
#: product-owned queue. The URL comes from MISERVER_API_URL (FR-042.8; the old MILLM_API_URL is retired since P7).
MISERVER = ProductSpec(
    name="miserver",
    health_path="/api/health",
    url_env="MISERVER_API_URL",
    url_attr="miserver_api_url",
    override_field="miserver_api_url_override",
    client_path="src.mcp_server.miserver_client:MiServerClient",
    approval_owner="product",
    identity="agent:miworkbench-mcp",
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
    identity="agent:miworkbench-mcp",
)

PRODUCTS: tuple[ProductSpec, ...] = (MISERVER, DATAWORKS)


def spec_for(name: str) -> ProductSpec:
    """The row for a product name; `KeyError` for a name no row carries."""
    for spec in PRODUCTS:
        if spec.name == name:
            return spec
    raise KeyError(name)
