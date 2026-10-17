"""No MCP tool may reach into the backend's application layer. It runs somewhere else.

⚠ **THE MCP SERVER IS A SEPARATE DEPLOYMENT, AND IT HAS NO DATABASE CREDENTIALS.**
`mistudio-mcp` is its own pod. Its env carries `MCP_AUTH_TOKEN`, `MCP_TOOL_CATEGORIES` and a
`/data` mount — and NOT `DATABASE_URL`. Constructing `src.core.config.Settings` there fails with
six missing fields, so a tool importing `SyncSessionLocal` or calling a service function cannot
run at all, however green its unit tests are.

**That is exactly what happened on 2026-10-02.** The first `millm_recalibrate_probe` re-cut the
threshold by calling `src.services.probe_recalibration.recalibrate_probe` with a sync session, and
it was:

* registered in `MILLM_CATEGORY_MODULES`, `VALID_CATEGORIES`, `DEFAULT_CATEGORIES` **and**
  `k8s/base/mcp.yaml` — all four layers, which is the rule this estate wrote after 16 tools shipped
  registered nowhere;
* asserted by a dedicated payload test, including which of the two probe ids the re-cut received;
* confirmed present in the LIVE registry of the deployed server (152 tools);

and it would have raised `ValidationError` on its first call in production. Every check passed
because every check ran in a process that HAD a database. The reachability rule says a capability
is not shipped until a test fails when its wiring is removed — this one's wiring was intact and
its *runtime* was not, which no reachability test asks about.

**The rule:** a tool reaches its service over HTTP, through the client it is handed. Every other
tool module in this server already does; at the time this file was written, the inventory of
`src.*` imports across all 20 tool modules was exactly the two lines the defect added.

**Feature 034 (FR-3, SC-7) widened it twice.** It scanned `tools/` only, so a CLIENT module —
`millm_client.py`, and now `dataworks_client.py` — could import a service unseen; it now scans the
whole `mcp_server/` package. And it could not see a sibling APP's package: a plain
`import midataworks` (or `millm`, `miforge`) would put another product's code in this pod, which the
suite's standalone rule forbids — products talk over HTTP, MCP and published contracts. Relative
imports are now resolved against the file's real package, so `from .models import x` inside
`tools/` (this package's own `tools/models.py`) is not mistaken for the backend's `src.models`.

Control (recorded 2026-10-16, task 1.11): adding `import midataworks` to `tools/millm_probes.py` →
`test_it_does_not_import_the_backend_application_layer[tools/millm_probes.py]` red; restored, diff
clean.

`ALLOWED` is the escape hatch and is deliberately awkward: an entry is a claim that a module
genuinely needs in-process backend state, which would mean the MCP deployment needs that
configuration too — a deployment decision, not an import decision.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

BACKEND = pathlib.Path(__file__).resolve().parents[2]
PACKAGE_DIR = BACKEND / "src" / "mcp_server"

#: Modules in the backend's application layer that a tool must not import. `core.config` is on the
#: list because it is the thing that raises: the MCP pod cannot build `Settings`.
FORBIDDEN_ROOTS = ("models", "services", "workers", "db", "core", "api", "ml")

#: Sibling apps of the suite. Each is its own deployment and its own repo; their code never runs
#: in this pod (suite rule: interoperation goes through HTTP APIs, MCP tools and published
#: contracts). Matched on the TOP-LEVEL package of an absolute import.
FORBIDDEN_SIBLINGS = ("midataworks", "millm", "miforge")

#: module filename → (import path, why it is legitimately needed in-process).
ALLOWED: dict[str, dict[str, str]] = {}


def _tool_modules() -> list[pathlib.Path]:
    """Every module of the MCP server package — tools, clients, server, gate, contract."""
    return sorted(PACKAGE_DIR.rglob("*.py"))


def _package_of(path: pathlib.Path) -> str:
    """`src.mcp_server.tools` for `src/mcp_server/tools/x.py`; `""` outside the backend tree."""
    try:
        rel = path.resolve().relative_to(BACKEND)
    except ValueError:
        return ""
    # The containing directory is the package for both `x.py` and `__init__.py`.
    return ".".join(rel.parts[:-1])


def _resolve(package: str, level: int, module: str) -> str:
    """The absolute dotted name a relative import reaches, or `""` when it cannot be known."""
    if not package:
        return ""
    parts = package.split(".")
    if level - 1 > len(parts):
        return ""
    base = parts[: len(parts) - (level - 1)]
    return ".".join(base + ([module] if module else []))


def _backend_imports(path: pathlib.Path) -> set[str]:
    """Every import of the backend's application layer in one tool module.

    Walks the AST rather than the text, so a name inside a docstring or a comment — this file's
    own docstring quotes `SyncSessionLocal` — cannot satisfy or trip the guard. A text scan would
    report that quote as a violation, which is the wrong-occurrence trap in reverse.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    package = _package_of(path)
    found: set[str] = set()

    def backend(absolute: str) -> bool:
        parts = absolute.split(".")
        return len(parts) > 1 and parts[0] == "src" and parts[1] in FORBIDDEN_ROOTS

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if node.level:
                absolute = _resolve(package, node.level, module)
                if absolute:
                    # Resolved against the file's real package: `from ...core.database` in
                    # `src/mcp_server/tools/` reaches `src.core.database`.
                    if backend(absolute):
                        found.add(absolute.split(".", 1)[1])
                elif module.split(".", 1)[0] in FORBIDDEN_ROOTS:
                    # Outside the tree (a scanner self-test sample): judge by the module root.
                    found.add(module)
            else:
                if backend(module):
                    found.add(module.split(".", 1)[1])
                if module.split(".", 1)[0] in FORBIDDEN_SIBLINGS:
                    found.add(module)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if backend(alias.name):
                    found.add(alias.name.split(".", 1)[1])
                if alias.name.split(".", 1)[0] in FORBIDDEN_SIBLINGS:
                    found.add(alias.name)
    return found


class TestTheGuardCanSeeItsOwnInputs:
    def test_it_finds_the_tool_modules(self) -> None:
        modules = _tool_modules()
        assert len(modules) >= 25, f"only {len(modules)} modules found — the path is stale"
        names = {p.relative_to(PACKAGE_DIR).as_posix() for p in modules}
        for known in ("tools/millm_probes.py", "tools/probe_monitors.py",
                      "tools/millm_circuits.py", "millm_client.py", "server.py", "client.py"):
            assert known in names

    def test_relative_imports_resolve_against_the_REAL_package(self) -> None:
        """`from .models import x` in `tools/` is this package's own `tools/models.py`;
        `from ...models import x` there is the backend's ORM. Only the second is forbidden."""
        assert _resolve("src.mcp_server.tools", 1, "models") == "src.mcp_server.tools.models"
        assert _resolve("src.mcp_server.tools", 3, "models.x") == "src.models.x"
        assert _resolve("src.mcp_server", 2, "core.config") == "src.core.config"

    def test_the_scanner_sees_every_SIBLING_form(self, tmp_path) -> None:
        """Feature 034 FR-3: each spelling of another app's package."""
        sample = tmp_path / "sib.py"
        sample.write_text(
            "import midataworks\n"
            "import millm.services.probe_runtime\n"
            "from miforge import trainer\n"
            "from millm.api import x\n"
            "import millmish\n"
            "from . import millm_probes\n"
        )
        assert _backend_imports(sample) == {
            "midataworks", "millm.services.probe_runtime", "miforge", "millm.api",
        }

    def test_the_scanner_can_SEE_a_forbidden_import(self, tmp_path) -> None:
        """A scan that matches nothing asserts nothing. Three arcs here have shipped one."""
        sample = tmp_path / "sample.py"
        sample.write_text(
            "from ...core.database import SyncSessionLocal\n"
            "from ...services.probe_recalibration import recalibrate_probe\n"
            "from ..client import MiStudioClient\n"
        )
        found = _backend_imports(sample)
        assert found == {"core.database", "services.probe_recalibration"}, found

    def test_the_scanner_ignores_a_DOCSTRING_mention(self, tmp_path) -> None:
        """This file's own docstring names `SyncSessionLocal`. A text scan would call that a
        violation — the wrong-occurrence trap, inverted."""
        sample = tmp_path / "doc.py"
        sample.write_text('"""Do not import SyncSessionLocal from ...core.database."""\n')
        assert _backend_imports(sample) == set()


class TestEveryToolStaysOnItsClient:
    @pytest.mark.parametrize(
        "path", _tool_modules(), ids=lambda p: p.relative_to(PACKAGE_DIR).as_posix()
    )
    def test_it_does_not_import_the_backend_application_layer(self, path) -> None:
        offending = _backend_imports(path) - set(
            ALLOWED.get(path.relative_to(PACKAGE_DIR).as_posix(), {})
        )
        assert not offending, (
            f"{path.name} imports {sorted(offending)} from the backend. The MCP server is a "
            f"SEPARATE deployment with no DATABASE_URL — `src.core.config.Settings` cannot even "
            f"be constructed there, so this tool would raise on its first call in production "
            f"however green its tests are. Reach the service over HTTP through the client the "
            f"module is handed, or add an ALLOWED entry stating why the MCP deployment needs "
            f"in-process backend state."
        )

    def test_the_inventory_is_still_EMPTY(self) -> None:
        """No module needs in-process backend state today, and that is the whole point: the
        escape hatch exists so an exception is a decision, and it has never been taken."""
        assert ALLOWED == {}, (
            f"ALLOWED has entries: {sorted(ALLOWED)}. Each is a claim that the MCP deployment "
            f"needs backend configuration it does not have — check k8s/base/mcp.yaml agrees."
        )
