"""ENH-009 P7: the transition aliases are gone (042 FR-042.35; FTASKS 18.3, the inverses; FTID §3.1).

Each test here is the INVERSE of a transition guarantee: it fails while the alias it names still works, and its
control (FTASKS 18.13, Procedure K; record `0xcc/reviews/042_P7_controls_2026-11-13.md`) restores the removed
line and watches it go red. The groups land one commit each (G1 … G9), so each test arrived with its group.

Evidence for the removals: the suite's post-sweep reading `rebrand_checks.py legacy-window --all --since
2026-11-13T10:11:41Z` (R31), every miWorkbench env and host alias `eligible`.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from src.core import legacy_names

REPO = Path(__file__).resolve().parents[3]


# ── G1: environment fallbacks (FTASKS 18.4) ────────────────────────────────────────────────────────────────
ENV_NAMES = sorted({n for r in legacy_names.ENV for n in (r.new, r.old)})
#: An old value no reader could produce by default, per row.
OLD_VALUES = {"MISERVER_API_URL": "http://old-srv.invalid", "MIWORKBENCH_BYPASS_PIN": "true",
              "MIWORKBENCH_API_URL": "http://old-wb.invalid", "MIWORKBENCH_GPU_RECYCLE_DIR": "/tmp/old-recycle-p7",
              "MIWORKBENCH_GPU_RECYCLE_RSS_GIB": "7.0625", "MIWORKBENCH_GPU_WORKER_MEMORY_DIR": "/tmp/old-mem-p7"}


def _read(new: str):
    """The value the PRODUCTION reader of row `new` ends up with (the readers T-07 uses)."""
    from src.core.config import Settings

    if new == "MISERVER_API_URL":
        from src.mcp_server.config import MCPSettings

        return Settings(_env_file=None).miserver_api_url, MCPSettings(_env_file=None).miserver_api_url
    if new == "MIWORKBENCH_BYPASS_PIN":
        return Settings(_env_file=None).bypass_settings_pin
    if new == "MIWORKBENCH_API_URL":
        from src.mcp_server.config import MCPSettings

        return MCPSettings(_env_file=None).api_url
    if new == "MIWORKBENCH_GPU_RECYCLE_DIR":
        from src.workers import gpu_worker_recycle

        return legacy_names.env_from(None, gpu_worker_recycle.RECYCLE_DIR_ENV)
    if new == "MIWORKBENCH_GPU_RECYCLE_RSS_GIB":
        from src.workers import gpu_worker_recycle

        return gpu_worker_recycle.threshold_bytes()
    if new == "MIWORKBENCH_GPU_WORKER_MEMORY_DIR":
        from src.services import gpu_worker_memory

        return gpu_worker_memory.memory_dir()
    raise AssertionError(f"no reader for {new}")


@pytest.fixture
def clean_env(monkeypatch):
    for name in ENV_NAMES + ["BYPASS_SETTINGS_PIN"]:
        monkeypatch.delenv(name, raising=False)
    legacy_names.reset()
    yield monkeypatch
    legacy_names.reset()


def test_every_env_row_has_an_old_value():
    assert set(OLD_VALUES) == {r.new for r in legacy_names.ENV}, "a new ENV row needs an old value here"


@pytest.mark.parametrize("new", sorted(OLD_VALUES))
def test_retired_env_rows_not_honoured_and_reported(new, clean_env):
    """With ONLY the old name set, the real reader returns what it returns with nothing set (the old name is
    not read), the use is not recorded, and `retired_but_set` names the pair (FTID §3.1: `retired`)."""
    row = next(r for r in legacy_names.ENV if r.new == new)
    assert row.status == "retired", f"{row.old} -> {row.new} is still {row.status!r}"
    unset = _read(new)
    legacy_names.reset()
    clean_env.setenv(row.old, OLD_VALUES[new])
    assert _read(new) == unset, f"the retired {row.old} is still read"
    snap = legacy_names.snapshot()
    assert {"old": row.old, "new": row.new} in snap["retired_but_set"], snap["retired_but_set"]
    assert ("env", row.old) not in {(u["kind"], u["old"]) for u in snap["in_use"]}, snap["in_use"]
    assert OLD_VALUES[new] not in repr(snap), "a value reached the evidence"


@pytest.mark.parametrize("new", ["MISERVER_API_URL", "MIWORKBENCH_BYPASS_PIN"])
def test_retired_settings_name_in_a_dotenv_file_not_honoured_and_reported(new, clean_env, tmp_path):
    """The Settings rows also arrive from a `.env` file, where `os.environ` never shows them (FTID §3.2)."""
    from src.core.config import Settings

    row = next(r for r in legacy_names.ENV if r.new == new)
    env_file = tmp_path / ".env"
    env_file.write_text(f"{row.old}={OLD_VALUES[new]}\n")
    settings = Settings(_env_file=env_file)
    value = settings.miserver_api_url if new == "MISERVER_API_URL" else settings.bypass_settings_pin
    assert value == ("" if new == "MISERVER_API_URL" else False), f"the retired {row.old} in .env is still read"
    snap = legacy_names.snapshot()
    assert {"old": row.old, "new": row.new} in snap["retired_but_set"], snap["retired_but_set"]
    assert snap["in_use"] == []


def test_no_deploy_file_falls_back_to_a_retired_name():
    """Compose interpolation and the manifests set the new names only (the compose file fell back to
    `${MILLM_API_URL}` until P7)."""
    olds = [r.old for r in legacy_names.ENV]
    files = [REPO / "docker-compose.yml", REPO / "docker-compose.dev.yml", REPO / "docker-compose.hub.yml",
             *sorted((REPO / "k8s" / "base").glob("*.yaml"))]
    found = []
    for path in files:
        if not path.is_file():
            continue
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            code = line.split("#", 1)[0]
            found += [f"{path.relative_to(REPO)}:{n}: {old}" for old in olds if re.search(rf"\b{old}\b", code)]
    assert found == [], found


# ── G2: old MCP category keys (FTASKS 18.5) ────────────────────────────────────────────────────────────────
def _old_category_keys() -> dict[str, str]:
    """The seven old keys and their twins, from the P3 cut's rename map (kept as history after P7)."""
    import json

    doc = json.loads((REPO / "backend" / "tests" / "support" / "mcp_rename_map.json").read_text(encoding="utf-8"))
    return dict(doc["categories"])


def test_old_category_keys_unknown(monkeypatch):
    """A manifest naming only the old `millm_*` keys switches NOTHING on since P7: each key is reported in
    `unknown_categories`, the live registry built from it holds no `miserver_*` tool, and nothing is recorded."""
    from src.mcp_server.config import VALID_CATEGORIES, MCPSettings
    from src.mcp_server.server import build_server

    keys = _old_category_keys()
    assert len(keys) == 7 and all(k.startswith("millm_") for k in keys), keys
    assert set(keys.values()) <= VALID_CATEGORIES
    monkeypatch.setenv("MISERVER_API_URL", "http://miserver.test")
    legacy_names.reset()
    try:
        settings = MCPSettings(tool_categories=",".join(sorted(keys)), allow_anonymous=True, _env_file=None)
        assert settings.unknown_categories() == set(keys), settings.unknown_categories()
        assert settings.enabled_categories() == set()
        mcp, _ = build_server(settings, stdio=True)
        names = {t.name for t in mcp._tool_manager.list_tools()}  # noqa: SLF001
        assert not {n for n in names if n.startswith(("miserver_", "millm_"))}, "an old key still enabled tools"
        assert legacy_names.snapshot()["in_use"] == []
        # The new keys still do (the control's twin: the registry is not simply empty).
        settings = MCPSettings(tool_categories=",".join(sorted(keys.values())), allow_anonymous=True, _env_file=None)
        mcp, _ = build_server(settings, stdio=True)
        assert {t.name for t in mcp._tool_manager.list_tools() if t.name.startswith("miserver_")}  # noqa: SLF001
    finally:
        legacy_names.reset()


# ── G3: relay rewrite (FTASKS 18.6; TI-21) ─────────────────────────────────────────────────────────────────
def test_relay_rewrite_gone_exact_hints_only(monkeypatch):
    """miServer's hints are `miserver_*` (its FR-040.35), so the client relays `error.next` UNCHANGED: a
    `millm_<x>` from a miServer is no longer turned into `miserver_<x>`, through the client the REAL
    `build_server()` built. The hint test reads exact names only (`LIVENESS_MODE`, flipped in this commit)."""
    import asyncio

    import httpx

    from src.mcp_server import server as server_mod
    from src.mcp_server.client import BackendError
    from src.mcp_server.config import MCPSettings
    from src.mcp_server.miserver_client import MiServerClient
    from src.mcp_server.products import PRODUCTS
    from src.mcp_server.tools import all_categories
    from tests.support import miserver_hints

    def answer(request: httpx.Request) -> httpx.Response:
        return httpx.Response(409, json={"success": False, "data": None, "error": {
            "code": "MODEL_NOT_LOADED", "message": "no model", "details": None, "next": "millm_list_probes"}})

    for spec in PRODUCTS:
        monkeypatch.setenv(spec.url_env, f"http://{spec.name}.test")
    built: list[MiServerClient] = []
    real = server_mod._client_for  # noqa: SLF001

    def capture(spec, url):
        client = real(spec, url)
        if isinstance(client, MiServerClient):
            client._http = httpx.AsyncClient(base_url=url, transport=httpx.MockTransport(answer))  # noqa: SLF001
            built.append(client)
        return client

    monkeypatch.setattr(server_mod, "_client_for", capture)
    settings = MCPSettings(tool_categories=",".join(sorted(all_categories())), allow_anonymous=True)
    mcp, _ = server_mod.build_server(settings, stdio=True)
    live = {t.name for t in mcp._tool_manager.list_tools()}  # noqa: SLF001
    assert "miserver_list_probes" in live and "millm_list_probes" not in live
    (client,) = built
    for call in (client.get("/api/probes"), client.post("/api/models/1/load"), client.raw_get("/api/x/export")):
        with pytest.raises(BackendError) as info:
            asyncio.run(call)
        detail = info.value.detail
        error = detail["error"] if isinstance(detail, dict) and isinstance(detail.get("error"), dict) else detail
        assert error["next"] == "millm_list_probes", f"the relay rewrite still runs: {error['next']!r}"
    assert miserver_hints.LIVENESS_MODE == "exact"
    assert miserver_hints.liveness_problems(["millm_list_probes"], live, [frozenset(live)]) == [
        "millm_list_probes: not a tool of the live registry"]


# ── G4: the miServer checkout's old names (FTASKS 18.7) ────────────────────────────────────────────────────
#: The test switches retired with G4. `MISTUDIO_REQUIRE_CROSS_REPO_CHECKS` is NOT here: it stays as the suite's
#: switch (FTASKS 18.1, decided 2026-11-13; 040 TPD-18, TI-19).
G4_RETIRED = ("MILLM_REPO", "MILLM_PYTHON")


def test_millm_repo_alias_gone(monkeypatch, tmp_path):
    """`sibling_repos.millm_repo` is gone, `$MILLM_REPO` and an old `miLLM` folder are not consulted, and no
    test reads `$MILLM_PYTHON` (AST: the name is a string constant only in the guards that look for it)."""
    import ast

    from tests.support import cross_repo, sibling_repos

    assert not hasattr(sibling_repos, "millm_repo"), "the millm_repo alias is still exported"
    assert sibling_repos.MISERVER_CANDIDATES == (("miServer", "MISERVER_REPO"),)
    monkeypatch.delenv("MISERVER_REPO", raising=False)
    monkeypatch.setenv("MILLM_REPO", str(tmp_path / "old-env"))
    parent = tmp_path / "suite"
    (parent / "miLLM").mkdir(parents=True)
    monkeypatch.setattr(sibling_repos, "REPO_ROOT", parent / "miWorkbench")
    assert sibling_repos.miserver_repo() == parent / "miServer", "an old name located miServer's checkout"
    assert cross_repo.REQUIRE_OLD == "MISTUDIO_REQUIRE_CROSS_REPO_CHECKS", "18.1: the suite's switch stays"

    here = Path(__file__).resolve()
    guards = {here, REPO / "backend" / "tests" / "unit" / "test_cross_repo_flag.py"}
    found = []
    for root in (REPO / "backend" / "tests", REPO / "rule_runner" / "tests"):
        for path in sorted(root.rglob("*.py")):
            if path in guards:
                continue
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                if isinstance(node, ast.Constant) and node.value in G4_RETIRED:
                    found.append(f"{path.relative_to(REPO)}:{node.lineno}: {node.value}")
    assert found == [], found


# ── G5: old hosts and origins (FTASKS 18.8) ────────────────────────────────────────────────────────────────
def test_old_hosts_not_served_and_origins_gone():
    """No rendered Ingress serves an old host (every `host` record is retired), `ALLOWED_ORIGINS` and the code
    default name no old browser host, the dev server's `allowedHosts` drops the old ingress names, and the
    public manual's rule no longer allows `mcp-studio.hitsai.net`."""
    import importlib.util
    import json

    from src.core.config import Settings
    from tests.support import k8s_render

    docs = k8s_render.render()
    served = {r["host"] for d in docs if d.get("kind") == "Ingress" for r in d["spec"].get("rules") or []
              if r.get("host")}
    assert "k8s-miworkbench.hitsai.local" in served and "mcp-workbench.hitsai.net" in served, sorted(served)
    old_hosts = {h.old for h in legacy_names.HOSTS}
    assert all(h.status == "retired" for h in legacy_names.HOSTS)
    assert sorted(served & old_hosts) == [], f"old hosts still served: {sorted(served & old_hosts)}"
    old_browser = {"k8s-mistudio.hitsai.local", "k8s-mistudio.hitsai.net", "mistudio.hitsai.net"}
    (origins,) = [json.loads(v) for d in docs if d.get("kind") == "Deployment"
                  for _, name, v in k8s_render.env_values(d) if name == "ALLOWED_ORIGINS"]
    assert "https://miworkbench.hitsai.net" in origins
    stale = sorted(o for o in [*origins, *Settings.model_fields["allowed_origins"].default]
                   if o.split("://", 1)[-1] in old_browser)
    assert stale == [], f"old origins still allowed: {stale}"
    vite = (REPO / "frontend" / "vite.config.ts").read_text(encoding="utf-8")
    m = re.search(r"allowedHosts:\s*\[([^\]]*)\]", vite)
    assert m, "vite.config.ts has no allowedHosts array (the check would read nothing)"
    allowed = set(re.findall(r"'([^']+)'", m.group(1)))
    assert "k8s-miworkbench.hitsai.local" in allowed, allowed
    assert sorted(allowed & old_hosts) == [], f"the dev server still allows {sorted(allowed & old_hosts)}"
    spec = importlib.util.spec_from_file_location("check_operator_strings",
                                                  REPO / "manual" / "scripts" / "check_operator_strings.py")
    checker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(checker)
    assert checker.scan_text("see mcp-studio.hitsai.net"), "the manual still allows the old MCP host"
    assert checker.scan_text("see mcp-workbench.hitsai.net") == []


# ── G6: the ExternalName and the old namespace (FTASKS 18.9) ───────────────────────────────────────────────
def test_externalname_and_old_destination_gone():
    """No rendered object lives in the old namespace and no ExternalName Service is rendered; the AppProject's
    destinations are the app's own namespace only. (The live Service and namespace are deleted by hand: the
    Application does not prune, PD-12.)"""
    import yaml

    from tests.support import k8s_render

    docs = k8s_render.render()
    assert any(d.get("kind") == "Deployment" for d in docs), "nothing rendered"
    namespaces = {d["metadata"].get("namespace") for d in docs if d.get("metadata")} - {None}
    assert namespaces == {"miworkbench"}, namespaces
    assert [d["metadata"]["name"] for d in docs if d.get("kind") == "Service"
            and d["spec"].get("type") == "ExternalName"] == []
    assert not (REPO / "k8s" / "base" / "transition.yaml").exists()
    app = list(yaml.safe_load_all((REPO / "k8s" / "argocd" / "miworkbench-app.yaml").read_text(encoding="utf-8")))
    (project,) = [d for d in app if d and d.get("kind") == "AppProject"]
    assert [d["namespace"] for d in project["spec"]["destinations"]] == ["miworkbench"], project["spec"]


# ── G7: the old image repositories and source repositories (FTASKS 18.10) ──────────────────────────────────
def test_single_image_push():
    """Each image goes to ONE Docker Hub repository, `hitsai/miworkbench-<x>`: the old `hitsai/mistudio-<x>`
    stopped receiving pushes at P7 (DR-27's transition)."""
    import subprocess

    script = REPO / ".github" / "scripts" / "image_repos.sh"
    for image in ("backend", "frontend", "rule-runner"):
        out = subprocess.run(["bash", "-c", f'source "{script}"; image_repos {image}'], capture_output=True,
                             text=True, timeout=30, check=True).stdout.split()
        assert out == [f"hitsai/miworkbench-{image}"], (image, out)


def test_old_source_repo_gone():
    """No Argo CD AppProject admits, and no Application reads, an old repository URL (the export's repo records,
    all retired since P7)."""
    import yaml

    from tests.support import legacy_export

    olds = {r["old"].lower() for r in legacy_export.load().repos()}
    assert olds, "the export lists no repo record"
    urls = []
    for path in sorted((REPO / "k8s" / "argocd").glob("*.yaml")):
        for doc in yaml.safe_load_all(path.read_text(encoding="utf-8")):
            if not doc:
                continue
            spec = doc.get("spec") or {}
            urls += [(path.name, u) for u in spec.get("sourceRepos") or []]
            if isinstance(spec.get("source"), dict):
                urls.append((path.name, spec["source"].get("repoURL", "")))
    assert ("miworkbench-app.yaml", "https://github.com/Onegaishimas/miWorkbench.git") in urls, urls
    stale = [(f, u) for f, u in urls
             if re.sub(r"^https://github\.com/|\.git$", "", u).lower() in olds]
    assert stale == [], stale


# ── G8: the k8s helpers' old names (FTASKS 18.11) ──────────────────────────────────────────────────────────
def test_millm_helper_aliases_gone():
    """Sourcing `scripts/k8s-helpers.sh` defines no `millm_*` function and no `MILLM_*` variable any more; the
    `miserver_*` twins are still there (so the check reads a script that loaded)."""
    import subprocess

    script = REPO / "scripts" / "k8s-helpers.sh"
    if not script.is_file():
        if not (REPO / "0xcc").is_dir():
            pytest.skip("mirror-shaped tree: scripts/ is stripped by sync-to-clean.yml")
        pytest.fail(f"{script} is missing from the private repository")
    probe = (f'source "{script}" >/dev/null 2>&1; '
             'compgen -A function | grep -E "^(millm|miserver)_" ; compgen -v | grep -E "^(MILLM|MISERVER)_" ; true')
    out = subprocess.run(["bash", "-c", probe], capture_output=True, text=True, timeout=30,
                         env={"PATH": "/usr/bin:/bin", "HOME": "/tmp"}).stdout.split()
    assert {"miserver_check", "miserver_deploy", "miserver_status", "MISERVER_NS"} <= set(out), out
    old = sorted(n for n in out if n.startswith(("millm_", "MILLM_")))
    assert old == [], f"old helper names still defined: {old}"
