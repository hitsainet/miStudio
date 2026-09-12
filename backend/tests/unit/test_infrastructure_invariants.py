"""Task 12 — deployment defects that make a deploy or a diagnosis wrong.

MIS-E2E-144  `k8s_deploy` re-applied a STALE second manifest, reverting the
             queue-split and SQL-echo incident fixes — at the moment the
             break-glass procedure is most likely to be used.
MIS-E2E-145  postgres and redis are Deployments over hostPath with the default
             RollingUpdate, so two pods briefly share one data directory.
MIS-E2E-146  compose published the Celery broker and the database on 0.0.0.0.
MIS-E2E-147  the compose frontend port, and an `&&`-chain that reported any
             failure as a schema warning and returned 0.
MIS-E2E-148  a guard that fails open, an ingress exposing `/api/internal`, and
             a global apt keyring.
"""

from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[3]
K8S_BASE = REPO / "k8s" / "base"
COMPOSE = REPO / "docker-compose.yml"
HELPERS = REPO / "scripts" / "k8s-helpers.sh"


def _docs(path: Path):
    return [d for d in yaml.safe_load_all(path.read_text()) if d]


def test_the_repo_layout_is_what_this_file_assumes():
    """Fail closed. Every assertion below reads a file; if the paths moved,
    they would all vanish silently — the failure mode MIS-E2E-148 names."""
    for p in (K8S_BASE, COMPOSE, HELPERS):
        assert p.exists(), f"{p} not found — this guard would pass vacuously"


# ── MIS-E2E-144 · one manifest ─────────────────────────────────────────────

def test_the_stale_standalone_manifest_is_gone():
    """It duplicated `k8s/base` and drifted: no `celery-worker-cpu`, no
    `CELERY_QUEUES`, no `ENVIRONMENT=production`."""
    assert not (REPO / "k8s" / "mistudio-deployment.yaml").exists(), (
        "the standalone manifest is back; k8s_deploy applying it reverts the "
        "queue-split and SQL-echo fixes"
    )


def test_the_deploy_helper_applies_the_kustomize_base():
    src = HELPERS.read_text()
    assert "kubectl apply -k" in src, (
        "k8s_deploy does not apply the kustomize base ArgoCD deploys"
    )
    assert "K8S_MANIFEST" not in src.replace("# `K8S_MANIFEST`", ""), (
        "the stale manifest variable is still in use"
    )


def test_the_deploy_helper_restarts_the_mcp_deployment():
    """`mistudio-mcp` runs the SAME backend image, and was never restarted —
    so new MCP tools stayed invisible after a break-glass deploy."""
    assert "deployment/mistudio-mcp" in HELPERS.read_text()


# ── MIS-E2E-147 · a failed deploy must fail ────────────────────────────────

def test_the_deploy_helper_does_not_swallow_failures_in_an_and_chain():
    """The body was one `&&` chain ending in `|| echo "WARNING: Schema
    verification failed"`, so a failed pull, apply or rollout printed a message
    about SCHEMA and returned 0."""
    src = HELPERS.read_text()
    start = src.index("k8s_deploy()")
    body = src[start: src.index("\n}", start)]
    assert "DEPLOY FAILED at:" in body, (
        "no step reports its own failure; a failed deploy still returns 0"
    )
    # The advisory warning must apply ONLY to schema verification.
    assert body.count("|| \\") == 0, "the &&-chain is back"


# ── MIS-E2E-145 · no two writers on one hostPath ───────────────────────────

@pytest.mark.parametrize("name", ["postgres", "redis"])
def test_stateful_deployments_use_recreate(name):
    """A Deployment over hostPath with RollingUpdate starts the new pod before
    terminating the old one, so two processes briefly hold the same directory."""
    dep = next(
        d for d in _docs(K8S_BASE / f"{name}.yaml") if d.get("kind") == "Deployment"
    )
    strategy = dep["spec"].get("strategy", {})
    assert strategy.get("type") == "Recreate", (
        f"{name} uses {strategy or 'the default RollingUpdate'} over a hostPath "
        f"volume — two pods can hold the same data directory"
    )


# ── MIS-E2E-146 · the broker is not on the LAN ─────────────────────────────

@pytest.mark.parametrize("service", ["postgres", "redis"])
def test_compose_binds_stateful_ports_to_loopback(service):
    """Redis is the CELERY BROKER: LAN reachability meant anyone could enqueue
    GPU jobs and read queued payloads."""
    compose = yaml.safe_load(COMPOSE.read_text())
    ports = compose["services"][service].get("ports", [])
    for spec in ports:
        text = str(spec)
        assert text.count(":") >= 2, (
            f"{service} publishes {text!r} on all interfaces; bind it to an "
            f"address (127.0.0.1 by default)"
        )
        assert "0.0.0.0:" not in text


def test_compose_frontend_targets_the_unprivileged_port():
    """nginx-unprivileged listens on 8080; compose still mapped to 80, so
    http://localhost:3000 was dead."""
    compose = yaml.safe_load(COMPOSE.read_text())
    ports = [str(p) for p in compose["services"]["frontend"].get("ports", [])]
    assert any(p.endswith(":8080") for p in ports), ports


# ── MIS-E2E-148 · the ingress and the keyring ──────────────────────────────

def test_every_host_serving_api_also_denies_api_internal():
    """Both nginx configs deny `/api/internal`; the ingress must agree.

    Parametrised over the hosts found in the file rather than a named pair —
    the `.net` host had the same gap and is the internet-facing one.
    """
    ingresses = [d for d in _docs(K8S_BASE / "ingress.yaml") if d.get("kind") == "Ingress"]
    assert ingresses, "no Ingress found — the scan broke"

    checked = 0
    for ing in ingresses:
        for rule in ing["spec"]["rules"]:
            paths = [p["path"] for p in rule["http"]["paths"]]
            if "/api" in paths:
                checked += 1
                assert "/api/internal" in paths, (
                    f"{rule['host']} exposes /api without denying /api/internal"
                )
    assert checked >= 2, f"only {checked} hosts serve /api — expected both"


def test_the_dockerfile_scopes_its_apt_keyring():
    """`apt-key adv` trusts the key for EVERY repository; `signed-by=` binds it
    to this source alone."""
    src = (REPO / "backend" / "Dockerfile").read_text()
    assert "apt-key adv" not in src.replace("# `apt-key adv`", "")
    assert "signed-by=/usr/share/keyrings/deadsnakes.gpg" in src


def test_the_queue_coverage_guard_fails_closed():
    """The fourth source-scrape guard in this audit to skip when its input
    moved. A guard that vanishes silently is worse than none."""
    src = (REPO / "backend" / "tests" / "unit" / "test_worker_queue_coverage.py").read_text()
    assert "pytest.skip(f\"manifest not found" not in src
    assert "assert MANIFEST.exists()" in src


# ── MIS-E2E-005 / -160 · the PIN bypass and what the PIN gates ─────────────

def test_the_pin_bypass_defaults_to_false():
    """`MISTUDIO_BYPASS_PIN=true` opens Settings with no PIN. It is a recovery
    path requiring filesystem access, and must never be the default."""
    from src.core.config import Settings

    assert Settings.model_fields["bypass_settings_pin"].default is False


def _env_pairs_in_yaml(path: Path) -> dict[str, str]:
    """`{ENV_NAME: value}` across every container in a k8s manifest.

    PARSED, not grepped. My first version scanned line by line for a name and
    "true" on the SAME line — but a Kubernetes env entry puts them on separate
    lines:

        - name: MISTUDIO_BYPASS_PIN
          value: "true"

    so the check could never fire. Mutation control C139 walked straight
    through it.
    """
    pairs: dict[str, str] = {}
    for doc in yaml.safe_load_all(path.read_text()):
        if not doc:
            continue
        spec = (doc.get("spec") or {}).get("template", {}).get("spec", {})
        for container in spec.get("containers", []) or []:
            for entry in container.get("env", []) or []:
                if "name" in entry and "value" in entry:
                    pairs[entry["name"]] = str(entry["value"])
    return pairs


def test_no_shipped_manifest_enables_the_pin_bypass():
    """`MISTUDIO_BYPASS_PIN=true` opens Settings with no PIN.

    It is a recovery path requiring filesystem access. Left on in a manifest it
    is indistinguishable from a product with no PIN at all — and unlike a code
    default, nothing surfaces it.
    """
    offenders = []
    checked = 0

    for path in K8S_BASE.glob("*.yaml"):
        checked += 1
        for name, value in _env_pairs_in_yaml(path).items():
            if name == "MISTUDIO_BYPASS_PIN" and value.lower() == "true":
                offenders.append(f"{path.name}: {name}={value}")

    # Compose and the env template are flat text, so a line scan is right there.
    for path in (COMPOSE, REPO / "docker-compose.dev.yml", REPO / ".env.example"):
        if not path.exists():
            continue
        checked += 1
        for line in path.read_text().splitlines():
            if line.lstrip().startswith("#"):
                continue
            if "MISTUDIO_BYPASS_PIN" in line and "true" in line.lower():
                offenders.append(f"{path.name}: {line.strip()}")

    assert checked >= 5, f"only {checked} deployment artifacts checked — scan broke"
    assert not offenders, f"the PIN bypass is enabled in a shipped artifact: {offenders}"


def test_the_manifest_env_parser_actually_finds_variables():
    """Negative control for the parser. If it returned nothing, the check above
    would pass against a manifest that DOES enable the bypass — which is exactly
    how its first version failed."""
    found = {}
    for path in K8S_BASE.glob("*.yaml"):
        found.update(_env_pairs_in_yaml(path))
    assert len(found) > 5, f"parsed only {len(found)} env vars from k8s/base: {found}"
    assert "ENVIRONMENT" in found, "a known env var is missing — the parser is wrong"


def test_both_sensitive_settings_tabs_are_pin_gated():
    """MIS-E2E-160. Only `api_keys` was wrapped in `<PinGate>`.

    The **Storage** tab arms step-granular checkpoint retention — irreversible
    deletion, where `checkpoint_prune_dry_run: false` means files go — and it
    was the one destructive surface in Settings left ungated, while the manual
    described the whole panel as lockable.
    """
    panel = (REPO / "frontend" / "src" / "components" / "panels" / "SettingsPanel.tsx").read_text()
    for tab, component in (("api_keys", "ApiKeysTab"), ("storage", "StorageTab")):
        needle = f"activeTab === '{tab}' && <PinGate><{component} /></PinGate>"
        assert needle in panel, (
            f"the {tab} tab is not wrapped in <PinGate>; expected `{needle}`"
        )


class TestGpuTasksAreRoutedToTheGpuQueue:
    """MIS-E2E-097, mutation M17.

    The k8s deployment runs two workers: a GPU one consuming the default queue
    and a CPU-only one consuming `low_priority`. Routing is what keeps a GPU
    task off the CPU worker — and `task_routes` globs match the TASK NAME, so a
    task whose name does not match its glob silently lands wherever the
    decorator says.

    M17 removed `queue="steering"` from the GPU steering task and the suite
    stayed green: nothing asserted where GPU work goes. Removing that routing
    sends a CUDA task to a worker with no GPU, where it fails at model load —
    or worse, runs on a node that has one and contends with the real GPU
    worker for the single card.
    """

    #: Tasks that touch CUDA and must never land on a CPU-only worker.
    GPU_MODULES = ("steering_tasks",)

    def _decorated_tasks(self, module_name: str):
        import ast
        from pathlib import Path

        src = (Path(__file__).resolve().parents[2] / "src" / "workers"
               / f"{module_name}.py")
        tree = ast.parse(src.read_text())
        out = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef):
                continue
            for dec in node.decorator_list:
                if not isinstance(dec, ast.Call):
                    continue
                target = getattr(dec.func, "attr", "")
                if target != "task":
                    continue
                kwargs = {k.arg: getattr(k.value, "value", None) for k in dec.keywords}
                out.append((node.name, kwargs))
        return out

    def test_the_scan_finds_the_tasks(self):
        tasks = self._decorated_tasks("steering_tasks")
        assert len(tasks) >= 3, (
            f"only {len(tasks)} celery tasks found in steering_tasks — the AST "
            f"scan is broken, and a broken scan agrees with everything"
        )

    def test_every_gpu_task_declares_the_steering_queue(self):
        offenders = []
        for module_name in self.GPU_MODULES:
            for name, kwargs in self._decorated_tasks(module_name):
                if kwargs.get("queue") != "steering":
                    offenders.append(f"{module_name}.{name} -> {kwargs.get('queue')!r}")
        assert not offenders, (
            "these CUDA tasks are not routed to the steering queue and will be "
            "picked up by the CPU-only worker: " + ", ".join(offenders)
        )

    def test_the_steering_queue_is_actually_consumed(self):
        """A queue nothing consumes is worse than no routing at all.

        The consumer is NOT in the k8s manifest — I asserted that first and it
        failed, which looked like a live defect until I checked. The long-lived
        GPU worker deliberately consumes
        `high_priority,datasets,processing,training,extraction,sae` and not
        `steering`; the steering consumer is spawned ON DEMAND by the API
        process with `-Q steering`, so the GPU sits free between runs. Verified
        against the running pod's command line before changing this assertion.
        """
        import ast
        from pathlib import Path

        endpoint = (Path(__file__).resolve().parents[2] / "src" / "api" / "v1"
                    / "endpoints" / "steering.py")
        tree = ast.parse(endpoint.read_text())

        spawns = []
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call)
                    and getattr(node.func, "attr", "") == "Popen"):
                continue
            argv = node.args[0] if node.args else None
            if not isinstance(argv, ast.List):
                continue
            literals = [e.value for e in argv.elts
                        if isinstance(e, ast.Constant) and isinstance(e.value, str)]
            if "-Q" in literals:
                spawns.append(literals[literals.index("-Q") + 1])

        assert spawns, (
            "nothing spawns a worker with `-Q`; the steering queue has no "
            "consumer, so a routed task would sit in the broker forever"
        )
        assert all(q == "steering" for q in spawns), (
            f"a spawned worker consumes {spawns}, not the steering queue the "
            f"GPU tasks are routed to"
        )
