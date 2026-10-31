"""Property 1 and the deploy shape of miStudio's operator-code runner (Feature 040 S4; FTDD Appendix C.4, C.7; FTASKS 25.5).

Read from what ArgoCD deploys: `k8s/base/kustomization.yaml` and the resources IT lists (never a hand list of
files). The runner is found by walking those resources.

* **Property 1 (no secret).** Every secret-carrying reference in the base is DERIVED (each `secretKeyRef`,
  `secretRef`, `configMapKeyRef`, `configMapRef` anywhere, and the environment variable names they feed); none
  reaches the runner container; no volume, no service-account token, a strict security context, no GPU
  (`NVIDIA_VISIBLE_DEVICES == "void"`, no `nvidia.com/gpu`). The live half (FTASKS 26.3) compares the runner's
  `/health` `env_names` with `secret_env_names()` below.
* **Property 2's manifest half.** The NetworkPolicy selects exactly the runner, has EMPTY egress and admits only
  the backend pod's label on 8080 — the MCP pod is not admitted.
* **Deploy safety (S4 choice).** `seccompProfile: RuntimeDefault` (never a node-installed Localhost profile) and
  nothing in `mistudio-backend` waits for the runner (no init container, probe or startup check names it): absent,
  the backend serves and code rules read pending (D41).
* **Wiring.** The API container and `celery-worker-rules` — the only callers — carry `CODE_RULE_RUNNER_URL`
  pointing at the runner's Service; the MCP pod does not.

Controls (040 controls record, S4): add a `secretKeyRef` to the runner → red; remove the NetworkPolicy → red;
drop `rule-runner.yaml` from the kustomization → red; Localhost seccomp → red; drop the URL from the rules worker → red.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterator, List, Set

import pytest
import yaml

REPO = Path(__file__).resolve().parents[3]
BASE = REPO / "k8s" / "base"
RUNNER = "mistudio-rule-runner"
RUNNER_URL = "http://mistudio-rule-runner:8080"


def kustomization() -> Dict[str, Any]:
    return yaml.safe_load((BASE / "kustomization.yaml").read_text())


def deployed_documents() -> List[Dict[str, Any]]:
    docs: List[Dict[str, Any]] = []
    for resource in kustomization()["resources"]:
        docs.extend(d for d in yaml.safe_load_all((BASE / resource).read_text()) if d)
    return docs


def _find(kind: str, name: str) -> Dict[str, Any]:
    found = [d for d in deployed_documents() if d.get("kind") == kind and d["metadata"]["name"] == name]
    assert len(found) == 1, f"{kind} {name} is not deployed by k8s/base/kustomization.yaml (found {len(found)})"
    return found[0]


def _walk(node: Any) -> Iterator[Any]:
    yield node
    if isinstance(node, dict):
        for value in node.values():
            yield from _walk(value)
    elif isinstance(node, list):
        for value in node:
            yield from _walk(value)


def secret_env_names() -> Set[str]:
    """Every environment variable name the deployed base feeds from a Secret or ConfigMap (derived)."""
    names: Set[str] = set()
    for doc in deployed_documents():
        for node in _walk(doc):
            if isinstance(node, dict) and isinstance(node.get("valueFrom"), dict):
                source = node["valueFrom"]
                if "secretKeyRef" in source or "configMapKeyRef" in source:
                    names.add(node["name"])
    return names


def _containers(deployment: Dict[str, Any]) -> List[Dict[str, Any]]:
    spec = deployment["spec"]["template"]["spec"]
    return list(spec.get("containers") or []) + list(spec.get("initContainers") or [])


def _env(container: Dict[str, Any]) -> Dict[str, Any]:
    return {e["name"]: e for e in container.get("env") or []}


def test_the_runner_is_in_the_kustomization_with_its_own_image():
    k = kustomization()
    assert "rule-runner.yaml" in k["resources"]
    assert any(i["name"] == "hitsai/mistudio-rule-runner" for i in k["images"])
    deployment = _find("Deployment", RUNNER)
    assert [c["image"].split(":")[0] for c in _containers(deployment)] == ["hitsai/mistudio-rule-runner"]


def test_the_derivation_finds_the_known_secrets():
    """A derivation that found nothing would pass Property 1 vacuously."""
    names = secret_env_names()
    assert {"DATABASE_URL", "SECRET_KEY", "MCP_AUTH_TOKEN"} <= names


def test_property_1_no_secret_reaches_the_runner():
    deployment = _find("Deployment", RUNNER)
    pod = deployment["spec"]["template"]["spec"]
    for node in _walk(deployment):
        if isinstance(node, dict):
            for key in ("secretKeyRef", "secretRef", "configMapKeyRef", "configMapRef", "envFrom", "secret",
                        "configMap", "projected", "hostPath"):
                assert key not in node, f"the runner carries {key!r} (Property 1)"
    assert not pod.get("volumes")
    for container in _containers(deployment):
        assert not container.get("volumeMounts")
        assert not container.get("envFrom")
        env = _env(container)
        assert not (set(env) & secret_env_names()), set(env) & secret_env_names()
        assert all("value" in e and "valueFrom" not in e for e in env.values())
        assert env["NVIDIA_VISIBLE_DEVICES"]["value"] == "void"
        limits = (container.get("resources") or {}).get("limits") or {}
        assert not any("gpu" in k for k in limits)
    assert pod["automountServiceAccountToken"] is False


def test_the_security_context_is_strict_and_needs_nothing_on_the_node():
    deployment = _find("Deployment", RUNNER)
    pod = deployment["spec"]["template"]["spec"]
    assert pod["securityContext"]["runAsNonRoot"] is True
    assert int(pod["securityContext"]["runAsUser"]) != 0
    # ⚠ DEPLOY SAFETY: RuntimeDefault, never Localhost — a missing node profile must not block a pod.
    assert pod["securityContext"]["seccompProfile"] == {"type": "RuntimeDefault"}
    for container in _containers(deployment):
        ctx = container["securityContext"]
        assert ctx["readOnlyRootFilesystem"] is True
        assert ctx["allowPrivilegeEscalation"] is False
        assert ctx["capabilities"]["drop"] == ["ALL"]
        assert "add" not in ctx["capabilities"]
        assert ctx.get("seccompProfile", {"type": "RuntimeDefault"}).get("type") == "RuntimeDefault"
    assert deployment["spec"]["strategy"]["rollingUpdate"]["maxUnavailable"] == 0


def _label(deployment: Dict[str, Any]) -> Dict[str, str]:
    return deployment["spec"]["template"]["metadata"]["labels"]


def test_property_2_the_policy_isolates_the_runner():
    policy = _find("NetworkPolicy", RUNNER)
    runner_labels = _label(_find("Deployment", RUNNER))
    backend_labels = _label(_find("Deployment", "mistudio-backend"))
    mcp_labels = _label(_find("Deployment", "mistudio-mcp"))
    spec = policy["spec"]
    assert spec["podSelector"]["matchLabels"] == runner_labels
    assert set(spec["policyTypes"]) == {"Ingress", "Egress"}
    assert spec["egress"] == []
    assert len(spec["ingress"]) == 1
    rule = spec["ingress"][0]
    assert rule["from"] == [{"podSelector": {"matchLabels": backend_labels}}]
    assert rule["ports"] == [{"protocol": "TCP", "port": 8080}]
    assert backend_labels != mcp_labels and rule["from"][0]["podSelector"]["matchLabels"] != mcp_labels
    service = _find("Service", RUNNER)
    assert service["spec"]["selector"] == runner_labels
    assert service["spec"].get("type", "ClusterIP") == "ClusterIP"


def test_only_the_api_and_the_rules_worker_call_the_runner():
    backend = {c["name"]: c for c in _containers(_find("Deployment", "mistudio-backend"))}
    callers = {name for name, c in backend.items() if "CODE_RULE_RUNNER_URL" in _env(c)}
    assert callers == {"backend", "celery-worker-rules"}
    for name in callers:
        assert _env(backend[name])["CODE_RULE_RUNNER_URL"]["value"] == RUNNER_URL
    mcp = _containers(_find("Deployment", "mistudio-mcp"))
    assert not any("CODE_RULE_RUNNER_URL" in _env(c) for c in mcp)


def test_nothing_in_the_backend_pod_waits_for_the_runner():
    """Deploy safety: an absent runner must leave the backend serving (code rules read pending)."""
    deployment = _find("Deployment", "mistudio-backend")
    spec = deployment["spec"]["template"]["spec"]
    assert not any("rule-runner" in yaml.safe_dump(c) for c in spec.get("initContainers") or [])
    for container in spec["containers"]:
        for probe in ("readinessProbe", "livenessProbe", "startupProbe"):
            assert "rule-runner" not in yaml.safe_dump(container.get(probe) or {})


@pytest.mark.parametrize("service", ["backend", "celery-worker-rules"])
def test_compose_mirrors_the_callers_and_the_runner_is_opt_in(service):
    compose = yaml.safe_load((REPO / "docker-compose.yml").read_text())
    assert "CODE_RULE_RUNNER_URL" in compose["services"][service]["environment"]
    runner = compose["services"]["rule-runner"]
    assert runner["profiles"] == ["code-rules"] and runner["read_only"] is True and runner["cap_drop"] == ["ALL"]
    assert not runner.get("volumes") and not runner.get("env_file")
    # Review R2 debt D5: the memory ceiling the backend bounds `code_rule_memory_bytes` by (test_review_r2_s4_debt.py).
    assert set(runner["environment"]) == {"NVIDIA_VISIBLE_DEVICES", "RUNNER_MAX_MEMORY_BYTES"}


def test_review_r2_image_updater_tracks_every_kustomization_image():
    """Review R2 (S4): every image of the kustomization is tracked by digest by ArgoCD Image Updater
    (`k8s/argocd/mistudio-app.yaml`), so a new runner image rolls the runner like a new backend image rolls the
    backend. Untracked, the runner keeps the `:latest` it first pulled, and a backend that vendors a new contract
    refuses its bundle (every code rule pending)."""
    docs = [d for d in yaml.safe_load_all((REPO / "k8s" / "argocd" / "mistudio-app.yaml").read_text()) if d]
    app = next(d for d in docs if d.get("kind") == "Application")
    notes = app["metadata"]["annotations"]
    prefix = "argocd-image-updater.argoproj.io/"
    tracked = {}
    for item in notes[prefix + "image-list"].split(","):
        alias, image = item.strip().split("=", 1)
        assert notes.get(f"{prefix}{alias}.update-strategy") == "digest", alias
        tracked[notes[f"{prefix}{alias}.kustomize.image-name"]] = image
    for entry in kustomization()["images"]:
        assert entry["name"] in tracked, f"{entry['name']} is not tracked by Image Updater"
        assert tracked[entry["name"]] == f"docker.io/{entry['name']}:latest"

