"""`scripts/k8s-helpers.sh`: the `miserver_*` helpers address miServer's objects (042 FR-042.27a; FTDD §11.5; TPD-19;
FTASKS 7.14, 7.15; defaults flipped in the P2 window, 11.2). The old `millm_*` aliases went at ENH-009 P7 (FTASKS
18.11; `test_legacy_retired.py::test_millm_helper_aliases_gone`).

The script is SOURCED in bash with `k8s()` (the one ssh line), `curl` and `date` stubbed, so nothing leaves
this machine; each function is run and the commands it hands `k8s()` are compared, text and count.

`sync-to-clean.yml` removes `scripts/` from the public mirror, so on a mirror-shaped tree (no `0xcc/`)
the file is absent and these tests skip BY NAME; in the private repository its absence is a failure (TI-23).
"""
from __future__ import annotations

import subprocess

import pytest

from tests.support.legacy_export import REPO_ROOT

HELPERS = REPO_ROOT / "scripts" / "k8s-helpers.sh"
MIRROR_SHAPED = not (REPO_ROOT / "0xcc").is_dir()

STUBS = r'''
k8s() { echo "K8S:$1"; }
curl() { echo '{"results":[{"last_updated":"T"}]}'; }
date() { echo DATE; }
'''


def _helpers() -> None:
    if not HELPERS.is_file():
        if MIRROR_SHAPED:
            pytest.skip(f"mirror-shaped tree: {HELPERS.relative_to(REPO_ROOT)} is stripped by sync-to-clean.yml")
        pytest.fail(f"{HELPERS} is missing from the private repository")


def _bash(body: str, env: dict[str, str] | None = None) -> str:
    _helpers()
    script = f"source {HELPERS} >/dev/null\n{STUBS}\n{body}\n"
    clean = {"PATH": "/usr/bin:/bin", "HOME": "/tmp", **(env or {})}
    res = subprocess.run(["bash", "-c", script], capture_output=True, text=True, timeout=60, env=clean)
    assert res.returncode == 0, res.stderr
    return res.stdout


def _run(fn: str, env: dict[str, str] | None = None) -> list[str]:
    args = "7" if fn.endswith("_logs") else ""
    return _bash(f"{fn} {args}", env).splitlines()


def test_p2_defaults_address_miservers_objects_and_overrides_win():
    # From the P2 window on (PD-18, FTASKS 11.2) the helpers address miServer's namespace and Deployments;
    # exporting other values points them elsewhere.
    assert _run("miserver_status") == ["K8S:microk8s kubectl get pods -n miserver -o wide"]
    assert _run("miserver_logs") == ["K8S:microk8s kubectl logs -n miserver deployment/miserver-backend --tail=7"]
    other = {"MISERVER_NS": "elsewhere", "MISERVER_APP": "other"}
    assert _run("miserver_logs", other) == ["K8S:microk8s kubectl logs -n elsewhere deployment/other-backend --tail=7"]


def test_k8s_deploy_names_miworkbench_in_order():
    """FTASKS 13.11 (FTDD §11.5, P3 half): the break-glass deploy addresses the renamed namespace, Deployments and
    images, in its order — pull, apply, restart all three, wait for each, pods, schema check — and names no old
    object. The Deployment names are read from the rendered manifests, so a manifest rename the helper missed is
    red here (K-13.26)."""
    from tests.support import k8s_render

    deployments = {d["metadata"]["name"] for d in k8s_render.render() if d.get("kind") == "Deployment"}
    assert {"miworkbench-backend", "miworkbench-frontend", "miworkbench-mcp"} <= deployments, deployments
    calls = [line[len("K8S:"):] for line in _bash("k8s_deploy").splitlines() if line.startswith("K8S:")]
    ns = "-n miworkbench"
    assert calls == [
        "docker pull hitsai/miworkbench-frontend:latest && docker pull hitsai/miworkbench-backend:latest",
        f"test -d {calls[1].split()[-1]}",
        f"microk8s kubectl apply -k {calls[1].split()[-1]}",
        f"microk8s kubectl rollout restart deployment/miworkbench-backend deployment/miworkbench-frontend "
        f"deployment/miworkbench-mcp {ns}",
        f"microk8s kubectl rollout status deployment/miworkbench-backend {ns} --timeout=180s",
        f"microk8s kubectl rollout status deployment/miworkbench-frontend {ns} --timeout=180s",
        f"microk8s kubectl rollout status deployment/miworkbench-mcp {ns} --timeout=180s",
        f"microk8s kubectl get pods {ns}",
        f"microk8s kubectl exec {ns} deployment/miworkbench-backend -c backend -- python scripts/verify_schema.py",
    ], calls
    named = {w.split("/", 1)[1] for c in calls for w in c.split() if w.startswith("deployment/")}
    assert named <= deployments, named - deployments
    every = _bash("declare -f").lower()
    assert "deployment/mistudio-" not in every and "-n mistudio " not in every
    assert _bash('echo "$K8S_NS"').strip() == "miworkbench"
