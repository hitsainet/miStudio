"""One rendering path for the manifest tests (042 FTID §11.2, TI-20, TPD-18).

`render()` returns the objects ArgoCD deploys from `k8s/base`: `kustomize build` (the `kustomize` binary, else
`kubectl kustomize`) when one is on `PATH`, else the `resources:` list of `kustomization.yaml` read with PyYAML,
recursing into directories, with the `images:` transform applied. Both paths are tested once
(`test_k8s_render.py`): their object sets and images must agree, so a test that runs on a machine without the
tool reads what the cluster would get.

Every manifest test of the family (T-31 – T-35, T-40, T-41, the DR-28/DR-29 exercises) reads this, so a new
`resources:` entry is seen by all of them at once.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import yaml

from tests.support.legacy_export import REPO_ROOT

BASE = REPO_ROOT / "k8s" / "base"
WORKLOADS = ("Deployment", "StatefulSet", "DaemonSet", "Job", "CronJob")


def tool() -> list[str] | None:
    """The command that builds a kustomization, or None when neither tool is installed."""
    if shutil.which("kustomize"):
        return ["kustomize", "build"]
    if shutil.which("kubectl"):
        return ["kubectl", "kustomize"]
    return None


def _docs(text: str) -> list[dict]:
    return [d for d in yaml.safe_load_all(text) if isinstance(d, dict)]


def render_with_tool(base: Path = BASE) -> list[dict]:
    cmd = tool()
    if cmd is None:
        raise RuntimeError("neither kustomize nor kubectl is on PATH")
    done = subprocess.run([*cmd, str(base)], capture_output=True, text=True, timeout=120)
    if done.returncode != 0:
        raise AssertionError(f"{' '.join(cmd)} {base} failed: {done.stderr.strip()}")
    return _docs(done.stdout)


def _set_image(container: dict, images: list[dict]) -> None:
    ref = container.get("image") or ""
    name, sep, tag = ref.rpartition(":") if ":" in ref.split("/")[-1] else (ref, "", "")
    for rule in images:
        if rule.get("name") == name:
            new_name = rule.get("newName", name)
            if "digest" in rule:
                container["image"] = f"{new_name}@{rule['digest']}"
            else:
                new_tag = rule.get("newTag", tag)
                container["image"] = f"{new_name}:{new_tag}" if new_tag else new_name


def _pod_specs(doc: dict) -> list[dict]:
    spec = doc.get("spec") or {}
    if doc.get("kind") == "CronJob":
        spec = (spec.get("jobTemplate") or {}).get("spec") or {}
    pod = (spec.get("template") or {}).get("spec")
    return [pod] if pod else []


def render_from_resource_list(base: Path = BASE) -> list[dict]:
    """The fallback: the files named by `resources:` (recursively), with `images:` applied."""
    kust = yaml.safe_load((base / "kustomization.yaml").read_text()) or {}
    out: list[dict] = []
    for entry in kust.get("resources") or []:
        target = (base / entry).resolve()
        if target.is_dir():
            out.extend(render_from_resource_list(target))
        else:
            out.extend(_docs(target.read_text()))
    images = kust.get("images") or []
    for doc in out:
        if doc.get("kind") in WORKLOADS:
            for pod in _pod_specs(doc):
                for c in [*(pod.get("containers") or []), *(pod.get("initContainers") or [])]:
                    _set_image(c, images)
    if kust.get("namespace"):
        for doc in out:
            if doc.get("kind") not in ("Namespace", "PersistentVolume", "ClusterRole", "ClusterRoleBinding"):
                doc.setdefault("metadata", {})["namespace"] = kust["namespace"]
    return out


def render(base: Path = BASE) -> list[dict]:
    return render_with_tool(base) if tool() else render_from_resource_list(base)


def key(doc: dict) -> tuple[str, str, str]:
    meta = doc.get("metadata") or {}
    return doc.get("kind", ""), meta.get("namespace", ""), meta.get("name", "")


def pod_specs(doc: dict) -> list[dict]:
    """The pod spec(s) of a workload (a CronJob's job template included)."""
    return _pod_specs(doc) if doc.get("kind") in WORKLOADS else []


def pod_template(doc: dict) -> dict:
    spec = doc.get("spec") or {}
    if doc.get("kind") == "CronJob":
        spec = (spec.get("jobTemplate") or {}).get("spec") or {}
    return spec.get("template") or {}


def containers(doc: dict) -> list[dict]:
    return [c for pod in pod_specs(doc) for c in [*(pod.get("containers") or []), *(pod.get("initContainers") or [])]]


def env_values(doc: dict) -> list[tuple[str, str, str]]:
    """`(container, name, value)` for every literal env value of a workload."""
    return [(c["name"], e["name"], str(e["value"])) for c in containers(doc) for e in c.get("env") or []
            if "value" in e]
