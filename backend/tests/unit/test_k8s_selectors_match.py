"""T-41 (042 FTASKS 13.8; FTDD §11.3 "Selectors"): every selector selects something.

Each Service selector, each workload's own `selector.matchLabels`, and each NetworkPolicy's `podSelector` and
`from`/`to` pod selectors must match the labels of a rendered pod template in the same namespace. A renamed `app:`
label with a stale selector leaves a Service with no endpoints, or (the rule runner's NetworkPolicy) every
backend call to the runner refused.
"""
from __future__ import annotations

from tests.support import k8s_render


def _templates(docs):
    return [(d["metadata"]["namespace"], d["metadata"]["name"], k8s_render.pod_template(d).get("metadata", {})
             .get("labels") or {}) for d in docs if k8s_render.pod_specs(d)]


def _matches(selector: dict, labels: dict) -> bool:
    return bool(selector) and all(labels.get(k) == v for k, v in selector.items())


def findings(docs) -> list[str]:
    templates = _templates(docs)
    out = []
    selectors = []
    for d in docs:
        kind, ns, name = k8s_render.key(d)
        spec = d.get("spec") or {}
        if kind == "Service" and spec.get("type") != "ExternalName":
            selectors.append((f"Service {name}", ns, spec.get("selector") or {}))
        if kind in ("Deployment", "StatefulSet", "DaemonSet"):
            own = spec["selector"]["matchLabels"]
            if not _matches(own, k8s_render.pod_template(d)["metadata"]["labels"]):
                out.append(f"{kind} {name}: selector {own} does not match its own template")
        if kind == "NetworkPolicy":
            selectors.append((f"NetworkPolicy {name} podSelector", ns, spec["podSelector"].get("matchLabels") or {}))
            for direction in ("ingress", "egress"):
                for rule in spec.get(direction) or []:
                    for peer in rule.get("from") or rule.get("to") or []:
                        if "podSelector" in peer and "namespaceSelector" not in peer:
                            selectors.append((f"NetworkPolicy {name} {direction} peer", ns,
                                              peer["podSelector"].get("matchLabels") or {}))
    if not selectors:
        out.append("no selector found: the scan is blind")
    for what, ns, sel in selectors:
        if not any(t_ns == ns and _matches(sel, labels) for t_ns, _, labels in templates):
            out.append(f"{what} ({ns}): {sel} matches no pod template")
    return out


def test_every_selector_matches_a_pod_template():
    problems = findings(k8s_render.render())
    assert not problems, "\n".join(problems)
