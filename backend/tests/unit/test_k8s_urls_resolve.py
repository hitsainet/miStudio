"""T-40 (042 FTASKS 13.8; FTDD §11.3 "In-cluster addresses", §17 C12): every in-cluster URL resolves.

A short Service name (`http://mistudio-backend:8000`) resolves only inside its own namespace, so after the move
every literal URL in a rendered env value must name a rendered Service in the workload's namespace (short form)
or in the namespace it spells (`name.ns[.svc[.cluster.local]]`), on a port that Service exposes — or one of the
sibling apps' addresses listed below. A missed one leaves the workers unable to emit progress, or the MCP pod
unable to reach the API.
"""
from __future__ import annotations

import re

from tests.support import k8s_render

URL = re.compile(r"^(?:https?|redis)://([a-z0-9.-]+)(?::(\d+))?(?:/|$)")
#: The siblings' Services this app calls (their own manifests; miServer's is checked against its repository by
#: `test_p2_consumer_address.py`).
SIBLINGS = {("miserver-backend", "miserver", 8000), ("midataworks-backend", "midataworks", 8000),
            ("miforge-backend", "miforge", 8000)}
EXTERNAL = re.compile(r"(^localhost$|\.hitsai\.(local|net)$|\.(com|org|io|net)$)")


def findings(docs) -> list[str]:
    services = {(d["metadata"]["name"], d["metadata"]["namespace"]): {p["port"] for p in d["spec"].get("ports") or []}
                for d in docs if d.get("kind") == "Service"}
    out = []
    checked = 0
    for d in docs:
        ns = d["metadata"].get("namespace")
        for container, name, value in k8s_render.env_values(d):
            for candidate in re.findall(r"[a-z]+://[^\s\"',\]]+", value):
                m = URL.match(candidate)
                if not m or EXTERNAL.search(m.group(1)):
                    continue
                host, port = m.group(1), int(m.group(2) or 80)
                parts = host.split(".")
                svc, svc_ns = (parts[0], ns) if len(parts) == 1 else (parts[0], parts[1])
                checked += 1
                if (svc, svc_ns, port) in SIBLINGS:
                    continue
                if (svc, svc_ns) not in services:
                    out.append(f"{d['metadata']['name']}/{container} {name}={value}: no Service {svc} in {svc_ns}")
                elif port not in services[(svc, svc_ns)]:
                    out.append(f"{d['metadata']['name']}/{container} {name}={value}: {svc} has no port {port}")
    if not checked:
        out.append("no in-cluster URL was checked: the scan is blind")
    return out


def test_every_incluster_url_names_a_rendered_service():
    problems = findings(k8s_render.render())
    assert not problems, "\n".join(problems)
    stale = [{"kind": "Deployment", "metadata": {"name": "w", "namespace": "miworkbench"},
              "spec": {"template": {"spec": {"containers": [{"name": "c", "env": [
                  {"name": "INTERNAL_API_URL", "value": "http://mistudio-backend:8000"}]}]}}}}]
    assert findings(stale) == ["w/c INTERNAL_API_URL=http://mistudio-backend:8000: no Service mistudio-backend in "
                               "miworkbench"]
