"""T-33 (042 FTASKS 13.6, 13.7; FTDD §11.3; CF-5 M8, CF-14, R20): every host this app answers on, both directions.

* Forward: every new host, and every `transition` host record's old host, is served by a rendered Ingress, and
  each alias routes exactly what its twin routes. A record whose new name is a public tunnel name
  (`k8s-mistudio.hitsai.net` → `miworkbench.hitsai.net`, R20) has its Ingress twin `k8s-miworkbench.hitsai.local`
  (cloudflared rewrites Host to it; suite memory `public-proxy-rewrites-host-header`).
* Reverse: every old-brand host on a rendered Ingress rule, and every in-cluster form of an old-brand Service
  (the ExternalName's `mistudio-backend.mistudio[.svc[.cluster.local]]`), is a host record — otherwise P7 would
  retire it on evidence nobody collected (the host middleware records only `HOSTS_OLD`).
* `observed` is false exactly for an old host none of whose paths reaches the backend or MCP Service (CF-14), and
  both processes' `unobserved_hosts` is that set.
* R20: no `k8s-*.hitsai.net` host other than an old alias; no `k8s-miworkbench.hitsai.net` origin.
* `ALLOWED_ORIGINS` (a JSON array) holds http and https for every browser host.

The table is `src/core/legacy_names.HOSTS` (pinned equal to the export's host records by
`test_legacy_names_table.py`); the hosts come from the rendered manifests (`tests/support/k8s_render.py`).
"""
from __future__ import annotations

import json
import re

from src.core import legacy_names
from tests.support import k8s_render

OLD_BRAND = re.compile(r"(^|[.-])(mistudio|mcp-studio|millm)([.-]|$)")
#: R20: public names reach the cluster only through the tunnel, which sends this Ingress host instead.
TUNNEL_TWINS = {"miworkbench.hitsai.net": "k8s-miworkbench.hitsai.local"}
#: Public browser names (not Ingress hosts, R20): the old one and its new twin.
PUBLIC_BROWSER_HOSTS = ("miworkbench.hitsai.net",)  # the old mistudio.hitsai.net left at P7 (FTASKS 18.8)


def _docs():
    return k8s_render.render()


def routes(docs) -> dict[str, set[tuple[str, str, int]]]:
    """`{host: {(path, service, port)}}` over every rendered Ingress."""
    out: dict[str, set] = {}
    for d in docs:
        if d.get("kind") != "Ingress":
            continue
        for rule in d["spec"].get("rules") or []:
            for p in (rule.get("http") or {}).get("paths") or []:
                svc = p["backend"]["service"]
                out.setdefault(rule["host"], set()).add((p["path"], svc["name"], svc["port"]["number"]))
    return out


def _service_of(docs, app_suffix: str) -> str:
    """The Service selecting the Deployment whose name ends with `app_suffix` (`-backend`, `-mcp`, `-frontend`)."""
    deps = [d for d in docs if d.get("kind") == "Deployment" and d["metadata"]["name"].endswith(app_suffix)]
    assert len(deps) == 1, [d["metadata"]["name"] for d in deps]
    labels = k8s_render.pod_template(deps[0]).get("metadata", {}).get("labels") or {}
    svcs = [d["metadata"]["name"] for d in docs if d.get("kind") == "Service" and (d["spec"].get("selector") or {})
            and all(labels.get(k) == v for k, v in d["spec"]["selector"].items())]
    assert len(svcs) == 1, svcs
    return svcs[0]


def observed_mismatches(route_map, hosts, recording_services) -> list[str]:
    """CF-14: an old host is observed iff some rule path reaches a Service whose code records it."""
    out = []
    table = {h.old: h for h in hosts}
    for host, rules in sorted(route_map.items()):
        if not OLD_BRAND.search(host) or host not in table:
            continue
        reaches = any(svc in recording_services for _, svc, _ in rules)
        if table[host].observed != reaches:
            out.append(f"{host}: observed={table[host].observed} but reaches code that records it: {reaches}")
    return out


def test_every_new_host_and_alias_is_served():
    docs = _docs()
    served = routes(docs)
    problems = []
    for h in legacy_names.HOSTS:
        if h.status != "transition" or not h.old.endswith((".hitsai.local", ".hitsai.net")):
            continue
        twin = TUNNEL_TWINS.get(h.new, h.new)
        for host in (h.old, twin):
            if host not in served:
                problems.append(f"{host} is on no rendered Ingress")
        if h.old in served and twin in served:
            # The same Services, not the same paths: `k8s-mcp-studio.hitsai.local` sends `/` to the MCP pod, and
            # its twin `k8s-miworkbench.hitsai.local` reaches the MCP pod at `/mcp` (its `/` is the frontend).
            lost = {s for _, s, _ in served[h.old]} - {s for _, s, _ in served[twin]}
            if lost:
                problems.append(f"{h.old} reaches {sorted(lost)}, which its twin {twin} does not")
    assert not problems, "\n".join(problems)
    new_hosts = {TUNNEL_TWINS.get(h.new, h.new) for h in legacy_names.HOSTS if "." in h.new and h.new.endswith(
        (".hitsai.local", ".hitsai.net"))}
    assert {"k8s-miworkbench.hitsai.local", "mcp-workbench.hitsai.local", "mcp-workbench.hitsai.net"} <= new_hosts


def _incluster_forms(name: str, namespace: str) -> list[str]:
    return [f"{name}.{namespace}", f"{name}.{namespace}.svc", f"{name}.{namespace}.svc.cluster.local"]


def test_every_old_ingress_host_and_incluster_address_is_in_table():
    docs = _docs()
    # Since ENH-009 P7 no old host is served (FTASKS 18.8) and no old Service rendered (18.9), so both sets may be
    # empty; the scan itself must still see the Ingresses and the Services it reads.
    assert "k8s-miworkbench.hitsai.local" in routes(docs), "the reverse check reads no Ingress"
    assert any(d.get("kind") == "Service" for d in docs), "the reverse check reads no Service"
    old_hosts = {h for h in routes(docs) if OLD_BRAND.search(h)}
    old_services = [(d["metadata"]["name"], d["metadata"]["namespace"]) for d in docs if d.get("kind") == "Service"
                    and (OLD_BRAND.search(d["metadata"]["name"]) or OLD_BRAND.search(d["metadata"]["namespace"]))]
    wanted = old_hosts | {f for n, ns in old_services for f in _incluster_forms(n, ns)}
    recorded = {h.old for h in legacy_names.HOSTS}
    assert not sorted(wanted - recorded), f"served but not watched: {sorted(wanted - recorded)}"
    assert not sorted(wanted - legacy_names.HOSTS_OLD), sorted(wanted - legacy_names.HOSTS_OLD)


def test_observed_flag_matches_ingress_routing():
    docs = _docs()
    recording = {_service_of(docs, "-backend"), _service_of(docs, "-mcp")}
    problems = observed_mismatches(routes(docs), legacy_names.HOSTS, recording)
    assert not problems, "\n".join(problems)
    expected_unobserved = sorted(h for h, rules in routes(docs).items() if OLD_BRAND.search(h)
                                 and not any(svc in recording for _, svc, _ in rules))
    assert legacy_names.unobserved_hosts() == expected_unobserved
    assert legacy_names.meta()["unobserved_hosts"] == expected_unobserved
    # Self-test: an old host routed only to the frontend while flagged observed is a finding.
    fixture = {"old-mistudio.hitsai.local": {("/", "frontend", 80)}}
    flagged = [legacy_names.HostAlias(old="old-mistudio.hitsai.local", new="x", status="transition")]
    assert len(observed_mismatches(fixture, flagged, recording)) == 1


def test_no_public_k8s_net_host_or_origin():
    docs = _docs()
    bad = sorted(h for h in routes(docs) if re.fullmatch(r"k8s-[\w-]+\.hitsai\.net", h)
                 and h not in legacy_names.HOSTS_OLD)
    assert not bad, f"R20: no public k8s-*.hitsai.net name exists; served: {bad}"
    assert not [o for o in _origins(docs) if "k8s-miworkbench.hitsai.net" in o]


def _origins(docs) -> list[str]:
    values = [v for d in docs if d.get("kind") == "Deployment" for _, name, v in k8s_render.env_values(d)
              if name == "ALLOWED_ORIGINS"]
    assert len(values) == 1, values
    origins = json.loads(values[0])  # a JSON array: pydantic-settings decodes it before any validator runs
    assert isinstance(origins, list), origins
    return origins


def test_allowed_origins_cover_browser_hosts():
    docs = _docs()
    frontend = _service_of(docs, "-frontend")
    browser = {h for h, rules in routes(docs).items() if h.endswith(".hitsai.local")
               and any(path == "/" and svc == frontend for path, svc, _ in rules)}
    assert "k8s-miworkbench.hitsai.local" in browser, browser
    origins = set(_origins(docs))
    missing = sorted(f"{scheme}://{h}" for h in [*browser, *PUBLIC_BROWSER_HOSTS] for scheme in ("http", "https")
                     if f"{scheme}://{h}" not in origins)
    assert not missing, missing
