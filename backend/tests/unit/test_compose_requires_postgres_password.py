"""Compose requires the database password; no file defaults it (042 FR-042.40; R18 T5, TI-18; FTASKS 5.8,
control K-5.16).

Every `docker-compose*.yml` used `${POSTGRES_PASSWORD:-<the workstation password>}`, so a stack started
without configuration ran on a password committed to the repository (and, before MIS-E2E-146, published
on 0.0.0.0). Compose now refuses to start without `POSTGRES_PASSWORD` (`${POSTGRES_PASSWORD:?…}`) and
`.env.example` carries a placeholder.

Read from the PARSED YAML: every string value under every service's `environment` (mapping or list form)
is checked, so a new service or a moved line cannot slip past a line-based search.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[3]
KNOWN = ("dev" + "password", "test" + "password")
VAR = re.compile(r"\$\{(?P<name>[A-Z0-9_]+)(?P<op>:?[-?])?(?P<rest>[^}]*)\}")
URL_USERINFO = re.compile(r"://[^\s/:@]+:(?P<pw>[^\s/@]+)@")

#: (file, service, key) -> reason. A whole VALUE named exactly, never a file. Not 042's: the local
#: Neuronpedia database has its own password, outside FR-042.40/TI-18 (recorded as an open question).
EXEMPT_VALUES: dict[tuple[str, str, str], str] = {
    ("docker-compose.yml", "neuronpedia-postgres", "POSTGRES_PASSWORD"): "Neuronpedia DB (open question Q-C1)",
    ("docker-compose.dev.yml", "neuronpedia-postgres", "POSTGRES_PASSWORD"): "Neuronpedia DB (open question Q-C1)",
    ("docker-compose.dev.yml", "neuronpedia", "POSTGRES_PRISMA_URL"): "Neuronpedia DB (open question Q-C1)",
    ("docker-compose.dev.yml", "neuronpedia", "POSTGRES_URL_NON_POOLING"): "Neuronpedia DB (open question Q-C1)",
    ("docker-compose.dev.yml", "neuronpedia", "DATABASE_URL"): "Neuronpedia DB (open question Q-C1)",
}
EXEMPT_VARS = {"NEURONPEDIA_POSTGRES_PASSWORD"}  # the same open question, wherever the variable is used


def _compose_files() -> list[Path]:
    return sorted(REPO.glob("docker-compose*.yml"))


def _env_items(service: dict) -> list[tuple[str, str]]:
    env = service.get("environment") or {}
    if isinstance(env, dict):
        return [(str(k), "" if v is None else str(v)) for k, v in env.items()]
    out = []
    for item in env:
        key, _, value = str(item).partition("=")
        out.append((key, value))
    return out


def findings_in(text: str, name: str) -> list[str]:
    doc = yaml.safe_load(text) or {}
    out = []
    for svc_name, svc in (doc.get("services") or {}).items():
        for key, value in _env_items(svc or {}):
            where = f"{name} {svc_name}.{key}"
            if (name, svc_name, key) in EXEMPT_VALUES:
                continue
            for word in KNOWN:
                if word in value:
                    out.append(f"{where}: carries a known credential")
            for m in VAR.finditer(value):
                if "PASSWORD" in m["name"] and m["name"] not in EXEMPT_VARS and m["op"] in ("-", ":-"):
                    out.append(f"{where}: ${{{m['name']}}} has a default ({m['op']}{m['rest']})")
            literal = VAR.sub("", value)
            if "PASSWORD" in key.upper() and key.upper() != "POSTGRES_PASSWORD_FILE" and literal.strip():
                out.append(f"{where}: a literal password")
            for m in URL_USERINFO.finditer(VAR.sub("VAR", value)):
                if m["pw"] != "VAR":
                    out.append(f"{where}: a URL with a literal password")
    return out


def test_no_password_default_in_compose():
    files = _compose_files()
    assert {p.name for p in files} >= {"docker-compose.yml", "docker-compose.dev.yml"}, files
    problems, required = [], 0
    for path in files:
        text = path.read_text(encoding="utf-8")
        problems += findings_in(text, path.name)
        required += len(re.findall(r"\$\{POSTGRES_PASSWORD:\?", text))
    assert not problems, "\n".join(problems)
    # Every file that starts or connects to the app's database requires the variable.
    for path in files:
        doc = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if "postgres" in (doc.get("services") or {}):
            pw = dict(_env_items(doc["services"]["postgres"])).get("POSTGRES_PASSWORD", "")
            assert pw.startswith("${POSTGRES_PASSWORD:?"), f"{path.name} postgres.POSTGRES_PASSWORD = {pw!r}"
    assert required >= 3, f"only {required} `${{POSTGRES_PASSWORD:?…}}` references found"
    for name in EXEMPT_VALUES:
        doc = yaml.safe_load((REPO / name[0]).read_text(encoding="utf-8"))
        assert name[2] in dict(_env_items(doc["services"][name[1]])), f"stale exemption {name}"


def test_env_example_holds_a_placeholder():
    text = (REPO / ".env.example").read_text(encoding="utf-8")
    assert re.search(r"^POSTGRES_PASSWORD=change-me$", text, re.M), "set POSTGRES_PASSWORD=change-me"
    assert not any(w in text for w in KNOWN)


def _j(*parts: str) -> str:
    return "".join(parts)  # planted samples are built at runtime, never literals of this file


@pytest.mark.parametrize("value", [
    _j("${POSTGRES_PASSWORD:-", KNOWN[0], "}"),
    "${POSTGRES_PASSWORD-x}",
    "s3cr3t",
    _j("postgresql://postgres:", "s3cr3t", "@postgres:5432/x"),
])
def test_the_check_finds_each_form(value):
    text = yaml.safe_dump({"services": {"svc": {"environment": {"POSTGRES_PASSWORD": value}}}})
    assert findings_in(text, "planted.yml"), value


def test_a_required_variable_passes():
    text = yaml.safe_dump({"services": {"svc": {"environment": [
        "POSTGRES_PASSWORD=${POSTGRES_PASSWORD:?set it}",
        "DATABASE_URL=postgresql://${POSTGRES_USER:-postgres}:${POSTGRES_PASSWORD:?set it}@postgres/x"]}}})
    assert findings_in(text, "ok.yml") == []
