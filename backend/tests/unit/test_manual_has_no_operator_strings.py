"""The public manual names no operator host, private address or credential.

The manual publishes to GitHub Pages from the public mirror. On 2026-10-21 its
MCP page carried the reference deployment's private IP and its `*.hitsai.local`
/ `*.hitsai.net` MCP host names, and the install guides offered `*.hitsai.local`
as defaults. The pages now use placeholders (`mistudio.example.local`,
`<your-host>`, `<ingress-ip>`).

The patterns live in `manual/publish-rules.json`, read by
`manual/scripts/check_operator_strings.py`, which the Deploy Manual workflow
also runs over the built site; this test loads that script so the two can never
disagree. The suite census reads the same JSON when it writes this repository's
public legacy-identifiers export (miSuite ENH-009 0.53, 2026-11-08).
"""

import importlib.util
import json
import re
import shutil
from pathlib import Path

import pytest

from tests.support import legacy_export

REPO = Path(__file__).resolve().parents[3]
MANUAL = REPO / "manual"
SCRIPT = MANUAL / "scripts" / "check_operator_strings.py"
RULES = MANUAL / "publish-rules.json"

#: The published sources: pages, theme code and static files, plus the config.
SCANNED_DIRS = ("docs", "src", "static")


def _checker(script: Path = SCRIPT):
    spec = importlib.util.spec_from_file_location("check_operator_strings", script)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


CHECK = _checker()

PUBLIC_OWNER = "hitsainet"


def _private_repo_plants() -> list:
    """042 FTASKS 6.12 / TPD-27: the private repository names come from the export's `repo` records (old
    and new spelling), never from a literal here. The public export withholds them (CF-2), so on the
    mirror the case skips BY NAME; in the private repository it always runs."""
    export = legacy_export.load()
    names = sorted({n for r in export.repos() for n in (r["old"], r["new"])
                    if not n.lower().startswith(PUBLIC_OWNER + "/")})
    if names:
        return [pytest.param(f"https://github.com/{n}", id=f"private-repo-{i}") for i, n in enumerate(names)]
    held = [w for w in export.withheld if w.startswith("repo:")]
    if export.is_public and held:
        return [pytest.param(None, id="private-repo-withheld",
                             marks=pytest.mark.skip(reason=f"withheld from the public export: {held}"))]
    return [pytest.param("", id="private-repo-missing")]  # fails below: the export must list them


def test_the_published_manual_sources_are_clean():
    findings, scanned = [], 0
    for name in SCANNED_DIRS:
        count, hits = CHECK.scan_tree(MANUAL / name)
        scanned += count
        findings += hits
    for cfg in ("docusaurus.config.ts", "sidebars.ts"):
        for lineno, pname, match in CHECK.scan_text((MANUAL / cfg).read_text()):
            findings.append(f"{cfg}:{lineno}: {pname}: {match}")
    # The walk must actually reach the pages; an empty scan is never "clean".
    assert scanned > 40, f"only {scanned} files scanned under manual/ — the walk is broken"
    assert not findings, (
        "operator-specific strings in the public manual; use placeholders such "
        "as mistudio.example.local, <your-host>, <ingress-ip>:\n" + "\n".join(findings)
    )


def test_the_walk_reaches_the_mcp_page():
    """The page this guard was written for is inside the scanned set."""
    files = set(CHECK.iter_files(MANUAL / "docs"))
    assert MANUAL / "docs" / "advanced" / "mcp-server.md" in files
    assert MANUAL / "docs" / "getting-started" / "install-guide-k8s.md" in files


@pytest.mark.parametrize(
    "line",
    [
        "Point a client at it: `192.168.244.61  mcp.example`",
        "echo '192.168.x.x  host' >> /etc/hosts",
        "pod IP 10.152.183.20",
        "http://k8s-mistudio.hitsai.local/mcp",
        "https://unapproved.hitsai.net/mcp",
        "https://private.mcp-studio.hitsai.net/mcp",
        "https://other-mcp-forge.hitsai.net/mcp",
        "ssh sean@mcs-lnxhost02",
        "HF_TOKEN=hf_" + "a" * 34,
        "OPENAI_API_KEY=sk-" + "b" * 40,
        "token ghp_" + "c" * 36,
        "AKIA" + "ABCDEFGHIJKLMNOP",
        "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.sig",
        "--header 'Authorization: Bearer " + "d" * 64 + "'",
    ],
)
def test_a_planted_line_is_caught(line, tmp_path):
    """Negative control: each pattern fails a planted page, through the walker."""
    page = tmp_path / "docs" / "planted.md"
    page.parent.mkdir()
    page.write_text("# Clean heading\n\n" + line + "\n")
    count, hits = CHECK.scan_tree(tmp_path / "docs")
    assert count == 1
    assert len(hits) == 1 and ":3:" in hits[0], hits


@pytest.mark.parametrize(
    "line",
    [
        "https://mcp.mistudio.example.local/mcp",
        "echo '<ingress-ip>  mistudio.example.local' >> /etc/hosts",
        '--header "Authorization: Bearer $MCP_AUTH_TOKEN"',
        "header `Authorization: Bearer <token>`",
        "the public site at https://hitsai.net and https://docs-miworkbench.hitsai.net/",
        "https://github.com/hitsainet/miStudio",
    ],
)
def test_placeholders_are_allowed(line):
    assert CHECK.scan_text(line) == []


def test_the_build_check_refuses_an_empty_directory(tmp_path, capsys):
    """CI runs the script on manual/build; a missing build must not pass."""
    assert CHECK.main([str(tmp_path)]) == 2


def test_the_deploy_workflow_runs_the_check_on_the_built_site():
    """The built-site half of the guard is wired: before upload, on manual/build."""
    import yaml

    wf = yaml.safe_load((REPO / ".github" / "workflows" / "deploy-manual.yml").read_text())
    steps = wf["jobs"]["build"]["steps"]
    names = [s.get("run", "") + " " + s.get("uses", "") for s in steps]
    check = [i for i, s in enumerate(names) if "check_operator_strings.py" in s]
    upload = [i for i, s in enumerate(names) if "upload-pages-artifact" in s]
    assert len(check) == 1, "deploy-manual.yml does not run check_operator_strings.py"
    assert "build" in steps[check[0]]["run"], "the check must scan the built site"
    assert upload and check[0] < upload[0], "the check must run before the site is uploaded"


@pytest.mark.parametrize("host", ["mcp-workbench", "mcp-dataworks", "mcp-forge"])
def test_published_mcp_endpoints_are_allowed(host):
    assert CHECK.scan_text(f"https://{host}.hitsai.net/mcp") == []


def test_the_old_mcp_endpoint_is_no_longer_published():
    """ENH-009 P7 (FTASKS 18.8): `mcp-studio.hitsai.net` left the allowance with the Ingress rule."""
    assert CHECK.scan_text("https://mcp-studio.hitsai.net/mcp")


@pytest.mark.parametrize("url", _private_repo_plants())
def test_a_private_repository_link_is_caught(url, tmp_path):
    """The private owner, in either repository spelling (miStudio before P1, miWorkbench after)."""
    assert url, "the export lists no private repository record; it must (CF-13)"
    page = tmp_path / "docs" / "planted.md"
    page.parent.mkdir()
    page.write_text("# Clean heading\n\n" + url + "\n")
    count, hits = CHECK.scan_tree(tmp_path / "docs")
    assert count == 1 and len(hits) == 1 and "private repository owner" in hits[0], hits


# ── 042 FR-042.2 (FTASKS 1.7): the public build may name its own docs domain, and only that one ─────────
def _built_page(tmp_path: Path, body: str) -> tuple[int, list[str]]:
    page = tmp_path / "build" / "advanced" / "x" / "index.html"
    page.parent.mkdir(parents=True)
    page.write_text(f"<html><head><link rel=\"canonical\" href=\"{body}\"></head><body>{body}</body></html>\n")
    return CHECK.scan_tree(tmp_path / "build")


def test_public_build_may_name_its_own_docs_domain(tmp_path):
    count, hits = _built_page(tmp_path, "https://docs-miworkbench.hitsai.net/x")
    assert count == 1 and hits == [], hits
    assert CHECK.scan_text("see docs-miworkbench.hitsai.net/advanced/approvals#top") == []


@pytest.mark.parametrize("host", [
    "docs-other.hitsai.net",            # another docs subdomain
    "docs-miworkbenchx.hitsai.net",     # a lookalike that extends the label
    "x.docs-miworkbench.hitsai.net",    # a subdomain of the allowed host
    "docs-miserver.hitsai.net",         # a sibling's docs name is not this manual's
    "docs.hitsai.net",
    "mcp-workbenchx.hitsai.net",        # 042 FTASKS 15.9: the P4 name is allowed as the WHOLE host only
    "x.mcp-workbench.hitsai.net",
    "mcp-miworkbench.hitsai.net",       # not a published endpoint
])
def test_other_docs_subdomain_is_refused(tmp_path, host):
    count, hits = _built_page(tmp_path, f"https://{host}/x")
    assert count == 1 and len(hits) == 1, hits  # one finding per line (scan_text reports a line once)
    assert all("operator host *.hitsai.net" in h for h in hits), hits


@pytest.mark.parametrize("text", [
    "docs-miworkbench.hitsai.net.local",
    "https://mcp-studio.hitsai.net.attacker.example/x",
    "mcp-forge.hitsai.net.internal",
    "docs-miworkbench.hitsai.net-x.example",
    "https://mcp-workbench.hitsai.net.attacker.example/x",
])
def test_an_allowed_host_with_a_suffix_is_refused(text):
    """Review rev-mistudio-p0 F4 (the miForge `\\b` lookahead finding): an allowed host is allowed only as the
    WHOLE host; a longer name that merely starts with it is an operator host like any other."""
    assert [h[1] for h in CHECK.scan_text(text)] == ["operator host *.hitsai.net"], CHECK.scan_text(text)


def test_an_allowed_host_ending_a_sentence_is_allowed():
    assert CHECK.scan_text("The manual lives at docs-miworkbench.hitsai.net.") == []


# ── miSuite ENH-009 0.53 (operator decision 2026-11-08): the rules are ONE JSON file ─────────────────────
#: How a rule's `flags` read in Python; the checker and the suite census both apply exactly these.
_FLAG_BITS = {"i": re.IGNORECASE, "m": re.MULTILINE, "s": re.DOTALL}


def _json_rules(path: Path) -> list[tuple[str, str, int]]:
    """(name, pattern, flag bits) read from the JSON directly, NOT through the checker."""
    out = []
    for rule in json.loads(path.read_text(encoding="utf-8"))["forbidden"]:
        bits = 0
        for c in rule.get("flags") or "":
            bits |= _FLAG_BITS[c]
        out.append((rule["name"], rule["pattern"], bits))
    return out


def _applied(checker) -> list[tuple[str, str, int]]:
    """What the live checker applies: its FORBIDDEN, the tuple scan_text() walks."""
    return [(name, rx.pattern, rx.flags & (re.IGNORECASE | re.MULTILINE | re.DOTALL)) for name, rx in checker.FORBIDDEN]


def test_the_checkers_rules_are_the_json_files_rules():
    """Every rule the live checker applies, and only those, in order, with the file's flags: census reads the same
    file for the public export, so a checker that drifted from it would let the two lists disagree."""
    expected = _json_rules(RULES)
    assert len(expected) >= 12 and len({n for n, _, _ in expected}) == len(expected), expected
    assert _applied(CHECK) == expected


def _checker_beside(tmp_path: Path, rules) -> Path:
    """A copy of the checker in a scratch `manual/scripts/`, with `rules` as the `manual/publish-rules.json` beside it
    (`None`: no file)."""
    scripts = tmp_path / "manual" / "scripts"
    scripts.mkdir(parents=True)
    shutil.copy(SCRIPT, scripts / SCRIPT.name)
    if rules is not None:
        (tmp_path / "manual" / "publish-rules.json").write_text(rules if isinstance(rules, str) else json.dumps(rules))
    return scripts / SCRIPT.name


def test_the_checker_reads_the_rules_file_beside_it(tmp_path):
    """The wiring, not a copy that happens to agree: a different file beside the script is what the script applies."""
    script = _checker_beside(tmp_path, {"forbidden": [{"name": "planted", "pattern": r"zebra-\d+", "flags": "i"}]})
    planted = _checker(script)
    assert _applied(planted) == [("planted", r"zebra-\d+", re.IGNORECASE)]
    assert planted.scan_text("ok\nsee ZEBRA-7 here") == [(2, "planted", "ZEBRA-7")]
    assert planted.scan_text("192.168.1.1") == []          # the repo's rules are not baked in


@pytest.mark.parametrize("rules,error,match", [
    (None, FileNotFoundError, "publish-rules.json"),
    ('{"forbidden": []}', ValueError, "the forbidden list is empty"),
    ("{", ValueError, "Expecting"),
    ('{"rules": []}', KeyError, "forbidden"),
    ({"forbidden": [{"name": "x", "pattern": "("}]}, re.error, "missing"),
    ({"forbidden": [{"pattern": "abc"}]}, KeyError, "name"),
    ({"forbidden": [{"name": "x", "pattern": "abc", "flags": "g"}]}, ValueError, "unknown flag"),
    ({"forbidden": [{"name": "x", "pattern": "abc", "flags": "ig"}]}, ValueError, "unknown flag"),
])
def test_a_missing_or_broken_rules_file_stops_the_check(tmp_path, rules, error, match):
    """The check never runs with a partial or empty list: loading the checker fails instead, for the stated reason
    (an unknown flag is refused by name, not by an accident further down)."""
    script = _checker_beside(tmp_path, rules)
    with pytest.raises(error, match=match):
        _checker(script)
