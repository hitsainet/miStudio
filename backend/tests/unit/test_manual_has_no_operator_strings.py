"""The public manual names no operator host, private address or credential.

The manual publishes to GitHub Pages from the public mirror. On 2026-10-21 its
MCP page carried the reference deployment's private IP and its `*.hitsai.local`
/ `*.hitsai.net` MCP host names, and the install guides offered `*.hitsai.local`
as defaults. The pages now use placeholders (`mistudio.example.local`,
`<your-host>`, `<ingress-ip>`).

The patterns live in `manual/scripts/check_operator_strings.py`, which the
Deploy Manual workflow also runs over the built site; this test loads that file
so the two can never disagree.
"""

import importlib.util
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
MANUAL = REPO / "manual"
SCRIPT = MANUAL / "scripts" / "check_operator_strings.py"

#: The published sources: pages, theme code and static files, plus the config.
SCANNED_DIRS = ("docs", "src", "static")


def _checker():
    spec = importlib.util.spec_from_file_location("check_operator_strings", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


CHECK = _checker()


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
        "https://github.com/Onegaishimas/miStudio",
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
        "the public site at https://hitsai.net and https://hitsainet.github.io/miStudio/",
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


@pytest.mark.parametrize("host", ["mcp-studio", "mcp-dataworks", "mcp-forge"])
def test_published_mcp_endpoints_are_allowed(host):
    assert CHECK.scan_text(f"https://{host}.hitsai.net/mcp") == []
