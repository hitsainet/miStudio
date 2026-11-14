"""The rule runner's installed package is `miworkbench_rule_runner` (042 PD-S2; FTDD §11.4; FTASKS 12.16 – 12.17,
control K-12.40).

The runner image installs `backend/src/rule_runner/` as a top-level package; its name appears in the Dockerfile
(the COPY target, the build-time fetch, the CMD), in CI's runner job (the import-root symlink and the fetch), in
the runner's own tests and the backend's runner harness, and as the server's logger name. All of them move
together, or the image starts a module that is not there. The package name enters no digest:
`contract.bundle_digest` hashes the bundle mapping only, pinned here on a fixed bundle.
"""
from __future__ import annotations

import ast
import re
import subprocess
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[3]
NEW, OLD = "miworkbench_rule_runner", "_".join(("mistudio", "rule", "runner"))


def _instructions() -> list[str]:
    text = (REPO / "rule_runner" / "Dockerfile").read_text()
    joined = re.sub(r"\\\n\s*", " ", text)
    return [ln.strip() for ln in joined.splitlines() if ln.strip() and not ln.lstrip().startswith("#")]


def test_dockerfile_workflow_and_imports_use_new_package():
    instructions = _instructions()
    assert f"COPY backend/src/rule_runner/ /app/{NEW}/" in instructions
    assert any(i.startswith("RUN") and f"python -m {NEW}.fetch" in i for i in instructions)
    assert any(i.startswith("CMD") and f"{NEW}.server" in i for i in instructions)

    job = yaml.safe_load((REPO / ".github" / "workflows" / "backend-tests.yml").read_text())["jobs"]
    runs = "\n".join(step.get("run", "") for j in job.values() for step in j.get("steps", []))
    assert f'/import/{NEW}"' in runs and f"-m {NEW}.fetch" in runs

    # Every runner test and the backend harness import (or launch) the new name; none the old.
    files = [*sorted((REPO / "rule_runner" / "tests").glob("*.py")), REPO / "backend" / "tests" / "support" /
             "real_rule_runner.py"]
    importing = 0
    for path in files:
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module and node.module.split(".")[0] in (NEW, OLD):
                assert node.module.split(".")[0] == NEW, f"{path.name}:{node.lineno} imports {node.module}"
                importing += 1
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.split(".")[0] in (NEW, OLD):
                        assert alias.name.split(".")[0] == NEW, f"{path.name}:{node.lineno} imports {alias.name}"
                        importing += 1
    assert importing >= 10, f"only {importing} imports of the runner package found: the walk is blind"

    tracked = subprocess.run(["git", "grep", "-l", OLD, "--", ".", ":!0xcc", ":!.claude",
                              ":(exclude,glob)**/docs/legacy-identifiers*.md"],
                             cwd=REPO, capture_output=True, text=True).stdout.split()
    assert tracked == [], f"the old package name is still named in: {tracked}"

    # The server module imports wasmtime (the runner image's, not this venv's), so its logger is read from the
    # module's AST: the one `log = logging.getLogger(<name>)` assignment.
    tree = ast.parse((REPO / "backend" / "src" / "rule_runner" / "server.py").read_text())
    names = [node.value.args[0].value for node in ast.walk(tree)
             if isinstance(node, ast.Assign) and [getattr(t, "id", None) for t in node.targets] == ["log"]
             and isinstance(node.value, ast.Call) and getattr(node.value.func, "attr", None) == "getLogger"]
    assert names == ["miworkbench.rule_runner"]


def test_bundle_digest_unchanged_for_fixed_bundle():
    from src.rule_runner import contract

    bundle = {"format": "midataworks.runner-bundle/v1", "guest": {"sha256": "a" * 64, "url": "https://x/g.wasm"},
              "limits": {"fuel": 1000, "memory_bytes": 65536}, "bundle_digest": "ignored"}
    assert contract.bundle_digest(bundle) == PINNED_DIGEST


#: Read on the pre-rename tree (2026-11-11) and unchanged by it: the package name enters no digest.
PINNED_DIGEST = "686c9bfd0cdcac5e5196fc1add556089f99c78eb609260688ebe0b67524ebccd"
