"""The runner image is built, from what, and verified by digest (Feature 040 S4; FTDD Appendix C.4, C.9, C.10;
FTASKS 25.4).

Read from the workflow and the Dockerfile as data (YAML, Dockerfile instructions), not as prose:

* `docker-images.yml` builds `rule-runner` (in `ALL`, with its repository, its paths and its build case: the repository
  root as context, `rule_runner/Dockerfile`), so a change to the runner package or the vendored contract rebuilds it;
* the Dockerfile copies ONLY the runner package (as `mistudio_rule_runner`) and the vendored contract, runs the
  digest-checking fetch, and carries NO literal 64-hex digest (the digest comes from the contract);
* `.wasm` / `.cwasm` never enter the build context;
* `backend-tests.yml` runs `rule-runner-tests` with `MISTUDIO_REQUIRE_RUNNER=1`, the image's own pins, the fetch, the
  host tests and the backend's real-runner tests, and its path filter includes `rule_runner/**`.

Controls (040 controls record, S4): drop the fetch from the Dockerfile → red; add a literal digest → red; drop
`rule-runner` from `ALL` → red; drop `MISTUDIO_REQUIRE_RUNNER` from the job → red.
"""
from __future__ import annotations

import re
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[3]
WORKFLOWS = REPO / ".github" / "workflows"
DOCKERFILE = REPO / "rule_runner" / "Dockerfile"
HEX64 = re.compile(r"\b[0-9a-f]{64}\b")


def _instructions() -> list:
    lines, current = [], ""
    for raw in DOCKERFILE.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        current = (current + " " + line[:-1].strip()) if line.endswith("\\") else (current + " " + line).strip()
        if not line.endswith("\\"):
            lines.append(current.strip())
            current = ""
    return lines


def _detect_script() -> str:
    doc = yaml.safe_load((WORKFLOWS / "docker-images.yml").read_text())
    return next(step["run"] for step in doc["jobs"]["detect"]["steps"] if step.get("id") == "detect")


def test_the_image_is_in_the_build_set_with_its_paths_and_build_case():
    script = _detect_script()
    all_line = next(l for l in script.splitlines() if l.strip().startswith("ALL=("))
    assert "rule-runner" in all_line.split("(")[1].split(")")[0].split()
    assert 'rule-runner) echo "hitsai/mistudio-rule-runner"' in script
    paths = re.search(r"rule-runner\) echo '([^']+)'", script).group(1)
    for changed in ("backend/src/rule_runner/server.py", "backend/src/schemas/vendored/midataworks-runner-v1.json",
                    "rule_runner/Dockerfile"):
        assert re.search(paths, changed), changed
    assert not re.search(paths, "backend/src/services/rules_evaluation.py")
    doc = yaml.safe_load((WORKFLOWS / "docker-images.yml").read_text())
    resolve = next(s["run"] for s in doc["jobs"]["build"]["steps"] if s.get("id") == "img")
    case = resolve.split("rule-runner)")[1].split(";;")[0]
    assert 'context=."' in case.replace(" ", "") or "context=." in case
    assert "dockerfile=./rule_runner/Dockerfile" in case


def test_the_dockerfile_copies_only_the_runner_and_fetches_by_digest():
    instructions = _instructions()
    copies = [i for i in instructions if i.startswith("COPY")]
    sources = sorted(s for i in copies for s in i.split()[1:-1])
    assert sources == ["LICENSE", "backend/src/rule_runner/", "backend/src/schemas/vendored/midataworks-runner-v1.json",
                       "rule_runner/constraints.txt", "rule_runner/requirements.txt"]
    assert any("COPY backend/src/rule_runner/ /app/mistudio_rule_runner/" == i for i in instructions)
    runs = " ".join(i for i in instructions if i.startswith("RUN"))
    assert "python -m mistudio_rule_runner.fetch" in runs, "the digest-checking fetch must run at build"
    assert not HEX64.search(DOCKERFILE.read_text()), "no literal digest: it comes from the vendored contract"
    assert instructions[0].startswith("FROM python:3.13-slim")
    assert any(i.startswith("USER ") and i.split()[1] != "root" for i in instructions)
    assert any(i.startswith("CMD") and "mistudio_rule_runner.server" in i for i in instructions)


def test_no_webassembly_enters_the_build_context():
    ignore = (REPO / "rule_runner" / "Dockerfile.dockerignore").read_text().splitlines()
    assert ignore[[i for i, l in enumerate(ignore) if l and not l.startswith("#")][0]] == "**"
    assert "**/*.wasm" in ignore and "**/*.cwasm" in ignore


def test_the_ci_job_runs_the_real_runner_and_fails_rather_than_skips():
    doc = yaml.safe_load((WORKFLOWS / "backend-tests.yml").read_text())
    job = doc["jobs"]["rule-runner-tests"]
    assert job["env"]["MISTUDIO_REQUIRE_RUNNER"] == "1"
    runs = "\n".join(step.get("run", "") for step in job["steps"])
    assert "-r rule_runner/requirements.txt" in runs and "-c rule_runner/constraints.txt" in runs
    assert "mistudio_rule_runner.fetch" in runs and "RUNNER_BUNDLE_DIR=" in runs
    assert "tests/unit/test_code_rule_hostile.py" in runs
    host = next(step for step in job["steps"] if "Runner host tests" in step.get("name", ""))
    assert host["working-directory"] == "rule_runner"
    on = doc.get("on") or doc.get(True)
    for event in ("push", "pull_request"):
        assert "rule_runner/**" in on[event]["paths"]
