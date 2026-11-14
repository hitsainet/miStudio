"""Every image is pushed to ONE Docker Hub repository, `hitsai/miworkbench-<x>` (042 FR-042.29; FTDD §11.2, T-23;
FTASKS 9.2, 9.3, 18.10).

From ENH-009 P1 to P7 each image went to two repositories (the old `hitsai/mistudio-<x>` beside the new one), from
one build tagged twice. P7 stopped the old push (`test_legacy_retired.py::test_single_image_push`, control
K-18.G7a); the old repositories keep their images (DR-27). Both the change detector (`detect`'s `repo_for`) and
the build (`img` step's `repos` output) read the repository from one function, `.github/scripts/image_repos.sh`.

The workflow's own `run:` scripts are EXECUTED here (expressions resolved from the previous steps' real outputs),
not pattern-matched: the assertion is on the tags `docker/build-push-action` would receive.
"""
from __future__ import annotations

import os
import re
import stat
import subprocess
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[3]
WORKFLOWS = REPO / ".github" / "workflows"
IMAGE_REPOS = REPO / ".github" / "scripts" / "image_repos.sh"
IMAGES = ("backend", "frontend", "rule-runner")
EXPR = re.compile(r"\$\{\{\s*([^}]+?)\s*\}\}")
SHA = "0123456789abcdef0123456789abcdef01234567"


def _pair(image: str) -> set[str]:
    """The repositories an image is pushed to: one since ENH-009 P7."""
    return {f"hitsai/miworkbench-{image}"}


def _workflow() -> dict:
    return yaml.safe_load((WORKFLOWS / "docker-images.yml").read_text())


def _resolve(text: str, ctx: dict[str, str]) -> str:
    """Substitute `${{ … }}` from `ctx`; an expression the context does not hold fails (never silently empty)."""
    def sub(m: re.Match[str]) -> str:
        key = m.group(1)
        assert key in ctx, f"unresolved expression {key!r} (context has {sorted(ctx)})"
        return ctx[key]
    return EXPR.sub(sub, text)


def _run_step(step: dict, ctx: dict[str, str], env: dict[str, str], tmp: Path, cwd: Path = REPO) -> dict[str, str]:
    out = tmp / f"out-{step['id']}"
    out.write_text("")
    step_env = {k: _resolve(str(v), ctx) for k, v in (step.get("env") or {}).items()}
    body = _resolve(step["run"], ctx)
    res = subprocess.run(["bash", "-eo", "pipefail", "-c", body], cwd=cwd, capture_output=True, text=True,
                         env={**os.environ, **env, **step_env, "GITHUB_OUTPUT": str(out)})
    assert res.returncode == 0, f"{step.get('name')}: {res.stdout}{res.stderr}"
    values = dict(line.split("=", 1) for line in out.read_text().splitlines() if line)
    for k, v in values.items():
        ctx[f"steps.{step['id']}.outputs.{k}"] = v
    return values


@pytest.mark.parametrize("ref", ["refs/heads/main", "refs/tags/v9.9.9"])
@pytest.mark.parametrize("image", IMAGES)
def test_each_image_pushes_to_exactly_one_repository(image, ref, tmp_path):
    build = _workflow()["jobs"]["build"]
    steps = build["steps"]
    by_id = {s["id"]: s for s in steps if "id" in s}
    ctx = {"matrix.name": image, "github.sha": SHA}
    env = {"GITHUB_REF": ref, "GITHUB_SHA": SHA}
    _run_step(by_id["img"], ctx, env, tmp_path)
    _run_step(by_id["version"], ctx, env, tmp_path)

    # ONE build: a single build-push step, pushing, with the version step's tags.
    builds = [s for s in steps if str(s.get("uses", "")).startswith("docker/build-push-action@")]
    assert len(builds) == 1, builds
    (push,) = builds
    assert push["with"]["push"] is True
    tags = [t.strip() for t in _resolve(str(push["with"]["tags"]), ctx).split(",") if t.strip()]
    repos: dict[str, set[str]] = {}
    for t in tags:
        repo, _, tag = t.rpartition(":")
        repos.setdefault(repo, set()).add(tag)
    assert set(repos) == _pair(image), tags
    suffix = ref.rsplit("/", 1)[1] if ref.startswith("refs/tags/") else SHA[:8]
    # The repository carries :latest and the version tag.
    assert all(t == {"latest", suffix} for t in repos.values()), repos
    assert len(tags) == 2, tags

    # Provenance is attested in the repository, for the one build's digest.
    attests = [s for s in steps if str(s.get("uses", "")).startswith("actions/attest-build-provenance@")]
    subjects = {_resolve(s["with"]["subject-name"], ctx).removeprefix("index.docker.io/") for s in attests}
    assert subjects == _pair(image), subjects
    assert {s["with"]["subject-digest"] for s in attests} == {"${{ steps.build.outputs.digest }}"}
    assert push.get("id") == "build"


def test_unknown_image_is_refused(tmp_path):
    step = next(s for s in _workflow()["jobs"]["build"]["steps"] if s.get("id") == "img")
    out = tmp_path / "out"
    res = subprocess.run(["bash", "-eo", "pipefail", "-c", _resolve(step["run"], {"matrix.name": "nope"})],
                         cwd=REPO, capture_output=True, text=True,
                         env={**os.environ, **{k: _resolve(str(v), {"matrix.name": "nope"})
                                               for k, v in (step.get("env") or {}).items()},
                              "GITHUB_OUTPUT": str(out)})
    assert res.returncode != 0, res.stdout


# ── the change detector reads the repository ───────────────────────────────────────────────────────

FAKE_CURL = r"""#!/usr/bin/env bash
# Stands in for Docker Hub: a repository listed in FAKE_PUBLISHED has :latest built from FAKE_REVISION;
# any other repository does not exist (curl -f exits 22).
url="${@: -1}"
for r in $FAKE_PUBLISHED; do
  if [[ "$url" == *"repository:${r}:pull"* ]]; then echo '{"token":"t"}'; exit 0; fi
  if [[ "$url" == *"/v2/${r}/manifests/"* ]]; then
    echo '{"mediaType":"application/vnd.oci.image.manifest.v1+json","config":{"digest":"sha256:c"}}'; exit 0; fi
  if [[ "$url" == *"/v2/${r}/blobs/"* ]]; then
    echo "{\"config\":{\"Labels\":{\"org.opencontainers.image.revision\":\"${FAKE_REVISION}\"}}}"; exit 0; fi
done
exit 22
"""


def _git(cwd: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=True).stdout.strip()


@pytest.fixture
def checkout(tmp_path):
    """A checkout holding the real `image_repos.sh` (what `actions/checkout` gives the job), one commit."""
    repo = tmp_path / "co"
    (repo / ".github" / "scripts").mkdir(parents=True)
    (repo / ".github" / "scripts" / "image_repos.sh").write_text(IMAGE_REPOS.read_text())
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "add", "-A")
    _git(repo, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "x")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    curl = bin_dir / "curl"
    curl.write_text(FAKE_CURL)
    curl.chmod(curl.stat().st_mode | stat.S_IEXEC)
    return repo, bin_dir


def _detect(checkout, published: set[str], tmp: Path) -> str:
    repo, bin_dir = checkout
    step = next(s for s in _workflow()["jobs"]["detect"]["steps"] if s.get("id") == "detect")
    ctx = {"github.ref": "refs/heads/main", "github.event.created": "false"}
    head = _git(repo, "rev-parse", "HEAD")
    env = {"PATH": f"{bin_dir}:{os.environ['PATH']}", "GITHUB_SHA": head, "FAKE_REVISION": head,
           "FAKE_PUBLISHED": " ".join(sorted(published))}
    return _run_step(step, ctx, env, tmp, cwd=repo)["images"]


def test_change_detection_consults_the_repository(checkout, tmp_path):
    every = set().union(*(_pair(i) for i in IMAGES))
    assert all("/miworkbench-" in r for r in every), every
    # Control: the repository already holds this commit → nothing to build (the stand-in registry is read).
    assert _detect(checkout, every, tmp_path) == "[]"
    # Never pushed → every image builds.
    assert _detect(checkout, set(), tmp_path) == '["backend","frontend","rule-runner"]'
    # An old repository holding this commit does not stand in for the new one (P7: it is not consulted).
    old = {r.replace("/miworkbench-", "/mistudio-") for r in every}
    assert _detect(checkout, old, tmp_path) == '["backend","frontend","rule-runner"]'
    # Per image, not all-or-nothing.
    assert _detect(checkout, every - {"hitsai/miworkbench-frontend"}, tmp_path) == '["frontend"]'


def test_release_notes_pull_the_new_names():
    doc = yaml.safe_load((WORKFLOWS / "release.yml").read_text())
    (step,) = [s for s in doc["jobs"]["release"]["steps"] if "softprops/action-gh-release" in str(s.get("uses"))]
    pulled = re.findall(r"docker pull (\S+?):\$\{\{ github\.ref_name \}\}", step["with"]["body"])
    assert sorted(pulled) == sorted(f"hitsai/miworkbench-{i}" for i in IMAGES), pulled
