"""T-49 (042 CF-27; FTASKS 7.43, 7.44): a changed ConfigMap rolls the workloads that read it.

Kubernetes does not restart a Deployment when a ConfigMap it reads changes, so a config-only push (S001's
P2 `MISERVER_API_URL` push is one) would leave the old values running while every manifest says otherwise.
So every Deployment, StatefulSet or DaemonSet that reads an inline `kind: ConfigMap` (a volume,
`envFrom.configMapRef` or `env[].valueFrom.configMapKeyRef`) must change its POD-TEMPLATE annotation
`rebrand.hitsai/config-rev` in the same commit that changes that ConfigMap's `data` (the pod template
changing is what rolls the pods).

EVERY commit of the push is checked against its first parent, not only the last one (miForge 015 P0 review
round 1, item 5: a ConfigMap change in the first of two pushed commits passed a HEAD~1 check). The range
starts at `CONFIG_REV_BASE` (CI sets it to `github.event.before`) when that is an ancestor of HEAD, else at
the merge-base with `origin/main`, else at HEAD~1; the working tree is compared with HEAD too, so a local
run sees an uncommitted change. CI checks out full history (`fetch-depth: 0`, backend-tests.yml).

The self-test on a fixture pair keeps the check from passing vacuously on a tree with no change.
"""
from __future__ import annotations

import os
import subprocess

import pytest
import yaml

from tests.support.legacy_export import REPO_ROOT

ANNOTATION = "rebrand.hitsai/config-rev"
WORKLOADS = ("Deployment", "StatefulSet", "DaemonSet")


def _docs(text: str) -> list[dict]:
    try:
        return [d for d in yaml.safe_load_all(text) if isinstance(d, dict)]
    except yaml.YAMLError:
        return []


def _configmaps(tree: dict[str, str]) -> dict[tuple[str, str], dict]:
    out = {}
    for text in tree.values():
        for doc in _docs(text):
            if doc.get("kind") == "ConfigMap":
                meta = doc.get("metadata") or {}
                out[(meta.get("namespace", ""), meta.get("name", ""))] = {
                    "data": doc.get("data") or {}, "binaryData": doc.get("binaryData") or {}}
    return out


def _consumers(tree: dict[str, str]) -> dict[tuple[str, str, str], tuple[set[str], str | None]]:
    """`{(kind, namespace, name): (ConfigMap names it reads, its pod-template config-rev)}`."""
    out = {}
    for text in tree.values():
        for doc in _docs(text):
            if doc.get("kind") not in WORKLOADS:
                continue
            template = (doc.get("spec") or {}).get("template") or {}
            pod = template.get("spec") or {}
            names = {(v.get("configMap") or {}).get("name") for v in pod.get("volumes") or []}
            for v in pod.get("volumes") or []:
                for src in (v.get("projected") or {}).get("sources") or []:
                    names.add((src.get("configMap") or {}).get("name"))
            for c in [*(pod.get("containers") or []), *(pod.get("initContainers") or [])]:
                names |= {(e.get("configMapRef") or {}).get("name") for e in c.get("envFrom") or []}
                names |= {((e.get("valueFrom") or {}).get("configMapKeyRef") or {}).get("name")
                          for e in c.get("env") or []}
            rev = ((template.get("metadata") or {}).get("annotations") or {}).get(ANNOTATION)
            meta = doc.get("metadata") or {}
            out[(doc["kind"], meta.get("namespace", ""), meta.get("name", ""))] = (names - {None}, rev)
    return out


def findings(before: dict[str, str], after: dict[str, str]) -> list[str]:
    old_maps, new_maps = _configmaps(before), _configmaps(after)
    old_users, new_users = _consumers(before), _consumers(after)
    out = []
    for key, content in sorted(new_maps.items()):
        if key in old_maps and old_maps[key] == content:
            continue
        for workload, (reads, rev) in sorted(new_users.items()):
            if workload[1] == key[0] and key[1] in reads:
                if rev is None or rev == old_users.get(workload, (set(), None))[1]:
                    out.append(f"ConfigMap {key[0]}/{key[1]} changed and {workload[0]} {workload[2]} kept "
                               f"{ANNOTATION}={rev!r}")
    return out


def _git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True)


def _git_out(*args: str) -> str:
    """`git <args>` stdout; a failure RAISES (review rev-mistudio-p0 F3, the miForge `_git` finding: a failed
    `ls-tree` read as an empty tree and a failed `rev-list` as an empty range, and both passed silently)."""
    done = _git(*args)
    if done.returncode != 0:
        raise AssertionError(f"git {' '.join(args)} failed ({done.returncode}): {done.stderr.strip()}")
    return done.stdout


def _tree_at(ref: str | None) -> dict[str, str]:
    """`{path: text}` of every YAML under k8s/ at `ref` (None: the working tree, tracked files)."""
    if ref is None:
        files = _git_out("ls-files", "--", "k8s").split()
        return {f: (REPO_ROOT / f).read_text() for f in files
                if f.endswith((".yaml", ".yml")) and (REPO_ROOT / f).is_file()}
    files = _git_out("ls-tree", "-r", "--name-only", ref, "--", "k8s").split()
    out = {}
    for f in files:
        if f.endswith((".yaml", ".yml")):
            shown = _git("show", f"{ref}:{f}")
            if shown.returncode == 0:
                out[f] = shown.stdout
    return out


def _is_commit(ref: str) -> bool:
    return bool(ref) and set(ref) != {"0"} and _git("cat-file", "-e", f"{ref}^{{commit}}").returncode == 0


def range_base() -> tuple[str | None, str]:
    """`(base, why)`: the commit the checked range starts after."""
    env_base = os.environ.get("CONFIG_REV_BASE", "")
    if _is_commit(env_base) and _git("merge-base", "--is-ancestor", env_base, "HEAD").returncode == 0:
        return env_base, "CONFIG_REV_BASE"
    if env_base and set(env_base) != {"0"} and os.environ.get("CI") == "true":
        # A push whose base CI cannot use (a force-push, a missing object) must not narrow to HEAD~1 and pass
        # every earlier commit unchecked (review rev-mistudio-p0 F3, the miForge HEAD~1 finding).
        pytest.fail(f"CONFIG_REV_BASE={env_base} is not an ancestor commit of HEAD: the pushed range cannot be "
                    "established, and checking only HEAD~1 would pass the earlier commits unchecked")
    if _is_commit("origin/main"):
        mb = _git("merge-base", "HEAD", "origin/main").stdout.strip()
        head = _git("rev-parse", "HEAD").stdout.strip()
        if mb and mb != head:
            return mb, "merge-base with origin/main"
    if _is_commit("HEAD~1"):
        return _git("rev-parse", "HEAD~1").stdout.strip(), "HEAD~1"
    return None, "no parent commit"


def test_configmap_change_bumps_consumers():
    base, why = range_base()
    if base is None:
        if os.environ.get("CI") == "true" and (REPO_ROOT / "0xcc").is_dir():
            pytest.fail("CI in the private repository has no parent commit: backend-tests.yml must check out "
                        "full history for T-49")
        pytest.skip("CONFIG-REV SKIPPED LOUDLY: no parent commit (the mirror's orphan snapshot or a shallow "
                    "clone); the comparison did NOT run")
    commits = _git_out("rev-list", "--reverse", "--first-parent", f"{base}..HEAD").split()
    after_head = _tree_at(None)
    assert after_head, "no manifest under k8s/ was read: the comparison would pass vacuously"
    problems = []
    for commit in commits:
        for p in findings(_tree_at(f"{commit}^1"), _tree_at(commit)):
            problems.append(f"{commit[:9]}: {p}")
    problems += [f"working tree: {p}" for p in findings(_tree_at("HEAD"), after_head)]
    print(f"config-rev: base {base[:9]} ({why}); {len(commits)} commit(s) checked, plus the working tree")
    assert not problems, "\n".join(problems)


CM = "apiVersion: v1\nkind: ConfigMap\nmetadata:\n  name: app-env\n  namespace: ns\ndata:\n  URL: {url}\n"
DEP = ("apiVersion: apps/v1\nkind: Deployment\nmetadata:\n  name: app\n  namespace: ns\n"
       "spec:\n  template:\n    metadata:\n      annotations:\n        rebrand.hitsai/config-rev: \"{rev}\"\n"
       "    spec:\n      containers:\n        - name: app\n          env:\n            - name: URL\n"
       "              valueFrom:\n                configMapKeyRef:\n                  name: app-env\n"
       "                  key: URL\n")


def test_self_test_on_fixture_pair():
    before = {"a.yaml": CM.format(url="http://old") + "---\n" + DEP.format(rev="aaa")}
    unchanged_rev = {"a.yaml": CM.format(url="http://new") + "---\n" + DEP.format(rev="aaa")}
    bumped = {"a.yaml": CM.format(url="http://new") + "---\n" + DEP.format(rev="bbb")}
    other_ns = {"a.yaml": CM.format(url="http://new") + "---\n" + DEP.format(rev="aaa").replace("ns\n", "other\n", 1)}
    assert len(findings(before, unchanged_rev)) == 1
    assert findings(before, bumped) == []
    assert findings(before, before) == []
    assert findings(before, other_ns) == []  # a ConfigMap is read only within its own namespace


def test_every_commit_of_the_range_is_checked(tmp_path, monkeypatch):
    """A ConfigMap change WITHOUT a bump in the first of two commits is found (miForge round 1 item 5)."""
    repo = tmp_path / "r"
    (repo / "k8s").mkdir(parents=True)
    env = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t",
           "GIT_COMMITTER_EMAIL": "t@t"}
    git = lambda *a: subprocess.run(["git", *a], cwd=repo, check=True, capture_output=True, env=env, text=True)  # noqa: E731
    git("init", "-q", "-b", "main")
    (repo / "0xcc").mkdir()
    (repo / "k8s" / "a.yaml").write_text(CM.format(url="http://old") + "---\n" + DEP.format(rev="aaa"))
    git("add", "-A"), git("commit", "-q", "-m", "base")
    base = git("rev-parse", "HEAD").stdout.strip()
    (repo / "k8s" / "a.yaml").write_text(CM.format(url="http://new") + "---\n" + DEP.format(rev="aaa"))
    git("commit", "-q", "-am", "config change, no bump")
    (repo / "README").write_text("x")
    git("add", "-A"), git("commit", "-q", "-m", "unrelated")
    monkeypatch.setitem(globals(), "REPO_ROOT", repo)
    monkeypatch.setenv("CONFIG_REV_BASE", base)
    assert range_base() == (base, "CONFIG_REV_BASE")
    with pytest.raises(AssertionError, match="config change, no bump|kept rebrand.hitsai/config-rev"):
        test_configmap_change_bumps_consumers()


def test_ci_checks_out_the_range():
    """The shards that run this test check out full history and pass the push's base (miDataworks W05)."""
    doc = yaml.safe_load((REPO_ROOT / ".github" / "workflows" / "backend-tests.yml").read_text())
    job = doc["jobs"]["test"]
    assert job["env"].get("CONFIG_REV_BASE") == "${{ github.event.before }}", job["env"]
    checkout = next(s for s in job["steps"] if str(s.get("uses", "")).startswith("actions/checkout@"))
    assert (checkout.get("with") or {}).get("fetch-depth") == 0, checkout


def test_an_unusable_base_on_ci_never_narrows_to_head_1(tmp_path, monkeypatch):
    """Review rev-mistudio-p0 F3: CONFIG_REV_BASE set to a commit CI cannot use (force-push, missing object)
    used to fall back to HEAD~1 silently, so a ConfigMap change in an earlier pushed commit passed."""
    repo = tmp_path / "r"
    (repo / "k8s").mkdir(parents=True)
    env = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t",
           "GIT_COMMITTER_EMAIL": "t@t"}
    git = lambda *a: subprocess.run(["git", *a], cwd=repo, check=True, capture_output=True, env=env, text=True)  # noqa: E731
    git("init", "-q", "-b", "main")
    (repo / "k8s" / "a.yaml").write_text(CM.format(url="http://old") + "---\n" + DEP.format(rev="aaa"))
    git("add", "-A"), git("commit", "-q", "-m", "base")
    (repo / "k8s" / "a.yaml").write_text(CM.format(url="http://new") + "---\n" + DEP.format(rev="aaa"))
    git("commit", "-q", "-am", "config change, no bump")
    (repo / "README").write_text("x")
    git("add", "-A"), git("commit", "-q", "-m", "unrelated")
    monkeypatch.setitem(globals(), "REPO_ROOT", repo)
    monkeypatch.setenv("CI", "true")
    monkeypatch.setenv("CONFIG_REV_BASE", "1" * 40)  # a push base this clone does not have
    with pytest.raises(pytest.fail.Exception, match="not an ancestor"):
        range_base()


def test_a_failed_git_read_is_never_an_empty_tree(tmp_path, monkeypatch):
    """Review rev-mistudio-p0 F3: `_tree_at` of a ref git cannot read raised nothing and returned {}."""
    monkeypatch.setitem(globals(), "REPO_ROOT", tmp_path)  # not a repository at all
    with pytest.raises(AssertionError, match="failed"):
        _tree_at("HEAD")
