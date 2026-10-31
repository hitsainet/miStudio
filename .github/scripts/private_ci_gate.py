#!/usr/bin/env python3
"""Refuse to publish a commit to the public mirror unless THIS repository's CI passed for it.

Run by `sync-to-clean.yml`'s `gate` job in the private repository, before anything is
pushed. The mirror's image build trusts the result: it no longer re-runs the suite, it only
checks that the snapshot carries the `Private-CI: passed` trailer this gate's sync writes. So
this file is the only thing standing between a red suite and a deployed image.

For each workflow in REQUIRED_WORKFLOWS (one file name per line, e.g. `backend-tests.yml`):

* A run for exactly this commit (push or dispatch; a pull_request run tested a merge commit,
  not this one) decides it: completed + success passes, completed + anything else refuses at
  once, still running waits. If the workflow declares an `all-shards` job, that job must also
  have concluded success, so a run is never accepted on a partial set of shards.
* No run for this commit: if the workflow has a `paths:` filter, GitHub legitimately created
  no run because the push touched none of those paths. Then the commit passes only if an
  earlier commit on main that is an ANCESTOR of this one passed the same workflow (with its
  all-shards job), and nothing under the filter's paths changed between the two - i.e. the
  content that workflow tests is byte-identical to content it passed. The paths are read from
  the workflow file itself, so the filter and this rule cannot drift apart.
* Otherwise (no run, and either no filter or no such ancestor): wait out a grace period for a
  late-created run, then refuse. No run is never read as a pass.

Exit 0 and write `sha=<commit>` to $GITHUB_OUTPUT on success; exit 1 with ::error:: otherwise.
Needs: GH_TOKEN (the job's GITHUB_TOKEN with `actions: read`), GITHUB_REPOSITORY, SHA,
REQUIRED_WORKFLOWS, a full-history checkout (ancestry and diffs), and PyYAML.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any

AGGREGATOR_JOB_ID = "all-shards"
API = "https://api.github.com"


def api_get(path: str) -> Any:
    req = urllib.request.Request(
        f"{API}{path}",
        headers={
            "Authorization": f"Bearer {os.environ['GH_TOKEN']}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.load(resp)


def load_workflow(repo_root: Path, workflow: str) -> dict:
    import yaml

    return yaml.safe_load((repo_root / ".github" / "workflows" / workflow).read_text())


def _on(doc: dict) -> dict:
    # PyYAML reads the bare key `on` as boolean True.
    on = doc.get("on", doc.get(True)) or {}
    return on if isinstance(on, dict) else {k: None for k in ([on] if isinstance(on, str) else on)}


def push_pathspecs(doc: dict) -> list[str] | None:
    """git pathspecs equivalent to the workflow's push filter; None when it has none."""
    push = _on(doc).get("push") or {}
    if push.get("paths"):
        specs = []
        for p in push["paths"]:
            specs.append(f":(exclude,glob){p[1:]}" if p.startswith("!") else f":(glob){p}")
        return specs
    if push.get("paths-ignore"):
        return ["."] + [f":(exclude,glob){p}" for p in push["paths-ignore"]]
    return None


def aggregator_name(doc: dict) -> str | None:
    job = (doc.get("jobs") or {}).get(AGGREGATOR_JOB_ID)
    if job is None:
        return None
    return job.get("name", AGGREGATOR_JOB_ID)


def _git_ok(*args: str, cwd: Path) -> bool:
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, check=False).returncode == 0


class Gate:
    def __init__(
        self,
        repo: str,
        sha: str,
        repo_root: Path,
        get: Callable[[str], Any] = api_get,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
        timeout: float = 45 * 60,
        grace: float = 4 * 60,
        interval: float = 30,
    ) -> None:
        self.repo, self.sha, self.root = repo, sha, repo_root
        self.get, self.sleep, self.clock = get, sleep, clock
        self.timeout, self.grace, self.interval = timeout, grace, interval

    def _runs(self, workflow: str, query: str) -> list[dict]:
        data = self.get(
            f"/repos/{self.repo}/actions/workflows/{workflow}/runs?{query}&per_page=100"
        )
        return [r for r in data.get("workflow_runs", []) if r.get("event") != "pull_request"]

    def _aggregator_state(self, run: dict, name: str | None) -> str:
        """`success`, `failed`, or `pending` for the run's aggregator job.

        ⚠ PENDING IS NOT FAILED. The jobs endpoint can lag the run: on 2026-10-22 a run read
        `completed/success` while its `all shards` job had no conclusion yet, the gate refused the
        commit, and nothing reached the mirror until a human re-ran the sync. A job that has not
        concluded (or is not listed yet) is waited for, up to the gate's timeout."""
        if name is None:
            return "success"
        jobs = self.get(
            f"/repos/{self.repo}/actions/runs/{run['id']}/jobs?filter=latest&per_page=100"
        )
        mine = [j for j in jobs.get("jobs", []) if j.get("name") == name]
        if any(j.get("conclusion") == "success" for j in mine):
            return "success"
        if any(j.get("conclusion") not in (None, "") for j in mine):
            return "failed"
        return "pending"

    def _aggregator_ok(self, run: dict, name: str | None) -> bool:
        return self._aggregator_state(run, name) == "success"

    def _ancestor_fallback(self, workflow: str, specs: list[str], agg: str | None) -> str | None:
        for run in self._runs(workflow, "branch=main&status=success"):
            x = run.get("head_sha")
            if run.get("conclusion") != "success" or not x:
                continue
            if not _git_ok("cat-file", "-e", f"{x}^{{commit}}", cwd=self.root):
                continue
            if not _git_ok("merge-base", "--is-ancestor", x, self.sha, cwd=self.root):
                continue
            if not _git_ok("diff", "--quiet", x, self.sha, "--", *specs, cwd=self.root):
                continue
            if self._aggregator_ok(run, agg):
                return x
        return None

    def check(self, workflow: str) -> tuple[bool, str]:
        doc = load_workflow(self.root, workflow)
        specs, agg = push_pathspecs(doc), aggregator_name(doc)
        start = self.clock()
        seen: str | None = None
        while True:
            runs = self._runs(workflow, f"head_sha={self.sha}")
            if not runs and seen is not None:
                # ⚠ A RUN ONCE LISTED IS NOT GONE. On 2026-10-22 one poll listed 3be4c5b0's Backend
                # Tests in progress and the next returned no runs at all; the gate took the
                # "no run" branch below and refused a commit whose tests then passed. Ask again.
                note = f"run {seen} not in this answer; asking again"
                if self.clock() - start >= self.timeout:
                    return False, f"{workflow}: timed out waiting for {self.sha} ({note})"
                print(f"waiting for {workflow}: {note}", flush=True)
                self.sleep(self.interval)
                continue
            if runs:
                run = max(
                    runs, key=lambda r: (r.get("created_at") or "", r.get("run_attempt") or 0)
                )
                seen = str(run.get("id"))
                status, concl = run.get("status"), run.get("conclusion")
                if status == "completed":
                    if concl != "success":
                        return (
                            False,
                            f"{workflow}: run {run.get('id')} for {self.sha} concluded {concl}",
                        )
                    state = self._aggregator_state(run, agg)
                    if state == "failed":
                        return (
                            False,
                            f"{workflow}: run {run.get('id')} passed but its '{agg}' job did not",
                        )
                    if state == "success":
                        return True, f"{workflow}: run {run.get('id')} passed for {self.sha}"
                    note = f"run {run.get('id')} passed; waiting for its '{agg}' job to report"
                else:
                    note = f"run {run.get('id')} {status}"
            else:
                if specs is not None:
                    x = self._ancestor_fallback(workflow, specs, agg)
                    if x:
                        return True, (
                            f"{workflow}: no run for {self.sha} (path filter not matched); ancestor "
                            f"{x} passed and nothing under {specs} changed since"
                        )
                if self.clock() - start >= self.grace:
                    why = (
                        "it has no path filter"
                        if specs is None
                        else "no passing ancestor has the same tested paths"
                    )
                    return False, f"{workflow}: no run for {self.sha}, and {why}"
                note = "no run yet"
            if self.clock() - start >= self.timeout:
                return False, f"{workflow}: timed out waiting for {self.sha} ({note})"
            print(f"waiting for {workflow}: {note}", flush=True)
            self.sleep(self.interval)


def required(text: str) -> list[str]:
    return [w.strip() for w in text.splitlines() if w.strip() and not w.strip().startswith("#")]


def main() -> int:
    repo, sha = os.environ["GITHUB_REPOSITORY"], os.environ["SHA"]
    workflows = required(os.environ.get("REQUIRED_WORKFLOWS", ""))
    if not workflows:
        print("::error::REQUIRED_WORKFLOWS is empty - refusing rather than passing vacuously")
        return 1
    gate = Gate(repo, sha, Path.cwd())
    failed = False
    for wf in workflows:
        ok, msg = gate.check(wf)
        print(msg if ok else f"::error::{msg}", flush=True)
        failed |= not ok
    if failed:
        print(f"::error::Private CI did not pass for {sha}. Nothing is published to the mirror.")
        return 1
    out = os.environ.get("GITHUB_OUTPUT")
    if out:
        with open(out, "a") as fh:
            fh.write(f"sha={sha}\n")
    print(f"Private CI passed for {sha}: {', '.join(workflows)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
