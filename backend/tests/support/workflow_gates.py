"""Repository gates in this repository's workflows, read from the PARSED YAML (042 FR-042.12; FTDD §11.1).

A workflow job or step that may run in only one repository says so with an `if:` comparing
`github.repository` with a name. ENH-009 P1 renames both repositories (the private and the public one, each
from its `…/miStudio` name to `…/miWorkbench`), so until P1 CLOSES every gate names both the
old and the new name for its owner; from the first export whose header says `closed: "P1"` a gate names
the new name only (CF-17, CF-21). The old/new pairs come from the export's `repo` records for THIS
repository (CF-13): no test spells a repository name.

`github.repository` is case-preserving and GitHub resolves names case-insensitively, so a gate may also
carry a case variant of the old name (`hitsainet/mistudio`); it counts as the old half.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from tests.support import legacy_export as le

REPO_ROOT = le.REPO_ROOT
WORKFLOWS = REPO_ROOT / ".github" / "workflows"
COMPARISON = re.compile(r"github\.repository\s*==\s*'([^']+)'")
#: Any other way of naming a repository in a condition is refused, so a gate cannot hide from this check.
OTHER_FORMS = re.compile(r"github\.repository\s*!=|github\.repository_owner|contains\(\s*github\.repository|"
                         r"startsWith\(\s*github\.repository")


@dataclass(frozen=True)
class Gate:
    workflow: str
    where: str  # "jobs.<id>" or "jobs.<id>.steps[<n>]"
    condition: str
    names: tuple[str, ...]


def load_workflows(root: Path = REPO_ROOT) -> dict[str, dict[str, Any]]:
    out = {}
    for path in sorted((root / ".github" / "workflows").glob("*.yml")):
        doc = yaml.safe_load(path.read_text())
        doc["on"] = doc.pop(True, doc.get("on"))  # PyYAML reads the bare key `on` as True
        out[path.name] = doc
    return out


def gates(workflows: dict[str, dict[str, Any]]) -> list[Gate]:
    found: list[Gate] = []
    for name, doc in workflows.items():
        for job_id, job in (doc.get("jobs") or {}).items():
            items = [(f"jobs.{job_id}", job.get("if"))]
            items += [(f"jobs.{job_id}.steps[{i}]", s.get("if")) for i, s in enumerate(job.get("steps") or [])]
            for where, cond in items:
                if cond is None:
                    continue
                cond = str(cond)
                names = tuple(COMPARISON.findall(cond))
                if names or "github.repository" in cond:
                    found.append(Gate(name, where, cond, names))
    return found


def this_repo_pairs(export: le.Export) -> dict[str, tuple[str, str]]:
    """`{owner: (old, new)}` for this repository's `repo` records (new name = the header's `repo`)."""
    pairs: dict[str, tuple[str, str]] = {}
    for r in export.repos():
        new_owner, new_name = r["new"].split("/", 1)
        if new_name == export.header["repo"]:
            pairs[r["old"].split("/", 1)[0]] = (r["old"], r["new"])
    return pairs


def findings(workflows: dict[str, dict[str, Any]], export: le.Export) -> list[str]:
    """Every gate that breaks the rule for the export's `closed` phase (never its `phase`, CF-21)."""
    pairs = this_repo_pairs(export)
    if not pairs:
        return ["the export has no repo record for this repository: nothing to check against"]
    new_only = export.closed_at_least("P1")
    out: list[str] = []
    for g in gates(workflows):
        at = f"{g.workflow} {g.where}: {g.condition!r}"
        if OTHER_FORMS.search(g.condition) or not g.names:
            out.append(f"{at}: names a repository in a form this check cannot read")
            continue
        by_owner: dict[str, list[str]] = {}
        for n in g.names:
            by_owner.setdefault(n.split("/", 1)[0], []).append(n)
        for owner, names in by_owner.items():
            if owner not in pairs:
                out.append(f"{at}: owner {owner!r} has no repo record for this repository")
                continue
            old, new = pairs[owner]
            olds = [n for n in names if n.lower() == old.lower()]
            news = [n for n in names if n == new]
            stray = [n for n in names if n not in olds and n not in news]
            if stray:
                out.append(f"{at}: names {stray} (neither {old!r} nor {new!r})")
            if new_only:
                if olds:
                    out.append(f"{at}: P1 is closed; the old half {olds} must go (CF-17)")
                if not news:
                    out.append(f"{at}: P1 is closed; it must name {new!r}")
            else:
                if not olds or not news:
                    out.append(f"{at}: until P1 closes it must name both {old!r} and {new!r} (FR-042.12)")
    return out


def owners_in(workflows: dict[str, dict[str, Any]], workflow: str, job: str) -> set[str]:
    cond = str(workflows[workflow]["jobs"][job].get("if", ""))
    return {n.split("/", 1)[0] for n in COMPARISON.findall(cond)}


def private_and_public_owner(workflows: dict[str, dict[str, Any]], export: le.Export) -> tuple[str, str]:
    """The public owner is the one the mirror's image build is gated on (`docker-images.yml` `build`); the
    private owner is the OTHER owner of this repository's `repo` records."""
    public = owners_in(workflows, "docker-images.yml", "build")
    owners = set(this_repo_pairs(export))
    assert len(public) == 1 and public <= owners, (public, owners)
    private = owners - public
    assert len(private) == 1, (owners, public)
    return private.pop(), public.pop()


def private_names(export: le.Export, private_owner: str) -> set[str]:
    """The names the private repository answers to: both until P1 closes, the new one after (CF-17)."""
    old, new = this_repo_pairs(export)[private_owner]
    return {new} if export.closed_at_least("P1") else {old, new}
