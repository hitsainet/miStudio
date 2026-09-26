"""Publish a probe definition to HuggingFace (033 FR-8, tasks 4.1–4.2).

Three files per probe, in one folder, uploaded as one commit:

  `<slug>.probe.json`   the definition
  `manifest.json`       every probe in the repo, MERGED — never replaced wholesale
  `README.md`           front-matter tags, `base_model`, and what has and has not been checked

⚠ THE MANIFEST IS MERGED BY NAME AND NEVER DELETES. A repo accumulates probes over time, and a
publisher that wrote its own list would silently remove every probe someone else published — the
kind of data loss that is invisible until a consumer looks for a probe that used to be there.
Replace the entry whose `name` matches, append otherwise, keep the rest untouched.

⚠ THE README SAYS WHAT HAS *NOT* BEEN CHECKED, and that is not modesty. A probe reaching rung 2 has
been measured on held-out out-of-distribution data and has NOT been compared against a judge, has
NOT been shown to be causal, and has NOT been validated on the consumer's own traffic. A reader who
takes "detects on unseen tasks" for "is reliable here" will over-trust it, and the document is the
only place that can say so.

⚠ THE TOKEN IS NEVER PERSISTED OR LOGGED. It arrives as an argument, reaches `HfApi`, and is
written nowhere — not to the row, not to the task result, not to a log line. `_redact` is applied to
anything echoed back.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..core.clock import utc_now
from ..core.config import settings

logger = logging.getLogger(__name__)

MANIFEST_NAME = "manifest.json"
README_NAME = "README.md"
#: The tag a consumer filters on: `list_models(filter=["mistudio-probe-definition"])`.
DISCOVERY_TAG = "mistudio-probe-definition"


class ProbePublishFailed(Exception):
    """An upload that did not happen, with a reason a reader can act on."""


def merge_manifest(existing: Optional[Dict[str, Any]], entry: Dict[str, Any]) -> Dict[str, Any]:
    """Replace by `name`, else append. NEVER delete (task 4.1).

    Pure, so the rule is testable without HuggingFace. `existing` is whatever was in the repo —
    including None (no manifest yet) and a malformed document, which is treated as absent rather
    than raising: refusing to publish because someone hand-edited the manifest would strand the
    probe, while rebuilding around the entry keeps the repo usable.
    """
    probes: List[Dict[str, Any]] = []
    if isinstance(existing, dict) and isinstance(existing.get("probes"), list):
        probes = [item for item in existing["probes"] if isinstance(item, dict)]
    elif existing:
        logger.warning(
            "the repo's %s is not a probe manifest (%s); treating it as absent rather than "
            "refusing to publish",
            MANIFEST_NAME, type(existing).__name__,
        )

    merged = [item for item in probes if item.get("name") != entry.get("name")]
    replaced = len(merged) != len(probes)
    merged.append(entry)
    merged.sort(key=lambda item: str(item.get("name") or ""))
    return {
        "kind": "mistudio.probe-manifest/v1",
        "updated_at": utc_now().isoformat(),
        "probes": merged,
        # Recorded so a reader of the commit can tell an update from a first publication.
        "last_action": "replaced" if replaced else "appended",
    }


def manifest_entry(definition: Dict[str, Any], filename: str) -> Dict[str, Any]:
    """The one-line summary a consumer scans before downloading a 2 MB document."""
    evidence = definition.get("evidence") or {}
    model = definition.get("model") or {}
    aurocs = [
        float(item["auroc"])
        for item in evidence.get("evaluations") or []
        if item.get("auroc") is not None
    ]
    return {
        "name": definition.get("name"),
        "file": filename,
        "kind": definition.get("kind"),
        "concept": definition.get("concept"),
        "base_model": model.get("hf_id"),
        "model_revision": model.get("revision"),
        "layer": (definition.get("read") or {}).get("layer"),
        "basis": definition.get("basis"),
        "rule": (definition.get("aggregation") or {}).get("rule"),
        "rung": evidence.get("rung"),
        "rung_language": evidence.get("rung_language"),
        "acknowledged_below_rung2": bool(evidence.get("acknowledgement")),
        "mean_auroc": round(sum(aurocs) / len(aurocs), 4) if aurocs else None,
        "evaluation_sets": len(evidence.get("evaluations") or []),
        "published_at": utc_now().isoformat(),
    }


def render_readme(definition: Dict[str, Any], manifest: Dict[str, Any]) -> str:
    """The model card: front matter a consumer can filter on, then the honest part."""
    model = definition.get("model") or {}
    evidence = definition.get("evidence") or {}
    aggregation = definition.get("aggregation") or {}
    decision = definition.get("decision") or {}
    read = definition.get("read") or {}
    rung = evidence.get("rung")

    rows = []
    for item in evidence.get("evaluations") or []:
        dataset = item.get("dataset") or {}
        name = dataset.get("config") or dataset.get("hf_id") or "?"
        ci = item.get("auroc_ci")
        rows.append(
            f"| `{name}` | {item.get('distribution')} | {item.get('auroc'):.4f} | "
            f"{f'[{ci[0]:.4f}, {ci[1]:.4f}]' if ci else '—'} | "
            f"{item.get('n_positive')}/{item.get('n_negative')} |"
        )

    checked = [
        "Trained on a labelled dataset whose label column is recorded in `provenance`.",
        f"Read at layer {read.get('layer')} of `{model.get('hf_id')}` at revision "
        f"`{str(model.get('revision'))[:12]}`, at `resid_post`.",
    ]
    if evidence.get("evaluations"):
        checked.append(
            f"Measured on {len(evidence['evaluations'])} evaluation set(s) with bootstrapped "
            f"confidence intervals, reported above."
        )
    if decision.get("threshold") is not None:
        checked.append(
            f"An operating point calibrated on `{decision.get('threshold_source')}` at a target "
            f"FPR of {decision.get('target_fpr')} (realised {decision.get('realised_fpr')})."
        )
    checked.append(
        f"{len((definition.get('test_vectors') or {}).get('vectors') or [])} test vectors scored "
        f"by the exporting build, so a consumer can verify its own implementation."
    )

    not_checked = [
        "**Causality.** These are correlational readouts. Nothing here shows the model USES this "
        "direction, only that it is linearly present.",
        "**Your traffic.** The evaluation sets are listed above. A probe's AUROC on data unlike "
        "them is unknown, and out-of-distribution numbers here vary by more than 0.3 between sets.",
    ]
    if rung is not None and int(rung) < 3:
        not_checked.append(
            "**A judge comparison.** This probe has not been measured against an LLM judge on the "
            "same sets, so there is no evidence it beats simply asking a model."
        )
    if rung is not None and int(rung) < 2:
        not_checked.append(
            "**Held-out generalisation.** This probe has NOT cleared chance on an "
            "out-of-distribution set. It was exported with an explicit acknowledgement, recorded "
            "in `evidence.acknowledgement`."
        )
    if evidence.get("acknowledgement"):
        acknowledgement = evidence["acknowledgement"]
        not_checked.append(
            f"**It was exported below rung 2 on a judgement**, by `{acknowledgement.get('by')}`: "
            f"\"{acknowledgement.get('reason')}\""
        )

    tags = [
        DISCOVERY_TAG,
        "mechanistic-interpretability",
        "probe",
        f"rung-{rung}",
        f"basis-{definition.get('basis')}",
    ]
    front = ["---", "tags:"]
    front += [f"  - {tag}" for tag in tags]
    if model.get("hf_id"):
        front.append(f"base_model: {model['hf_id']}")
    front += ["library_name: mistudio", "---", ""]

    body = [
        f"# {definition.get('name')}",
        "",
        definition.get("description") or "",
        "",
        f"**Concept.** {definition.get('concept') or 'not recorded'}",
        "",
        f"**Evidence rung {rung} — {evidence.get('rung_language')}.**",
        "",
        "| field | value |",
        "|---|---|",
        f"| base model | `{model.get('hf_id')}` |",
        f"| revision | `{model.get('revision')}` |",
        f"| layer / hook | {read.get('layer')} / `{read.get('hook_point')}` |",
        f"| basis | `{definition.get('basis')}` |",
        f"| scope | `{definition.get('scope')}` |",
        f"| rule | `{aggregation.get('rule')}` (streamable: {aggregation.get('streamable')}) |",
        f"| threshold | {decision.get('threshold')} |",
        "",
    ]
    if rows:
        body += [
            "## Evaluations",
            "",
            "| set | distribution | AUROC | 95% CI | +/− |",
            "|---|---|---|---|---|",
            *rows,
            "",
        ]
    body += [
        "## What has been checked",
        "",
        *[f"- {line}" for line in checked],
        "",
        "## What has NOT been checked",
        "",
        *[f"- {line}" for line in not_checked],
        "",
        "## Using it",
        "",
        "The definition validates against "
        "[`probe-definition-v1.json`](https://raw.githubusercontent.com/hitsainet/miStudio/main/"
        "docs/schemas/probe-definition-v1.json). Reproduce `test_vectors` before trusting an "
        "implementation: a mismatch beyond "
        f"`{(definition.get('test_vectors') or {}).get('tolerance')}` means the basis, the "
        "normalisation or the combining rule differs, and every one of those produces plausible "
        "scores about the wrong thing.",
        "",
        f"_{len(manifest.get('probes') or [])} probe(s) in this repository._",
        "",
    ]
    return "\n".join(front + body)


def _redact(value: Optional[str]) -> str:
    return "***" if value else "(none)"


def publish(
    db: Any,
    probe_id: str,
    *,
    repo_id: str,
    private: bool = True,
    token: Optional[str] = None,
    api: Any = None,
) -> Dict[str, Any]:
    """Upload the built definition, its merged manifest and a README as one commit.

    `api` is injected by the tests — a mocked `HfApi` whose `upload_folder` payload is asserted, so
    the three files, the tags and the merge are checked without a network call.
    """
    from ..models.probe_monitor import ProbeMonitor

    probe = db.query(ProbeMonitor).filter(ProbeMonitor.id == probe_id).first()
    if probe is None:
        raise ProbePublishFailed(f"probe {probe_id} not found")
    if not probe.definition_path:
        raise ProbePublishFailed(
            f"probe {probe_id} has no built definition; build it before publishing"
        )
    path = settings.resolve_data_path(probe.definition_path)
    if not path.exists():
        raise ProbePublishFailed(f"the definition file {path} is gone; build again")

    definition = json.loads(path.read_text())
    filename = f"{definition.get('name') or probe_id}.probe.json"

    if api is None:
        from huggingface_hub import HfApi

        api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)

    existing = _read_existing_manifest(api, repo_id)
    manifest = merge_manifest(existing, manifest_entry(definition, filename))

    outbox = Path(settings.data_dir) / "probe_definitions" / "outbox" / probe_id
    if outbox.exists():
        import shutil

        shutil.rmtree(outbox)
    outbox.mkdir(parents=True, exist_ok=True)
    (outbox / filename).write_text(json.dumps(definition, indent=2) + "\n")
    (outbox / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2) + "\n")
    (outbox / README_NAME).write_text(render_readme(definition, manifest))

    api.upload_folder(
        folder_path=str(outbox),
        repo_id=repo_id,
        repo_type="model",
        commit_message=(
            f"probe {definition.get('name')} — rung {(definition.get('evidence') or {}).get('rung')}"
        ),
    )
    revision = _resolve_revision(api, repo_id)

    published = list(probe.published or [])
    published.append(
        {
            "repo_id": repo_id,
            "revision": revision,
            "path": filename,
            "private": bool(private),
            "at": utc_now().isoformat(),
            "sha256": probe.definition_sha256,
        }
    )
    probe.published = published
    db.commit()
    logger.info(
        "probe %s published to %s (private=%s, token=%s)",
        probe_id, repo_id, private, _redact(token),
    )
    return {
        "status": "published",
        "probe_id": probe_id,
        "repo_id": repo_id,
        "revision": revision,
        "path": filename,
        "url": f"https://huggingface.co/{repo_id}/blob/main/{filename}",
        "manifest_action": manifest["last_action"],
        "probes_in_repo": len(manifest["probes"]),
    }


def _read_existing_manifest(api: Any, repo_id: str) -> Optional[Dict[str, Any]]:
    """The repo's current manifest, or None. A missing file is normal, not an error."""
    try:
        from huggingface_hub import hf_hub_download

        downloader = getattr(api, "hf_hub_download", None) or hf_hub_download
        local = downloader(repo_id=repo_id, filename=MANIFEST_NAME, repo_type="model")
        return json.loads(Path(local).read_text())
    except Exception as exc:  # noqa: BLE001 - a first publication has no manifest
        logger.info("no existing %s in %s (%s); starting one", MANIFEST_NAME, repo_id, exc)
        return None


def _resolve_revision(api: Any, repo_id: str) -> Optional[str]:
    """The commit the upload produced, so `published` pins what was actually sent."""
    try:
        info = api.model_info(repo_id=repo_id)
        return getattr(info, "sha", None) or (info.get("sha") if isinstance(info, dict) else None)
    except Exception as exc:  # noqa: BLE001 - the upload succeeded; the sha is a nicety
        logger.warning("could not resolve %s's revision after upload: %s", repo_id, exc)
        return None
