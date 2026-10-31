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

import hashlib
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


#: The identity keys a manifest entry, a README and a `published` record carry (Feature 039 FR-21).
IDENTITY_FIELDS = ("detector_digest", "decision_digest", "decision_revision")


def publication_identity(definition: Dict[str, Any], probe: Any = None) -> Dict[str, Any]:
    """The identity a published copy states: the document's own `identity` block when it carries
    one (`identity_source: "built"`), else — a file built before 039 — the values the read surfaces
    recomputed for it (`identity_fields(probe)`, E6 (a)). The shipped bytes are never touched.
    """
    carried = definition.get("identity")
    if isinstance(carried, dict) and carried.get("detector_digest"):
        return {**{key: carried.get(key) for key in IDENTITY_FIELDS}, "identity_source": "built"}
    if probe is not None:
        from .probe_decision import identity_fields

        fields = identity_fields(probe)
        return {**{key: fields.get(key) for key in IDENTITY_FIELDS},
                "identity_source": fields.get("identity_source") or "not_recorded"}
    return {**dict.fromkeys(IDENTITY_FIELDS), "identity_source": "not_recorded"}


def manifest_entry(
    definition: Dict[str, Any], filename: str, *, sha256: Optional[str] = None,
    identity: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """The one-line summary a consumer scans before downloading a 2 MB document.

    ⚠ IT CARRIED NO DIGEST AT ALL, which is how the re-serialisation defect above went unnoticed:
    nothing in the published repository stated what the file should hash to, so nothing could
    disagree with it. A manifest that summarises a document and cannot pin it is a catalogue, not
    provenance.
    """
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
        "sha256": sha256,
        "authoritative_input": (definition.get("test_vectors") or {}).get(
            "authoritative_input", "token_ids"
        ),
        "messages_reproduce_token_ids": (definition.get("test_vectors") or {}).get(
            "messages_reproduce_token_ids"
        ),
        "published_at": utc_now().isoformat(),
        # ⚠ FEATURE 039 (FR-21): THE JOIN KEYS, so a hub reader can match a served probe's events
        # to this file without downloading 2 MB to recompute them.
        **(identity if identity is not None else publication_identity(definition)),
    }


def _identity_rows(manifest: Dict[str, Any], definition: Dict[str, Any]) -> List[str]:
    """The three identity rows of the README (FR-21), read from the manifest entry the publisher
    just built — so the card and the manifest cannot state two different identities."""
    name = definition.get("name")
    entry = next((item for item in manifest.get("probes") or [] if item.get("name") == name), {})
    value = lambda key: f"`{entry[key]}`" if entry.get(key) is not None else "not recorded"  # noqa: E731
    return [
        f"| detector digest | {value('detector_digest')} |",
        f"| decision digest | {value('decision_digest')} |",
        f"| decision revision | {value('decision_revision')} |",
    ]


def _definition_sha256(manifest: Dict[str, Any], definition: Dict[str, Any]) -> str:
    """This document's digest, from the manifest entry the publisher just built.

    ⚠ NOTHING IN A PUBLISHED REPOSITORY USED TO STATE WHAT THE FILE SHOULD HASH TO, which is how
    the publisher's re-serialisation defect survived: there was no claim for the bytes to disagree
    with. Read from the manifest rather than recomputed here, so the card and the manifest cannot
    state two different digests.
    """
    name = definition.get("name")
    for entry in manifest.get("probes") or []:
        if entry.get("name") == name and entry.get("sha256"):
            return str(entry["sha256"])
    return "not recorded"



def _round_trip_sentence(definition: Dict[str, Any]) -> str:
    """One sentence stating what the exporter MEASURED about its own `messages`.

    Three states, three different sentences. "Not checked" is not "it does not round-trip": a
    consumer may reasonably treat them differently, and claiming the stronger one would be stating
    something the exporter does not know.
    """
    measured = (definition.get("test_vectors") or {}).get("messages_reproduce_token_ids")
    if measured is True:
        return (
            "The exporting build re-rendered these conversations and confirmed they reproduce the "
            "token ids exactly, so starting from `messages` is safe for this document."
        )
    if measured is False:
        return (
            "**The exporting build re-rendered these conversations and they do NOT reproduce the "
            "token ids**, so scoring `messages` will not match and a mismatch means nothing about "
            "your implementation."
        )
    return (
        "The exporting build did not check whether these conversations reproduce the token ids, so "
        "treat `messages` as unverified and score the ids."
    )


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

    vector_field = (definition.get("test_vectors") or {}).get(
        "authoritative_input", "token_ids"
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
        f"| sha256 | `{_definition_sha256(manifest, definition)}` |",
        *_identity_rows(manifest, definition),
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
        # ⚠ THE README IS WHERE A CONSUMER LOOKS FIRST, AND IT DID NOT SAY THIS.
        # Measured on this estate: re-rendering a vector's `messages` and scoring that misses the
        # recorded score by up to 1.153 — 23x the tolerance above — on every vector, while scoring
        # the recorded `token_ids` reproduces exactly. A reader who follows the paragraph above
        # without this one concludes their implementation is broken when it is correct.
        f"### ⚠ Score `{vector_field}`, not `messages`",
        "",
        f"`test_vectors.authoritative_input` is `{vector_field}`. {_round_trip_sentence(definition)}",
        "",
        "`messages` is a reconstruction provided so a consumer has something sendable. Where the "
        "training corpus is prose rather than conversation, re-rendering it through the model's "
        "chat template adds tokens the scored row never had — which moves the score far more than "
        "the tolerance above, on every vector. Tokenise nothing; score the ids as given.",
        "",
        f"_{len(manifest.get('probes') or [])} probe(s) in this repository._",
        "",
    ]
    return "\n".join(front + body)


def render_repo_readme(manifest: Dict[str, Any], definitions_by_file: Dict[str, Dict[str, Any]]) -> str:
    """The repository's model card: ONE SECTION PER PROBE IN THE MERGED MANIFEST (Feature 041 T15).

    ⚠ PRE-EXISTING DEFECT, FIXED HERE: `publish` wrote `render_readme(definition, manifest)` — a card about
    the probe just published — so a repository holding several probes described only the last one, while
    its manifest listed them all. Each probe in the manifest gets its section: rendered from its definition
    when the publisher has it (the files of this publish), else a summary from its manifest entry (a probe
    published earlier, whose definition is not local — TI13; no extra download).

    The front matter is the union of the sections' tags, with `base_model` only when every probe states
    the same one."""
    probes = list(manifest.get("probes") or [])
    tags: List[str] = [DISCOVERY_TAG, "mechanistic-interpretability", "probe"]
    bases = set()
    sections: List[str] = []
    for entry in probes:
        definition = definitions_by_file.get(str(entry.get("file")))
        bases.add(entry.get("base_model"))
        if definition is not None:
            card = render_readme(definition, manifest)
            body = card.split("---\n", 2)[2] if card.startswith("---\n") else card
            for line in card.split("\n"):
                if line.startswith("  - ") and line[4:] not in tags:
                    tags.append(line[4:])
            sections.append("\n".join(("#" + line) if line.startswith("#") else line for line in body.split("\n")))
        else:
            value = lambda key: f"`{entry.get(key)}`" if entry.get(key) is not None else "not recorded"  # noqa: E731
            sections.append("\n".join([
                f"## {entry.get('name')}",
                "",
                f"Published earlier; its definition is `{entry.get('file')}` in this repository. Summary from the "
                "manifest:",
                "",
                "| field | value |",
                "|---|---|",
                f"| base model | {value('base_model')} |",
                f"| layer | {value('layer')} |",
                f"| rule | {value('rule')} |",
                f"| rung | {value('rung')} ({entry.get('rung_language') or 'not recorded'}) |",
                f"| sha256 | {value('sha256')} |",
                f"| detector digest | {value('detector_digest')} |",
                f"| decision digest | {value('decision_digest')} |",
                f"| decision revision | {value('decision_revision')} |",
                "",
            ]))
    front = ["---", "tags:", *[f"  - {tag}" for tag in tags]]
    if len(bases) == 1 and None not in bases:
        front.append(f"base_model: {next(iter(bases))}")
    front += ["library_name: mistudio", "---", ""]
    head = [f"# Probe definitions ({len(probes)})", "",
            "Each probe below has its own section, read from its own definition (or, for one published "
            "earlier, from the manifest).", ""]
    return "\n".join(front + head) + "\n".join(sections)


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

    # ⚠ THE BYTES ARE READ ONCE AND SHIPPED UNCHANGED. `definition` is for reading fields out of;
    # `payload` is what goes to the Hub. They used to be the same object round-tripped through
    # `json.dumps(..., indent=2) + "\n"`, and Python's separators differ from pydantic's — so the
    # published file was NOT the file that was built:
    #
    #     row's recorded sha   3713fe5a…      what the Hub actually served   be7e9d6a…
    #
    # A consumer verifying the download against the digest in its provenance would have found a
    # mismatch on every single published probe. This is the same defect as the builder's three
    # independent serialisations, one layer further out — the fix is the same: measure and ship the
    # same bytes, and never re-render a document you already have.
    payload = path.read_bytes()
    definition = json.loads(payload)
    definition_sha256 = hashlib.sha256(payload).hexdigest()
    filename = f"{definition.get('name') or probe_id}.probe.json"

    if api is None:
        from huggingface_hub import HfApi

        api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)

    existing = _read_existing_manifest(api, repo_id)
    identity = publication_identity(definition, probe)
    manifest = merge_manifest(
        existing, manifest_entry(definition, filename, sha256=definition_sha256, identity=identity)
    )

    outbox = Path(settings.data_dir) / "probe_definitions" / "outbox" / probe_id
    if outbox.exists():
        import shutil

        shutil.rmtree(outbox)
    outbox.mkdir(parents=True, exist_ok=True)
    (outbox / filename).write_bytes(payload)        # the built bytes, not a re-render
    (outbox / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2) + "\n")
    (outbox / README_NAME).write_text(render_repo_readme(manifest, {filename: definition}))

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
            # Feature 039 (FR-21): which detector and decision this copy states.
            **{key: identity.get(key) for key in IDENTITY_FIELDS},
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


class FamilyPublishRefused(ProbePublishFailed):
    """Refused before any Hugging Face call (`member_filename_collision`, a missing definition)."""

    def __init__(self, code: str, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.code = code
        self.details = details or {}


def publish_family(
    db: Any,
    probe_ids: List[str],
    *,
    repo_id: str,
    private: bool = True,
    token: Optional[str] = None,
    api: Any = None,
) -> Dict[str, Any]:
    """Feature 041 FR-24: every member's built definition, the merged manifest and one README, in ONE
    `upload_folder` commit. The single publish's rules, generalised: each member's bytes are read once and
    shipped untouched; `merge_manifest` folded once per member; filenames that collide are refused BEFORE
    any Hugging Face call; each probe's `published` list gains its entry after the upload."""
    from ..models.probe_monitor import ProbeMonitor

    members = []
    for probe_id in probe_ids:
        probe = db.query(ProbeMonitor).filter(ProbeMonitor.id == probe_id).first()
        if probe is None or not probe.definition_path:
            raise FamilyPublishRefused("family_export_blocked", f"probe {probe_id} has no built definition",
                                       {"probe_id": probe_id})
        path = settings.resolve_data_path(probe.definition_path)
        if not path.exists():
            raise FamilyPublishRefused("family_export_blocked", f"probe {probe_id}'s definition file is gone",
                                       {"probe_id": probe_id})
        payload = path.read_bytes()
        definition = json.loads(payload)
        members.append((probe, payload, definition, f"{definition.get('name') or probe_id}.probe.json"))
    by_file: Dict[str, List[str]] = {}
    for probe, _payload, _definition, filename in members:
        by_file.setdefault(filename, []).append(probe.id)
    collisions = {f: ids for f, ids in by_file.items() if len(ids) > 1}
    if collisions:
        raise FamilyPublishRefused("member_filename_collision",
                                   f"members share a definition name, so their files would overwrite each other: "
                                   f"{collisions}", {"collisions": collisions})

    # ⚠ THE EXPORT'S BLOCKERS, AGAIN, AT UPLOAD TIME (Review R1): the route checks them before queueing, but a
    # member can be re-cut or invalidated while the task waits on `processing`; this commit must never carry a
    # definition behind its decision or below rung 2 without its acknowledgement.
    from ..models.probe_monitor import ProbeMonitorRun
    from .probe_family_export import blockers as export_blockers

    probes = [probe for probe, _p, _d, _f in members]
    runs = {r.id: r for r in db.query(ProbeMonitorRun).filter(ProbeMonitorRun.id.in_([p.run_id for p in probes]))}
    found = export_blockers(probes, runs)
    if found:
        raise FamilyPublishRefused("family_export_blocked", f"{len(found)} blocker(s) stop the family publish",
                                   {"blockers": [{k: v for k, v in b.items() if k != "next"} for b in found]})

    if api is None:
        from huggingface_hub import HfApi

        api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)
    manifest = _read_existing_manifest(api, repo_id)
    identities = {}
    for probe, payload, definition, filename in members:
        identities[probe.id] = publication_identity(definition, probe)
        manifest = merge_manifest(manifest, manifest_entry(
            definition, filename, sha256=hashlib.sha256(payload).hexdigest(), identity=identities[probe.id]))
    outbox = Path(settings.data_dir) / "probe_definitions" / "outbox" / f"family-{hashlib.sha256(''.join(sorted(probe_ids)).encode()).hexdigest()[:16]}"
    if outbox.exists():
        import shutil

        shutil.rmtree(outbox)
    outbox.mkdir(parents=True, exist_ok=True)
    for _probe, payload, _definition, filename in members:
        (outbox / filename).write_bytes(payload)    # the built bytes, not a re-render
    (outbox / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2) + "\n")
    (outbox / README_NAME).write_text(render_repo_readme(manifest, {f: d for _p, _b, d, f in members}))
    api.upload_folder(folder_path=str(outbox), repo_id=repo_id, repo_type="model",
                      commit_message=f"probe family — {len(members)} probe(s)")
    revision = _resolve_revision(api, repo_id)
    now = utc_now().isoformat()
    for probe, _payload, _definition, filename in members:
        published = list(probe.published or [])
        published.append({"repo_id": repo_id, "revision": revision, "path": filename, "private": bool(private),
                          "at": now, "sha256": probe.definition_sha256,
                          **{key: identities[probe.id].get(key) for key in IDENTITY_FIELDS}})
        probe.published = published
    db.commit()
    logger.info("probe family of %d published to %s (private=%s, token=%s)", len(members), repo_id, private,
                _redact(token))
    return {"status": "published", "repo_id": repo_id, "revision": revision,
            "files": [{"probe_id": p.id, "path": f} for p, _b, _d, f in members],
            "probes_in_repo": len(manifest["probes"])}


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
