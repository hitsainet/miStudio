"""Assemble a `mistudio.probe-definition/v1` from 032's rows (033 FR-4, FR-5, FR-6, BR-016).

⚠ THE TEST VECTORS GO THROUGH `forward_scores`, NOT THROUGH A COPY OF IT. That is the whole value
of the vectors: a consumer that reproduces them has reproduced the code path that produced this
probe's published metrics. A second scoring implementation here would be a second detector, and
the parity check would then be measuring the wrong one — miLLM would agree with a function
miStudio never used.

⚠ AND THE SAMPLE IS SEEDED, BALANCED, AND SPANS EVERY EVALUATION SET. An unseeded sample makes two
builds of the same probe produce different vectors, so a consumer cannot tell "the implementation
disagrees" from "the sample changed". An unbalanced one lets a probe that only ever fires pass a
parity check on positives alone. And a sample drawn from one evaluation set checks parity on one
distribution, which is exactly the case where an implementation difference hides.
"""
from __future__ import annotations

import hashlib
import json
import logging
import random
from datetime import timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..core.clock import utc_now
from ..core.config import settings
from ..schemas.probe_definition import (
    MAX_DEFINITION_BYTES,
    MAX_TEST_VECTORS,
    MAX_VECTOR_TOKENS,
    MIN_TEST_VECTORS,
    Acknowledgement,
    Aggregation,
    DatasetRef,
    Decision,
    Evidence,
    EvaluationEntry,
    JudgeEvidence,
    ModelIdentity,
    ProbeDefinitionV1,
    ProbeHead as ProbeHeadSpec,
    Provenance,
    ReadPoint,
    SaeReference,
    TestVector,
    TestVectors,
)

logger = logging.getLogger(__name__)

#: How many rows a definition carries by default. The contract allows 8–32; 16 is enough for a
#: consumer to catch a basis or normalisation error on both classes and several distributions,
#: and small enough that the document stays well under the 2 MB cap.
DEFAULT_VECTOR_COUNT = 16

#: The suggested absolute score tolerance. fp16 versus bf16 at `resid_post` differs by about 1.5%
#: relative on this estate, and batch composition moves a score slightly, so exact equality would
#: report a mismatch against a CORRECT implementation. Measured in 033 acceptance 8.1 and agreed
#: with miLLM 024; a builder may override it from that measurement.
DEFAULT_TOLERANCE = 0.05


class ProbeExportRefused(Exception):
    """A build the evidence, the configuration or the caps do not permit.

    Carries `status` so the endpoint maps it without re-deciding: re-deriving the code from the
    message is how a 422 becomes a 500.
    """

    def __init__(self, message: str, status: int = 422) -> None:
        super().__init__(message)
        self.status = status


def sample_vectors(
    rows: Sequence[Dict[str, Any]],
    *,
    count: int = DEFAULT_VECTOR_COUNT,
    seed: int = 1337,
) -> List[Dict[str, Any]]:
    """A balanced, seeded sample with every source set represented (FR-4).

    `rows` are dicts carrying at least `label` (1/0) and `dataset` (the set's name). Returns the
    chosen rows, in a deterministic order.

    THE ORDER OF PRIORITIES IS DELIBERATE, and each one is a failure it prevents:

      1. **One row from every set first.** A sample drawn proportionally would omit the smallest
         set entirely, and the smallest set is where an implementation difference hides — it is
         the one whose distribution is least like the training data.
      2. **Then balance the classes.** A parity check on positives alone passes for a probe that
         fires on everything.
      3. **Then fill deterministically.** Same seed, same rows, so a consumer comparing two
         builds is comparing implementations rather than samples.
    """
    if count < MIN_TEST_VECTORS or count > MAX_TEST_VECTORS:
        raise ProbeExportRefused(
            f"{count} test vectors is outside the contract's {MIN_TEST_VECTORS}–"
            f"{MAX_TEST_VECTORS} range"
        )
    if not rows:
        raise ProbeExportRefused(
            "there are no evaluation rows to sample test vectors from; the probe has not been "
            "evaluated, so an exported definition would carry no parity check",
            status=409,
        )

    rng = random.Random(seed)
    ordered = sorted(rows, key=lambda row: (str(row.get("dataset")), int(row.get("index", 0))))

    by_set: Dict[str, List[Dict[str, Any]]] = {}
    for row in ordered:
        by_set.setdefault(str(row.get("dataset")), []).append(row)

    chosen: List[Dict[str, Any]] = []
    taken = set()

    def take(row: Dict[str, Any]) -> None:
        key = (str(row.get("dataset")), int(row.get("index", 0)))
        if key in taken:
            return
        taken.add(key)
        chosen.append(row)

    # 1 — every set represented, and each set's own contribution balanced where it can be.
    for name in sorted(by_set):
        pool = by_set[name]
        for wanted in (1, 0):
            candidates = [row for row in pool if int(row.get("label", 0)) == wanted]
            if candidates and len(chosen) < count:
                take(rng.choice(candidates))

    # 2 — balance the classes overall.
    while len(chosen) < count:
        positives = sum(1 for row in chosen if int(row.get("label", 0)) == 1)
        negatives = len(chosen) - positives
        wanted = 1 if positives <= negatives else 0
        remaining = [
            row
            for row in ordered
            if int(row.get("label", 0)) == wanted
            and (str(row.get("dataset")), int(row.get("index", 0))) not in taken
        ]
        if not remaining:
            # 3 — one class is exhausted; fill from whatever is left rather than returning short.
            remaining = [
                row
                for row in ordered
                if (str(row.get("dataset")), int(row.get("index", 0))) not in taken
            ]
            if not remaining:
                break
        take(rng.choice(remaining))

    if len(chosen) < MIN_TEST_VECTORS:
        raise ProbeExportRefused(
            f"only {len(chosen)} evaluation rows are available and the contract requires at "
            f"least {MIN_TEST_VECTORS} test vectors",
            status=409,
        )
    return sorted(chosen, key=lambda row: (str(row.get("dataset")), int(row.get("index", 0))))


def truncate_tokens(token_ids: Sequence[int], limit: int = MAX_VECTOR_TOKENS) -> List[int]:
    """Truncate BEFORE scoring, keeping the END of the row.

    ⚠ THE TAIL, NOT THE HEAD. `last` reads the final scored token and `rolling_mean_max` weights
    the end; truncating from the right would change what those rules see, so a consumer scoring
    the truncated row would legitimately disagree with a vector scored on the full one. Keeping
    the tail means the recorded score is the score OF THE ROW IN THE FILE.
    """
    tokens = list(token_ids)
    if len(tokens) <= limit:
        return tokens
    return tokens[-limit:]


def dataset_ref_from_view(db: Any, view: Any) -> DatasetRef:
    """`{hf_id, config, split, revision}` from a probe dataset view and its dataset row.

    ⚠ NEVER A LOCAL PATH. A definition naming `/data/datasets/...` is unusable off this box and
    leaks the filesystem layout; the contract refuses one, and this is where it would come from.

    ⚠ IT TAKES THE SESSION BECAUSE THERE IS NO `view.dataset` RELATIONSHIP, and the first version
    of this function assumed one. `ProbeMonitorDataset` has a `dataset_id` FOREIGN KEY and no
    ORM relationship beside it, so `getattr(view, "dataset", None)` returned None and every build
    refused with "has no HuggingFace identifier, only a local path" — over a `Dataset` row whose
    `hf_repo_id` is `Arrrlex/models-under-pressure`, sitting one query away. Three of the four
    Stage 2 acceptance builds died on it.

    The refusal was at least precise enough to diagnose in one read, which is the argument for
    naming what a refusal looked at.
    """
    from ..models.dataset import Dataset

    dataset = (
        db.query(Dataset).filter(Dataset.id == view.dataset_id).first()
        if getattr(view, "dataset_id", None) is not None
        else None
    )
    metadata = (getattr(dataset, "extra_metadata", None) or {}) if dataset else {}
    # ⚠ There was an `or getattr(view, "hf_id", None)` here. `ProbeMonitorDataset` has no such
    # field, so the fallback could never fire — a second source that reads as resilience and
    # is decoration. The dataset row is the only place a HuggingFace id is recorded.
    hf_id = getattr(dataset, "hf_repo_id", None)
    if not hf_id:
        raise ProbeExportRefused(
            f"probe dataset {getattr(view, 'id', '?')} has no HuggingFace identifier, only a "
            f"local path; a definition must name a dataset a consumer can fetch"
        )
    return DatasetRef(
        hf_id=hf_id,
        config=getattr(view, "config", None),
        split=getattr(view, "split", None),
        # `Dataset` has no `revision` column; a download records it in the metadata blob when the
        # source pinned one. None is honest here — the contract makes it optional precisely
        # because a dataset is often taken from a moving branch.
        revision=metadata.get("revision") if isinstance(metadata, dict) else None,
    )


def serialise(definition: ProbeDefinitionV1) -> str:
    """THE on-disk form of a definition. One function, so nothing can describe a different one.

    ⚠ THREE THINGS USED TO SERIALISE THE DOCUMENT INDEPENDENTLY AND TWO OF THEM WERE WRONG.
    `check_size` and `sha256_of` both called `model_dump_json()` — compact — while the writer called
    `model_dump_json(indent=2)`. So for the first definition this estate built:

        recorded sha256   b0f9f404…   the compact form's
        file's sha256     60a93126…   the bytes actually on disk
        recorded bytes      308,552   the compact form's
        file's bytes        491,721   the bytes actually on disk

    **The integrity pin did not describe the file it pinned.** A consumer checking the document
    against the sha in its own manifest — the thing a checksum is FOR — would reject a perfectly
    good file, and the publisher ships that sha in the README and the manifest entry. A checksum
    that is always wrong is worse than none, because it turns "this file is intact" into noise the
    first reader learns to ignore.

    And `check_size`'s docstring said "measured on what a consumer downloads" while measuring
    something else, 1.6x smaller. A document 1.3 MB compact and 2.1 MB indented passed a 2 MB cap
    and landed over it.

    Indented is the right choice to standardise on — a contract document people read in a diff —
    so the fix is to measure and hash what is written, not to write what was measured.
    """
    return definition.model_dump_json(indent=2)


def check_size(definition: ProbeDefinitionV1) -> int:
    """Refuse a document over the cap, measured on what a consumer downloads (FR-2, 2.3)."""
    payload = serialise(definition)
    size = len(payload.encode("utf-8"))
    if size > MAX_DEFINITION_BYTES:
        raise ProbeExportRefused(
            f"the definition serialises to {size:,} bytes, over the "
            f"{MAX_DEFINITION_BYTES:,}-byte cap. Reduce the test-vector count or their length — "
            f"{len(definition.test_vectors.vectors)} vectors of up to {MAX_VECTOR_TOKENS} tokens "
            f"are the usual cause"
        )
    return size


def sha256_of(definition: ProbeDefinitionV1) -> str:
    """The sha256 of the bytes `write_definition` writes — see `serialise`."""
    return hashlib.sha256(serialise(definition).encode("utf-8")).hexdigest()


def write_definition(definition: ProbeDefinitionV1, path: Path) -> Tuple[int, str]:
    """Write the document atomically and return `(bytes, sha256)` OF WHAT WAS WRITTEN.

    The size and the digest are measured on `payload` here rather than recomputed from the model,
    so the row, the task result, the manifest and the file cannot disagree even if `serialise`
    changes. That is the whole point: the caller cannot accidentally describe a different
    serialisation, because it is never handed the model to serialise again.

    Written to a temporary name and moved, so a crash mid-write cannot leave a half-document that
    parses as far as the truncation and then fails — the shape that makes a corrupt export look
    like a contract violation.
    """
    payload = serialise(definition).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_suffix(".json.partial")
    staging.write_bytes(payload)
    staging.replace(path)
    return len(payload), hashlib.sha256(payload).hexdigest()


def check_export_gate(
    probe: Any,
    run: Any,
    *,
    acknowledge_below_rung2: Optional[Dict[str, Any]] = None,
    sae_row: Any = None,
    actor: str = "operator",
) -> Optional[Acknowledgement]:
    """The refusals, IN THE ORDER FR-5/FTID §5 fixes them (task 2.4).

    ⚠ THE ORDER IS PART OF THE CONTRACT, not an implementation detail. Each check answers a
    different question and the answers stop being useful if they arrive in the wrong sequence:
    telling someone their evidence is too thin, when the real problem is that the run is still
    going, sends them to do the wrong work. So:

      1. gone            → 404, nothing else can be said about it
      2. SAE, no HF home → 422, unfixable by acknowledging anything
      3. not ready       → 409, transient; retry after it finishes
      4. rung / ack      → 422, a judgement the caller can make
      5. still running   → 409, transient

    Returns the `Acknowledgement` to embed, or None when the rung did not need one.
    """
    from ..schemas.evidence_ladder import probe_rung_language

    # 1 — gone
    if probe is None:
        raise ProbeExportRefused("probe not found", status=404)

    # 2 — an SAE probe whose dictionary has no published home
    if getattr(probe, "variant", "dense") == "sae":
        if sae_row is None:
            raise ProbeExportRefused(
                f"probe {probe.id} is a k-sparse SAE probe whose SAE row is gone, so the "
                f"dictionary it reads cannot be named in the definition"
            )
        if not (getattr(sae_row, "hf_repo_id", None) and getattr(sae_row, "hf_filepath", None)):
            raise ProbeExportRefused(
                f"probe {probe.id} reads SAE {sae_row.id}, which has no HuggingFace location. A "
                f"consumer cannot encode without the dictionary, so publish the SAE first — this "
                f"is not something an acknowledgement can waive"
            )

    # 3 — the probe's own artefacts
    if not getattr(probe, "weights_path", None):
        raise ProbeExportRefused(
            f"probe {probe.id} has no weights on disk; its run did not finish training",
            status=409,
        )

    # 4 — the evidence gate
    rung = int(getattr(probe, "rung", 0) or 0)
    if rung < 2:
        if not acknowledge_below_rung2:
            raise ProbeExportRefused(
                f"probe {probe.id} is at rung {rung} — \"{probe_rung_language(rung)}\" — which is "
                f"below the rung 2 an export claims. Re-send with "
                f"`acknowledge_below_rung2: {{\"reason\": \"…\"}}` recording who accepts that and "
                f"why; the acknowledgement travels inside the definition"
            )
        reason = str(acknowledge_below_rung2.get("reason") or "").strip()
        if len(reason) < 10:
            raise ProbeExportRefused(
                "the acknowledgement needs a reason of at least 10 characters; \"ok\" records "
                "nothing, and the point of the field is that a probe served on thin evidence "
                "carries the name of the person who decided that was acceptable"
            )
        return Acknowledgement(
            by=str(acknowledge_below_rung2.get("by") or actor),
            at=utc_now(),
            reason=reason,
        )

    # 5 — a run still in flight
    status = str(getattr(run, "status", "") or "").lower()
    if status in {"pending", "running", "cancelling"}:
        raise ProbeExportRefused(
            f"run {getattr(run, 'id', '?')} is {status}; its evaluations and rung can still "
            f"change, and a definition built now would state evidence that is about to be "
            f"superseded",
            status=409,
        )
    return None


def resolve_model_revision(model_row: Any) -> str:
    """The model's resolved commit SHA — never a branch name (FR-1).

    ⚠ "main" MOVES, and a probe read from a different revision of the same repo is a probe over a
    different distribution. This estate has already paid for that at the tokenizer level, where
    two LiquidAI models sharing a family name agreed on **0.00%** of token ids and 20 of 22
    extractions silently read the wrong one.

    The revision comes from the HuggingFace cache's snapshot directory name, which IS the commit
    SHA. Refuses rather than writing "main".

    ⚠ IT IS RESOLVED THROUGH `resolve_model_snapshot`, THE LOADER'S OWN FUNCTION, AND THAT IS THE
    POINT. This used to glob the cache itself and take `sorted(...)[-1]`, while the loader takes
    `glob(...)[0]` — two different orderings over the same directory. With one snapshot on disk
    they agree, and every model here has one today, so the disagreement is invisible. With two
    (this estate has already re-downloaded a model and pulled a newer upstream revision) the
    definition would pin one commit while the vectors beside it were scored under another — a file
    that is internally inconsistent and says nothing about it. Deriving the pin from the call that
    loads the weights makes that divergence impossible rather than unlikely.

    ⚠ THE TWO "RECORDED" TIERS THIS FUNCTION USED TO TRY WERE FICTION. It read
    `model_row.hf_revision` and `model_row.metadata_["revision"]`; `Model` has neither column, and
    nothing in this codebase persists a model revision anywhere, so both tiers returned None on
    every call and the docstring's "falls back to" described the only path there was. A tier that
    cannot fire reads as defence in depth and is decoration. If a revision column is added later,
    add a tier here WITH a test that fires it.
    """
    from .activation_service import resolve_model_snapshot

    file_path = getattr(model_row, "file_path", None)
    if file_path:
        try:
            snapshot = Path(resolve_model_snapshot(str(settings.resolve_data_path(file_path))))
        except Exception as exc:  # noqa: BLE001 - any resolution failure is a refusal, with cause
            raise ProbeExportRefused(
                f"model {getattr(model_row, 'id', '?')} has no resolvable snapshot, so no commit "
                f"can be pinned: {exc}"
            ) from exc
        # A cache snapshot's directory name is the commit sha; a flat download's is the model id.
        if snapshot.parent.name == "snapshots":
            return snapshot.name

    raise ProbeExportRefused(
        f"model {getattr(model_row, 'id', '?')} has no resolved revision: no HuggingFace cache "
        f"snapshot was found, and a flat download records no commit. A definition must pin a "
        f"commit, because the same repo at another revision is a different distribution"
    )


def _model_hf_id(model_row: Any) -> str:
    """The model's HuggingFace repo id — `repo_id`, and a refusal when there is none.

    ⚠ THIS READ `model_row.hf_repo_id`, WHICH DOES NOT EXIST ON `Model`. The column is `repo_id`.
    A bare attribute access on a wrong name is an `AttributeError`, so every definition build
    crashed at this line — it was the second of five wrong field names in this file, all of them
    written against hand-built test objects that had whatever attribute the code asked for.

    ⚠ AND THE FALLBACK IT CRASHED PAST WAS WORSE THAN THE CRASH. It was
    `model_row.hf_repo_id or model_row.name`, so a locally-imported model with no repo would have
    put its DISPLAY NAME in the `hf_id` field of a portable contract — a string no consumer can
    fetch, in the one field that tells a consumer what to fetch. A refusal is the honest answer.
    """
    repo_id = getattr(model_row, "repo_id", None)
    if repo_id:
        return str(repo_id)
    raise ProbeExportRefused(
        f"model {getattr(model_row, 'id', '?')} has no HuggingFace repo id, only the display name "
        f"{getattr(model_row, 'name', '?')!r}. A consumer fetches the model by repo id, so a "
        f"display name in that field would be a definition nobody can load"
    )


def chat_template_sha256(tokenizer: Any) -> Optional[str]:
    """Of the tokenizer's chat template, or None when it has none.

    Part of the model's identity: the role mask and every rendered row depend on the template, so
    a model whose template changed renders different text from the same messages. A consumer must
    be able to notice — this estate has seen a template change alter tool-call rendering between
    two revisions of one model.
    """
    template = getattr(tokenizer, "chat_template", None)
    if not template:
        return None
    return hashlib.sha256(str(template).encode("utf-8")).hexdigest()


def sae_reference(sae_row: Any, feature_indices: Sequence[int], labels: Optional[Sequence[str]] = None) -> SaeReference:
    """The `sae` block — location, revision, integrity and NORMALISATION (FR-6).

    ⚠ THE NORMALISATION IS NOT DECORATION. `encode_with_training_normalization` exists because
    encoding with a bare `encode()` reads the right weights in the wrong basis, produces plausible
    features, and is invisible in the numbers (MIS-E2E-083). A consumer that cannot reproduce the
    normalisation cannot reproduce the probe, so the mode and its constants travel in the file.
    """
    weights_sha = _sae_weights_sha256(sae_row)
    normalization = _sae_normalization(sae_row)
    return SaeReference(
        hf_repo=sae_row.hf_repo_id,
        path=sae_row.hf_filepath,
        revision=str(getattr(sae_row, "hf_revision", None) or "main"),
        weights_sha256=weights_sha,
        architecture=str(getattr(sae_row, "architecture", None) or "unknown"),
        d_model=int(sae_row.d_model),
        n_features=int(sae_row.n_features),
        normalization=normalization,
        feature_indices=sorted(int(index) for index in feature_indices),
        feature_labels=list(labels) if labels else None,
    )


def _sae_normalization(sae_row: Any) -> Dict[str, Any]:
    """The normalisation the SAE was TRAINED with, read from where it is actually recorded.

    ⚠ THIS WAS THE MOST DANGEROUS OF THE FIVE WRONG FIELD NAMES, AND THE HARDEST TO SEE.
    It read `getattr(sae_row, "normalize_activations", None) or "constant_norm_rescale"`.
    `ExternalSAE` has no such column — the value lives at
    `sae_metadata["training_hyperparameters"]["normalize_activations"]` — so the getattr returned
    None on every SAE and the `or` default fired every time.

    For the SAE this was found on, the default happened to be RIGHT: its recorded mode is
    `constant_norm_rescale`. That is precisely why nothing caught it. Export an SAE trained with
    `none` or `anthropic_rescale` and the definition would state `constant_norm_rescale` with
    complete confidence, and the consumer would encode in the wrong basis — plausible features,
    wrong meaning, invisible in every number. That is MIS-E2E-083, reintroduced inside the field
    whose own docstring warns about it.

    ⚠ AN UNRECORDED MODE IS NOW A REFUSAL, WHICH IS A DELIBERATE TIGHTENING. A downloaded SAE
    whose publisher recorded nothing cannot be exported until someone records what it used. The
    alternative is a file that guesses, and a guess in this field is indistinguishable from a fact
    to every consumer that reads it. `source` travels beside the mode so a reader can see which
    record answered.
    """
    metadata = getattr(sae_row, "sae_metadata", None) or {}
    hyperparameters = metadata.get("training_hyperparameters") or {} if isinstance(metadata, dict) else {}

    mode: Optional[str] = None
    source: Optional[str] = None
    # ⚠ No `getattr(sae_row, "normalize_activations", ...)` tier here, deliberately. That read is
    # the defect this function exists to fix, and a tier for a column that does not exist is the
    # decoration the docstring above objects to. If the column is ever added, add a tier WITH a
    # test that fires it — `test_probe_definition_reads_real_columns.py` will accept it then.
    for candidate, label in (
        (hyperparameters.get("normalize_activations") if isinstance(hyperparameters, dict) else None,
         "sae_metadata.training_hyperparameters"),
        (metadata.get("normalize_activations") if isinstance(metadata, dict) else None,
         "sae_metadata"),
    ):
        if candidate:
            mode, source = str(candidate), label
            break

    if not mode:
        raise ProbeExportRefused(
            f"SAE {getattr(sae_row, 'id', '?')} records no activation-normalisation mode, so a "
            f"definition cannot state one. Guessing is not available here: a consumer that "
            f"normalises differently from the training run encodes in the wrong basis, which "
            f"produces plausible features and is invisible in every metric. Record the mode on "
            f"the SAE (its training hyperparameters) and export again"
        )

    normalization: Dict[str, Any] = {"mode": mode, "source": source}
    for field in ("normalization_constant", "activation_norm_target", "norm_scale"):
        value = getattr(sae_row, field, None)
        if value is None and isinstance(hyperparameters, dict):
            value = hyperparameters.get(field)
        if value is not None:
            normalization[field] = value
    return normalization


def _sae_weights_sha256(sae_row: Any) -> str:
    """Hash the dictionary's weights so a consumer can prove it fetched the same file.

    Computed from the local copy — the one this probe was actually trained against. Refuses when
    there is none, rather than emitting a placeholder: a wrong integrity hash is worse than no
    export, because it makes a mismatch look like corruption at the consumer's end.
    """
    local = getattr(sae_row, "local_path", None)
    if not local:
        raise ProbeExportRefused(
            f"SAE {sae_row.id} has no local copy to hash; the definition's `weights_sha256` would "
            f"be a guess, and a wrong integrity hash makes a mismatch look like corruption at the "
            f"consumer's end"
        )
    from pathlib import Path

    path = settings.resolve_data_path(local)
    candidates = [path] if path.is_file() else sorted(path.glob("**/*.safetensors"))
    if not candidates:
        raise ProbeExportRefused(f"no SAE weights file found under {path}")
    digest = hashlib.sha256()
    with open(candidates[0], "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build(
    db: Any,
    probe_id: str,
    *,
    acknowledge_below_rung2: Optional[Dict[str, Any]] = None,
    actor: str = "operator",
    vector_count: int = DEFAULT_VECTOR_COUNT,
    seed: int = 1337,
    tolerance: float = DEFAULT_TOLERANCE,
    model_loader: Optional[Any] = None,
) -> Tuple[ProbeDefinitionV1, Dict[str, Any]]:
    """Assemble the definition, scoring its test vectors through 032's `forward_scores`.

    Returns `(definition, build_record)`. The record is what goes in `definition_build`: what was
    asked for and what was resolved, so the export is reproducible from the row.

    `model_loader` is injected by the tests, exactly as 032's run does — the production loader
    requires CUDA on purpose, because vectors scored on CPU activations of a quantized model are
    not the vectors a consumer will reproduce.
    """
    import torch

    from ..models.external_sae import ExternalSAE
    from ..models.model import Model
    from ..models.probe_monitor import (
        ProbeMonitor,
        ProbeMonitorDataset,
        ProbeMonitorEvaluation,
        ProbeMonitorJudgeRun,
        ProbeMonitorRun,
    )
    from ..schemas.evidence_ladder import probe_rung_language
    from .probe_monitor_capture import forward_scores
    from .probe_monitor_run import (
        _pad_id,
        _prepare_examples,
        _probe_device,
        context_from_row,
        load_probe,
        sae_encoder_for,
    )

    probe = db.query(ProbeMonitor).filter(ProbeMonitor.id == probe_id).first()
    run = (
        db.query(ProbeMonitorRun).filter(ProbeMonitorRun.id == probe.run_id).first()
        if probe is not None
        else None
    )
    sae_row = (
        db.query(ExternalSAE).filter(ExternalSAE.id == probe.sae_id).first()
        if probe is not None and probe.sae_id
        else None
    )
    # ⚠ THE RUNG IS RECOMPUTED FROM THE EVIDENCE BEFORE IT IS GATED ON, BY THE LADDER'S OWN
    # FUNCTION. `probe.rung` is a cache of a decision the ladder makes from the evaluation rows,
    # and a cache can be stale: `pm_8428015843e3` sat at rung 0 with five complete
    # out-of-distribution evaluations because its run raised on a later probe and never reached the
    # stage that promotes rungs. The gate then refused it for lacking evidence it had, and an
    # acknowledged export would have written "rung 0 — trained" above those five AUROCs.
    #
    # Asking `recompute_rung` rather than reimplementing the ladder here is the point. The rung is
    # a function of `ci_low`, `out_of_distribution` and `judge_completed` per set; a second copy of
    # that arithmetic in the exporter would agree today and drift, and the disagreement would show
    # up as a document that contradicts the UI. One authority, consulted.
    if probe is not None:
        from .probe_monitor_run import recompute_rung

        recompute_rung(db, probe.id)
        db.refresh(probe)

    acknowledgement = check_export_gate(
        probe,
        run,
        acknowledge_below_rung2=acknowledge_below_rung2,
        sae_row=sae_row,
        actor=actor,
    )

    model_row = db.query(Model).filter(Model.id == run.model_id).first()
    if model_row is None:
        raise ProbeExportRefused(
            f"model {run.model_id} is gone, so this probe's identity cannot be pinned", status=409
        )

    context = context_from_row(run)
    head, trained = load_probe(db, probe_id)

    loader = model_loader
    if loader is None:
        from .probe_monitor_run import _load_model_for_run

        loader = _load_model_for_run
    model, tokenizer, architecture = loader(run)

    # ── the evaluation rows, and a sample of them ─────────────────────────────
    evaluations = (
        db.query(ProbeMonitorEvaluation)
        .filter(ProbeMonitorEvaluation.probe_id == probe_id)
        .order_by(ProbeMonitorEvaluation.created_at.asc())
        .all()
    )
    views = {
        view.id: view
        for view in db.query(ProbeMonitorDataset).filter(
            ProbeMonitorDataset.id.in_([e.dataset_id for e in evaluations] or [""])
        )
    }

    candidates: List[Dict[str, Any]] = []
    rendered_by_key: Dict[Tuple[str, int], Any] = {}
    for evaluation in evaluations:
        view = views.get(evaluation.dataset_id)
        if view is None:
            continue
        examples, rendered, _summary, labels = _prepare_examples(
            db, view, context, tokenizer, role="eval"
        )
        for index, (example, label) in enumerate(zip(rendered, labels)):
            candidates.append(
                {"dataset": view.config or view.id, "index": index, "label": int(label)}
            )
            rendered_by_key[(view.config or view.id, index)] = (example, view)

    sampled = sample_vectors(candidates, count=vector_count, seed=seed)

    # ── score them through the ONE scoring path ───────────────────────────────
    encoder = None
    if probe.variant == "sae":
        encoder = sae_encoder_for(
            db, probe.sae_id, probe.sae_feature_indices, device=_probe_device(model)
        )

    chosen_examples = []
    for row in sampled:
        example, _view = rendered_by_key[(row["dataset"], row["index"])]
        example.input_ids = truncate_tokens(example.input_ids)
        example.token_roles = list(example.token_roles)[-len(example.input_ids):]
        example.token_message = list(example.token_message)[-len(example.input_ids):]
        chosen_examples.append(example)

    scored = forward_scores(
        model,
        chosen_examples,
        head,
        rule=trained.rule,
        rule_params=trained.rule_params,
        scope=context.scope,
        architecture=architecture,
        pad_id=_pad_id(tokenizer),
        encoder=encoder,
    )

    vectors = [
        TestVector(
            messages=_messages_of(example),
            token_ids=list(example.input_ids),
            token_scores=[float(value) for value in row.token_scores],
            score=float(row.aggregate),
            verdict=(
                None if probe.threshold is None else bool(row.aggregate >= probe.threshold)
            ),
        )
        for example, row in zip(chosen_examples, scored)
    ]

    # ── evidence, verbatim from the rows ─────────────────────────────────────
    entries: List[EvaluationEntry] = []
    for evaluation in evaluations:
        metrics = evaluation.metrics or {}
        if not metrics.get("scored"):
            continue                        # a refusal is not evidence of detection
        view = views.get(evaluation.dataset_id)
        if view is None:
            continue
        one_percent = next(
            (
                point
                for point in metrics.get("operating_points") or []
                if abs(float(point.get("target_fpr", 0)) - float(probe.target_fpr or 0.01)) < 1e-9
            ),
            None,
        )
        ci = metrics.get("ci") or {}
        entries.append(
            EvaluationEntry(
                dataset=dataset_ref_from_view(db, view),
                distribution=(
                    "out_of_distribution"
                    if metrics.get("out_of_distribution")
                    else "in_distribution"
                ),
                n_positive=int(metrics.get("n_positive", 0)),
                n_negative=int(metrics.get("n_negative", 0)),
                auroc=float(metrics["auroc"]),
                auroc_ci=(
                    [float(ci["low"]), float(ci["high"])]
                    if ci.get("low") is not None and ci.get("high") is not None
                    else None
                ),
                recall_at_target_fpr=(
                    float(one_percent["recall"]) if one_percent else None
                ),
            )
        )

    judge_run = (
        db.query(ProbeMonitorJudgeRun)
        .filter(
            ProbeMonitorJudgeRun.probe_id == probe_id,
            ProbeMonitorJudgeRun.status == "completed",
        )
        .order_by(ProbeMonitorJudgeRun.created_at.desc())
        .first()
    )
    judge = None
    if judge_run is not None:
        metrics = judge_run.metrics or {}
        judge = JudgeEvidence(
            model=str(judge_run.model),
            prompt_version=str(judge_run.prompt_version),
            per_set_auroc=_judge_per_set_auroc(metrics),
        )

    train_view = (
        db.query(ProbeMonitorDataset)
        .filter(ProbeMonitorDataset.id == run.train_dataset_id)
        .first()
    )

    definition = ProbeDefinitionV1(
        name=_slug(probe, run),
        description=f"{trained.rule} probe over layer {probe.layer} of {model_row.name}",
        concept=_concept_of(train_view),
        model=ModelIdentity(
            hf_id=_model_hf_id(model_row),
            revision=resolve_model_revision(model_row),
            d_model=_d_model_of(model_row, head),
            n_layers=_n_layers_of(model_row),
            architecture=str(model_row.architecture or architecture or "unknown"),
            chat_template_sha256=chat_template_sha256(tokenizer),
            mistudio_model_id=model_row.id,
        ),
        read=ReadPoint(layer=int(probe.layer), hook_point="resid_post"),
        scope=_contract_scope(context.scope),
        basis="sae_features" if probe.variant == "sae" else "residual",
        head=ProbeHeadSpec(
            weights=[float(value) for value in head.weight.tolist()],
            bias=float(head.bias),
            norm_mean=[float(value) for value in head.mean.tolist()],
            norm_std=[float(value) for value in head.std.tolist()],
            attention_query=(
                [float(value) for value in head.attention_query.tolist()]
                if head.attention_query is not None
                else None
            ),
        ),
        sae=(
            sae_reference(sae_row, probe.sae_feature_indices or [])
            if probe.variant == "sae"
            else None
        ),
        aggregation=Aggregation(
            rule=trained.rule,
            params=dict(trained.rule_params or {}),
            streamable=bool(probe.streamable),
        ),
        decision=Decision(
            threshold=probe.threshold,
            target_fpr=probe.target_fpr,
            realised_fpr=probe.realised_fpr,
            threshold_source=probe.threshold_source,
            calibration=(
                dataset_ref_from_view(
                    db,
                    db.query(ProbeMonitorDataset)
                    .filter(ProbeMonitorDataset.id == run.calibration_dataset_id)
                    .first()
                )
                if run.calibration_dataset_id
                else None
            ),
        ),
        evidence=Evidence(
            rung=int(probe.rung or 0),
            rung_language=probe_rung_language(int(probe.rung or 0)),
            acknowledgement=acknowledgement,
            evaluations=entries,
            judge=judge,
        ),
        provenance=Provenance(
            train_dataset=dataset_ref_from_view(db, train_view) if train_view else None,
            label_mapping=dict(train_view.label_mapping or {}) if train_view else {},
            keyword_filter=(train_view.keyword_filter if train_view else None),
            split=(train_view.split if train_view else None),
            run_id=run.id,
            probe_id=probe.id,
            created_at=probe.created_at,
            exported_at=utc_now(),
            mistudio_version=_mistudio_version(),
        ),
        test_vectors=TestVectors(
            tolerance=tolerance,
            # ⚠ MEASURED, NOT ASSERTED. The exporter re-renders its own `messages` and compares the
            # ids, because on this corpus they do NOT round-trip and a consumer that starts from
            # them is off by up to 1.153 on a 0.05 tolerance — see `TestVectors`' docstring.
            messages_reproduce_token_ids=_messages_round_trip(tokenizer, vectors, context.max_length),
            vectors=vectors,
        ),
    )

    size = check_size(definition)
    record = {
        "requested": {
            "vector_count": vector_count,
            "seed": seed,
            "tolerance": tolerance,
            "acknowledged": acknowledgement is not None,
        },
        "resolved": {
            "model_revision": definition.model.revision,
            "vectors": len(vectors),
            "sets_sampled": sorted({row["dataset"] for row in sampled}),
            "bytes": size,
            "sha256": sha256_of(definition),
            "basis": definition.basis,
        },
        "built_at": utc_now().isoformat(),
    }
    return definition, record


def _judge_per_set_auroc(metrics: Dict[str, Any]) -> Dict[str, float]:
    """The judge's AUROC per set, from where `probe_monitor_judge` actually writes it.

    ⚠ THIS READ `metrics["per_set_auroc"]`, A KEY THE JUDGE NEVER WRITES. The judge writes
    `metrics["per_set"][<dataset id>]` — a dict per set carrying `auroc`, `ci`, `roc`,
    `operating_points`, counts — so the lookup found nothing and `per_set_auroc` was `{}` on every
    export. The sixth wrong field name in this feature, and the worst-placed of them.

    **The document claimed rung 3 — "detects on unseen tasks, compared with a judge" — and carried
    no comparison.** The rung's entire justification is the judge, and the block asserting it was
    empty while validating perfectly, because an empty dict is a legal value for the field. The real
    numbers were sitting in the judge run all along: 0.8827, 0.8189, 0.9895, 0.9553, 0.7255.

    An absent `auroc` on a set is skipped rather than defaulted: a set the judge could not score
    belongs missing from this map, not present as 0.0, which a reader would take for a judge that
    performed at chance.
    """
    per_set = metrics.get("per_set")
    if not isinstance(per_set, dict):
        return {}
    resolved: Dict[str, float] = {}
    for name, entry in per_set.items():
        value = entry.get("auroc") if isinstance(entry, dict) else None
        if value is None:
            continue
        resolved[str(name)] = float(value)
    return resolved


def _messages_round_trip(tokenizer: Any, vectors: Sequence[Any], max_length: int) -> Optional[bool]:
    """Do the vectors' own `messages` re-render to their `token_ids`? Measured, never assumed.

    ⚠ ON THIS ESTATE THEY DO NOT, AND THE DOCUMENT NOW SAYS SO. The corpus is plain prose, so
    `_messages_of` wraps each row in a single user turn to give a consumer something sendable;
    re-rendering that through the chat template adds template scaffolding the scored row never had.
    Measured in 033 acceptance 8.1: six extra tokens, first difference at index 2, and scores off by
    up to 1.153 against a 0.05 tolerance — on every one of sixteen vectors.

    A consumer told nothing would take `messages`, score them, compare, fail, and conclude its own
    implementation was wrong. This turns that into a field it can read first.

    Returns None rather than False when the check itself cannot run: "not checked" and "checked and
    it does not round-trip" are different claims, and a consumer may reasonably treat them
    differently.
    """
    from .probe_monitor_render import render_messages

    if tokenizer is None:
        return None
    try:
        for vector in vectors:
            rendered = render_messages(
                tokenizer, [dict(message) for message in vector.messages], max_length=max_length
            )
            if truncate_tokens(list(rendered.input_ids)) != list(vector.token_ids):
                return False
        return True
    except Exception as exc:  # noqa: BLE001 - an unrenderable reconstruction is "not checked"
        logger.warning("could not check whether the test vectors' messages round-trip: %s", exc)
        return None


def _messages_of(example: Any) -> List[Dict[str, Any]]:
    """The rendered row as messages. Falls back to a single user turn for plain prose, which is
    what the training corpus is — a definition still has to carry SOMETHING a consumer can send."""
    messages = getattr(example, "messages", None)
    if messages:
        return [dict(message) for message in messages]
    return [{"role": "user", "content": getattr(example, "text", "") or ""}]


def _slug(probe: Any, run: Any) -> str:
    return f"{run.id}-{probe.id}-{probe.rule}-L{probe.layer}"[:200]


def _concept_of(view: Any) -> Optional[str]:
    """What "positive" means, read from the label mapping rather than written by hand."""
    if view is None or not view.label_mapping:
        return None
    positives = sorted(
        label for label, target in (view.label_mapping or {}).items() if target == "positive"
    )
    if not positives:
        return None
    return f"positive = {', '.join(positives)}"


def _contract_scope(scope: str) -> str:
    """032's scope vocabulary mapped onto the contract's.

    032 has `all | assistant | user | last_assistant`; the contract has `all | prompt | response`.
    The mapping is deliberate and lossy in one direction only — `last_assistant` narrows to
    `response`, which is the honest superset, because a consumer told `response` scores more
    tokens than the probe was trained on rather than fewer. Refusing would be the alternative;
    widening is safe, narrowing would not be.
    """
    return {
        "all": "all",
        "user": "prompt",
        "assistant": "response",
        "last_assistant": "response",
    }.get(scope, "all")


def _d_model_of(model_row: Any, head: Any) -> int:
    config = getattr(model_row, "architecture_config", None) or {}
    for key in ("hidden_size", "d_model", "n_embd"):
        if config.get(key):
            return int(config[key])
    raise ProbeExportRefused(
        f"model {model_row.id} records no hidden size, so the definition cannot state d_model. "
        f"Refusing rather than inferring it from the head, which would be the SAE's k on a "
        f"k-sparse probe and silently wrong"
    )


def _n_layers_of(model_row: Any) -> int:
    config = getattr(model_row, "architecture_config", None) or {}
    for key in ("num_hidden_layers", "n_layers", "num_layers"):
        if config.get(key):
            return int(config[key])
    raise ProbeExportRefused(
        f"model {model_row.id} records no layer count, so `layer < n_layers` cannot be checked"
    )


def _mistudio_version() -> Optional[str]:
    try:
        from ..api.v1.endpoints.version import _read_version

        return _read_version()
    except Exception:  # noqa: BLE001 - a missing version must not block an export
        return None


# ── the invalidation hook (task 2.5) ──────────────────────────────────────────

#: The fields whose change makes a cached definition a LIE. Not "any change": a probe's
#: `selected` flag or its name moving does not alter what the document states.
INVALIDATING_FIELDS = ("threshold", "target_fpr", "realised_fpr", "threshold_source", "rung")


def invalidate_definition(db: Any, probe_id: str, *, reason: str) -> bool:
    """Drop a probe's cached definition because what it states has changed (FR-7, task 2.5).

    ⚠ A CACHED DEFINITION IS A CLAIM WITH A TIMESTAMP, and 032 keeps changing the things it
    claims. `recompute_rung` runs after every evaluation AND after every judge run; calibration
    rewrites the threshold. A document built before either still parses, still validates, and
    states an operating point or an evidence rung that is no longer true — which is worse than
    having no export, because a consumer has no way to tell.

    Returns whether anything was cleared, so a caller can log it. Deliberately does NOT rebuild:
    a rebuild is a GPU job, and doing one implicitly inside an evaluation's commit path would
    queue work nobody asked for.
    """
    from ..models.probe_monitor import ProbeMonitor

    probe = db.query(ProbeMonitor).filter(ProbeMonitor.id == probe_id).first()
    if probe is None or not probe.definition_path:
        return False

    stale = probe.definition_path
    probe.definition_path = None
    probe.definition_built_at = None
    probe.definition_sha256 = None
    build = dict(probe.definition_build or {})
    build["invalidated"] = {"at": utc_now().isoformat(), "reason": reason, "was": stale}
    probe.definition_build = build
    db.commit()
    logger.info(
        "probe %s: cached definition invalidated (%s); it stated evidence that has changed",
        probe_id, reason,
    )
    return True
