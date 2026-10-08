"""
Probe monitor models (Feature 032, PADR IDL-51 storage / IDL-52 evidence ladder).

A probe monitor is `score_t = w·ẑ_t + b` over ONE decoder layer's `resid_post`
activations, followed by a combining rule. Five tables carry it:

  `probe_monitor_datasets`    a label-mapped VIEW over a downloaded dataset
  `probe_monitor_runs`        one GPU job: render → capture → select → train → evaluate
  `probe_monitors`            one trained probe (layer × rule × variant)
  `probe_monitor_evaluations` one probe scored on one dataset
  `probe_monitor_judge_runs`  the LLM-monitor baseline a probe is compared against

WHY A VIEW RATHER THAN A COPY (IDL-51). A probe dataset does not duplicate rows;
it records the columns, the label mapping and the filter that turn a downloaded
dataset into labelled examples, plus the COUNTS that mapping produced. So the
same download serves several probe datasets, and a mapping is auditable after the
fact — which matters because BR-003 is emphatic that a keyword filter NARROWS a
set and never labels it. The counts are stored rather than recomputed for the same
reason: `excluded` and `unparseable` rows are the ones a reader needs to see, and a
recount against a re-downloaded dataset could silently disagree.

WHY TENSORS ARE NOT HERE (FTDD T3). Weights live in safetensors under the run's
`artifact_dir` and the row carries the PATH. Evaluation scores are `.npy` beside
them. A probe's numbers are large, immutable once written, and read by 033 when it
builds a definition — none of which a JSONB column is for.

⚠ `probe_monitors.threshold` IS NOT A QUALITY NUMBER. It is the operating point
calibrated at `target_fpr`, and `realised_fpr` records what that threshold actually
spends, which is usually not the target. `threshold_source` says which data set it
came from, because a threshold from validation negatives and one from a held-out
calibration corpus are not interchangeable claims.
"""

import uuid

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID

from ..core.clock import utc_now
from ..core.database import Base

#: Every status column here uses these strings, following the
#: `circuit_capture_runs` precedent rather than a native PG enum. A native enum
#: needs a migration to gain a value, and this repo has already paid for three
#: tables whose enums lacked CANCELLED when cancellation arrived.
PROBE_MONITOR_STATUSES = ("pending", "running", "completed", "failed", "cancelled")

#: ⚠ EVERY TIMESTAMP HERE IS `DateTime(timezone=True)`, AND THAT IS NOT DECORATION.
#: The defaults are `utc_now()`, which returns an AWARE datetime. Against a bare
#: `DateTime` column Postgres is `TIMESTAMP WITHOUT TIME ZONE`, and asyncpg refuses an
#: aware value for it outright: `invalid input for query argument … can't subtract
#: offset-naive and offset-aware datetimes`. That is what happened in production on the
#: very first probe dataset — the mapping succeeded (4000 positive / 4000 negative, 0
#: unparseable) and the INSERT then failed, so the work was done and thrown away.
#:
#: The clock guard says "every DateTime column in this schema is `timezone=True`". I
#: changed the VALUE to aware to satisfy it and left the COLUMN naive, which is half a
#: fix and worse than neither: the old naive default at least matched its column.


def _pmd_id() -> str:
    return f"pmd_{uuid.uuid4().hex[:12]}"


def _pmr_id() -> str:
    return f"pmr_{uuid.uuid4().hex[:12]}"


def _pm_id() -> str:
    return f"pm_{uuid.uuid4().hex[:12]}"


def _pme_id() -> str:
    return f"pme_{uuid.uuid4().hex[:12]}"


def _pmj_id() -> str:
    return f"pmj_{uuid.uuid4().hex[:12]}"


class ProbeMonitorDataset(Base):
    """A label-mapped view over one (dataset, config, split).

    `role` separates the three jobs a set can do and they are NOT
    interchangeable: `train` fits the probe, `eval` measures it, and
    `calibration` supplies negatives for the FPR threshold. A calibration set is
    NOT labelled for the concept (FR-2 / BR-003) — ultrachat is a source of
    ordinary conversation, not of "low stakes" ground truth — so it may never be
    promoted to an evaluation set, and `distribution` is meaningless on it.
    """

    __tablename__ = "probe_monitor_datasets"

    id = Column(String(36), primary_key=True, default=_pmd_id)
    name = Column(String(255), nullable=False)
    #: ⚠ UUID, NOT String(36). `datasets.id` is a real PG `uuid` column while every
    #: other id on this estate is a prefixed varchar, so a `String(36)` FK here is
    #: refused outright by Postgres — which is how the migration round trip caught
    #: it. Mirror the target's type, never the local convention.
    dataset_id = Column(
        UUID(as_uuid=True), ForeignKey("datasets.id", ondelete="CASCADE"), nullable=False
    )

    #: The HF config and split this view reads. Both are part of the view's
    #: identity: one repo's `training` and `toolace_balanced` configs are
    #: different data, and FR-2 makes them different downloads.
    config = Column(String(255), nullable=True)
    split = Column(String(255), nullable=True)

    input_column = Column(String(255), nullable=False)
    label_column = Column(String(255), nullable=False)

    #: {raw_value: "positive" | "negative" | "excluded"}. `excluded` is explicit
    #: and counted — an unmapped value is a configuration error, not a silent
    #: drop, because a row that vanishes between the input and the counts is how
    #: an AUROC over 80 rows gets presented as one over 100.
    label_mapping = Column(JSONB, nullable=False, default=dict)

    #: {terms: [...], mode: "any"|"all", case_sensitive: bool} or NULL.
    #: ⚠ A FILTER NARROWS, IT NEVER LABELS (BR-003). Nothing in this column may
    #: reach the label mapping, and `test_probe_monitor_inputs` mutates the
    #: filter to assign a label as a negative control.
    keyword_filter = Column(JSONB, nullable=True)

    #: Rows sharing a value here must not straddle the train/validation split —
    #: a contrastive pair split across the boundary leaks.
    pair_column = Column(String(255), nullable=True)

    role = Column(String(16), nullable=False, default="eval")          # train|eval|calibration
    distribution = Column(String(24), nullable=True)                   # in_distribution|out_of_distribution

    #: {positive, negative, excluded, filtered_out, unparseable}. Stored, not
    #: derived: these are the numbers a refusal cites.
    counts = Column(JSONB, nullable=False, default=dict)

    created_at = Column(DateTime(timezone=True), nullable=False, default=utc_now)

    __table_args__ = (
        Index("ix_probe_monitor_datasets_dataset_id", "dataset_id"),
        Index("ix_probe_monitor_datasets_role", "role"),
    )


class ProbeMonitorRun(Base):
    """One GPU job that turns a probe dataset into trained probes."""

    __tablename__ = "probe_monitor_runs"

    id = Column(String(36), primary_key=True, default=_pmr_id)
    #: String(255): `models.id` is varchar(255) (`m_{uuid}`), not 36.
    model_id = Column(String(255), ForeignKey("models.id", ondelete="CASCADE"), nullable=False)
    train_dataset_id = Column(
        String(36), ForeignKey("probe_monitor_datasets.id", ondelete="CASCADE"), nullable=False
    )
    #: Evaluation sets are a LIST of probe-dataset ids rather than a join table:
    #: the set of sets is chosen once at submit and never edited, and the order
    #: is what the report renders.
    eval_dataset_ids = Column(JSONB, nullable=False, default=list)
    calibration_dataset_id = Column(
        String(36), ForeignKey("probe_monitor_datasets.id", ondelete="SET NULL"), nullable=True
    )

    #: {layers|stride, rules, scope, max_length, val_fraction, seed,
    #:  top_n_layers, sae_variant, sae_k, target_fpr, batch_size, dtype}
    config = Column(JSONB, nullable=False, default=dict)

    stage = Column(String(32), nullable=True)
    status = Column(String(16), nullable=False, default="pending")
    progress = Column(Float, nullable=True)
    error_message = Column(Text, nullable=True)
    celery_task_id = Column(String(155), nullable=True)

    gpu_request = Column(String(64), nullable=True)
    gpu_uuid = Column(String(64), nullable=True)
    gpu_uuids = Column(JSONB, nullable=True)

    artifact_dir = Column(String(1000), nullable=True)

    #: FR-15: everything needed to reproduce — model revision, dtype, template
    #: hash, dataset revisions, seeds, miStudio version, per-stage wall times.
    environment = Column(JSONB, nullable=False, default=dict)

    #: The FULL layer × pool grid with its validation AUROC, not only the winner.
    #: A sweep that reports only its argmax cannot be audited for a near-tie, and
    #: a near-tie is exactly when the chosen layer is arbitrary.
    layer_selection = Column(JSONB, nullable=True)

    created_at = Column(DateTime(timezone=True), nullable=False, default=utc_now)
    updated_at = Column(
        DateTime(timezone=True), nullable=False, default=utc_now, onupdate=utc_now
    )
    completed_at = Column(DateTime(timezone=True), nullable=True)

    __table_args__ = (
        Index("ix_probe_monitor_runs_model_status", "model_id", "status"),
    )


class ProbeMonitor(Base):
    """One trained probe: a layer, a rule, and a variant."""

    __tablename__ = "probe_monitors"

    id = Column(String(36), primary_key=True, default=_pm_id)
    run_id = Column(String(36), ForeignKey("probe_monitor_runs.id", ondelete="CASCADE"), nullable=False)

    layer = Column(Integer, nullable=False)
    rule = Column(String(32), nullable=False)
    rule_params = Column(JSONB, nullable=False, default=dict)

    variant = Column(String(8), nullable=False, default="dense")       # dense|sae
    #: SET NULL, not CASCADE: deleting an SAE must not delete the record that a
    #: probe was trained over it. The probe becomes unbuildable (033 refuses it)
    #: and that is a state to REPORT, not to erase.
    #: String(255): `external_saes.id` is varchar(255) (`sae_{uuid}`).
    sae_id = Column(String(255), ForeignKey("external_saes.id", ondelete="SET NULL"), nullable=True)
    sae_feature_indices = Column(JSONB, nullable=True)

    val_metrics = Column(JSONB, nullable=False, default=dict)
    selected = Column(Boolean, nullable=False, default=False)

    weights_path = Column(String(1000), nullable=True)
    norm_path = Column(String(1000), nullable=True)

    threshold = Column(Float, nullable=True)
    target_fpr = Column(Float, nullable=True)
    realised_fpr = Column(Float, nullable=True)
    threshold_source = Column(String(32), nullable=True)   # calibration_set|validation_negatives
    calibration_dataset_id = Column(
        String(36), ForeignKey("probe_monitor_datasets.id", ondelete="SET NULL"), nullable=True
    )
    #: The negative scores the threshold was cut from, as a .npy path.
    #:
    #: ⚠ KEPT SO THE OPERATING POINT CAN MOVE WITHOUT A GPU. A threshold is the
    #: (1 - target_fpr) quantile of exactly these numbers, so re-deriving it at another
    #: target is arithmetic over an array already computed. Without this, answering "what
    #: would 5% look like" meant a fresh ~70-minute run that re-captured activations to
    #: recompute a percentile. `ProbeMonitorEvaluation.scores_path` has always kept its
    #: arrays for the same reason; calibration did not, and that asymmetry is what made the
    #: operating point feel fixed at training time when it is the one part free to move.
    calibration_scores_path = Column(String(1000), nullable=True)

    #: The per-row SCORED TOKEN COUNTS those negatives came from, as a .npy path.
    #:
    #: ⚠ WITHOUT THIS, `length_bands` CANNOT BE RE-CUT AND `calibration_scores_path` IS ONLY
    #: HALF A DIAL. `length_band_decisions` refuses outright on a length/score mismatch, and its
    #: boundaries are quantiles of the lengths actually observed — so they cannot be guessed from
    #: the scores. The lengths were in memory beside the scores on every run since bands shipped
    #: and were discarded, which made a per-length re-cut cost a GPU pass to recover 8 KB.
    #:
    #: NULL means the probe predates this, not that its rows had no lengths. Aligned
    #: index-for-index with `calibration_scores_path`; a reader must check both lengths match
    #: rather than assuming it.
    calibration_lengths_path = Column(String(1000), nullable=True)

    #: EVERY OPERATING POINT THIS PROBE HAS EVER SERVED, append-only, or NULL for a probe whose
    #: bar is still the one its run cut.
    #:
    #: ⚠ NULL MEANS "NEVER RE-CUT", NOT "UNKNOWN". A probe recalibrated to a new `target_fpr`
    #: keeps serving the same weights under the same id, so a reader comparing two verdicts
    #: months apart has no other way to learn that the bar between them moved. Deliberately no
    #: server default and no backfill: a default would make every historical row claim a
    #: provenance it never had, which is the `chat_format NOT NULL DEFAULT 'auto'` mistake.
    #:
    #: Each entry: `{at, revision, from: {threshold, target_fpr, realised_fpr}, to: {...},
    #: n_negatives, threshold_source, windows_recut, reason}`. `revision` is the integer a
    #: consumer projects onto each verdict it judges, so an old verdict stays answerable after
    #: the bar has moved again.
    calibration_history = Column(JSONB, nullable=True)

    #: A THRESHOLD PER CONTRACT WINDOW — `{window: {threshold, target_fpr, realised_fpr,
    #: n_negatives}}`, or NULL for a probe calibrated before windows existed.
    #:
    #: ⚠ A THRESHOLD IS A QUANTILE OF ONE WINDOW'S NEGATIVES. `threshold` above was cut under
    #: the run's own scope; read over a narrower window the same number no longer names the same
    #: false-positive rate, because it is a quantile of a different distribution. A consumer
    #: reporting a `prompt` verdict against the whole-request bar is ranking, not measuring.
    #:
    #: NULL is meaningful and must stay distinguishable from an empty dict: "this probe predates
    #: per-window calibration" is a different claim from "its windows were calibrated and came
    #: back empty", and only the first should make a consumer fall back to `threshold`.
    window_decisions = Column(JSONB, nullable=True)

    #: A THRESHOLD PER ABSOLUTE TOKEN-LENGTH BAND — a list of
    #: `{min_tokens, max_tokens, threshold, threshold_source, target_fpr, realised_fpr,
    #: n_negatives}`, or NULL for a probe calibrated before this existed or without a
    #: calibration view.
    #:
    #: ⚠ ONE CONSTANT THRESHOLD IS MISCALIBRATED AT EVERY LENGTH BUT THE ONE IT WAS CUT AT.
    #: A probe's score drifts with input length in a direction that depends on the corpus.
    #: Measured on the shipped `mean` probe over five out-of-distribution sets: realised FPR
    #: reaching 5.4x its 1% budget on the longest quartile of `anthropic_hh_balanced`, and
    #: recall falling 0.500 -> 0.297 on `mental_health_balanced` — the direction that took a
    #: live monitor silent on turn four of a real conversation.
    #:
    #: ⚠ BOUNDARIES ARE ABSOLUTE TOKEN COUNTS, NOT QUANTILES. They are CHOSEN as quantiles of
    #: the calibration corpus, because that puts equal evidence in each band — but a consumer
    #: sees one request at a time and cannot compute a quantile of anything.
    #:
    #: NULL stays distinguishable from `[]` for the same reason as `window_decisions`: "never
    #: attempted" and "attempted and empty" are different claims about the same probe.
    length_bands = Column(JSONB, nullable=True)

    #: THE RENDER FORM THIS PROBE WAS TRAINED, CALIBRATED AND EVALUATED UNDER (2026-10-08), e.g.
    #: `{"generation_prompt": true, "add_special_tokens": false}` — what miLLM serves. NULL means
    #: NOT RECORDED, which for every row older than this column means rendered WITHOUT the
    #: generation prompt; it is never read as the served form (`probe_monitor_run.probe_render`).
    #: Copied from the run's `environment["render_form"]` when the probe is persisted.
    render_form = Column(JSONB, nullable=True)

    #: Whether the rule has an exact online form. 033 PUBLISHES this, so it must
    #: be true: `test_probe_monitor_model` checks every streaming form against its
    #: batch form to 1e-6.
    streamable = Column(Boolean, nullable=False, default=False)

    #: `ProbeRung` (IDL-52) as an int, with the reasons that earned it. Never a
    #: causal claim — a probe is a detector.
    rung = Column(Integer, nullable=False, default=0)
    rung_reasons = Column(JSONB, nullable=False, default=list)

    # ── the exported definition (033 FR-7) ────────────────────────────────────
    #: The cached `mistudio.probe-definition/v1` document on disk, or NULL when none has been
    #: built. A PATH rather than the document itself: it is up to 2 MB and the row is read by
    #: every listing.
    definition_path = Column(String(1000), nullable=True)
    #: Indexed because "which probes have been exported" is the one filter a listing applies.
    #: DECLARED HERE as well as in the migration: `test_alembic_drift_matches_the_ratchet`
    #: compares the migrated schema against the models, and an index only one side knows about
    #: is drift — which is how a migration and an ORM start describing different databases.
    definition_built_at = Column(DateTime(timezone=True), nullable=True, index=True)
    #: Of the serialised bytes, so a consumer can verify the download and 033's publisher can
    #: tell "already uploaded" from "changed since".
    definition_sha256 = Column(String(64), nullable=True)
    #: What the build was ASKED for and what it resolved — the acknowledgement, the sampling
    #: seed, the model revision it pinned, the vector count. A definition nobody can reproduce
    #: from the row is not reproducible (FR-15's rule, applied to the export).
    definition_build = Column(JSONB, nullable=True)
    #: Every HuggingFace publication of this probe, appended never replaced:
    #: `[{repo_id, revision, path, at}]`. A list because a probe can be published to a private
    #: repo and later to a public one, and the earlier upload does not stop existing.
    #: `server_default` as well as `default`, because the migration needs one to add a NOT NULL
    #: column to a table that may have rows, and the drift guard compares the two sides.
    published = Column(JSONB, nullable=False, default=list, server_default=text("'[]'::jsonb"))

    created_at = Column(DateTime(timezone=True), nullable=False, default=utc_now)

    __table_args__ = (
        Index("ix_probe_monitors_run_id", "run_id"),
        Index("ix_probe_monitors_run_selected", "run_id", "selected"),
    )


class ProbeMonitorEvaluation(Base):
    """One probe scored on one probe dataset.

    ⚠ A REFUSAL IS A ROW. Below 20 examples of either class the row is written
    with `status="refused"` and `metrics.reason`, never omitted and never given a
    0.5. A missing row reads as "not run yet"; a 0.5 reads as "measured and
    chance". Neither is what happened.
    """

    __tablename__ = "probe_monitor_evaluations"

    id = Column(String(36), primary_key=True, default=_pme_id)
    probe_id = Column(String(36), ForeignKey("probe_monitors.id", ondelete="CASCADE"), nullable=False)
    dataset_id = Column(
        String(36), ForeignKey("probe_monitor_datasets.id", ondelete="CASCADE"), nullable=False
    )

    status = Column(String(16), nullable=False, default="pending")     # + "refused"
    n_positive = Column(Integer, nullable=True)
    n_negative = Column(Integer, nullable=True)

    #: AUROC, its CI, operating points, ROC points (capped at 200), length
    #: bands, and on a refusal the reason and both counts.
    metrics = Column(JSONB, nullable=False, default=dict)
    scores_path = Column(String(1000), nullable=True)

    created_at = Column(DateTime(timezone=True), nullable=False, default=utc_now)

    __table_args__ = (
        Index("ix_probe_monitor_evaluations_probe_id", "probe_id"),
        Index("ix_probe_monitor_evaluations_dataset_id", "dataset_id"),
    )


class ProbeMonitorJudgeRun(Base):
    """An LLM monitor over the same sets, as the BASELINE a probe is compared with.

    Not a grader of the probe. FR-13 rung 3 is "judge-compared", which means the
    two were measured on the same data — it says nothing about the judge being
    right. `parse_failures` is a first-class column because a judge whose output
    stopped parsing produces a silently smaller sample, and the run must refuse
    over the FR-11 limit rather than report the remainder.
    """

    __tablename__ = "probe_monitor_judge_runs"

    id = Column(String(36), primary_key=True, default=_pmj_id)
    endpoint = Column(String(500), nullable=False)
    model = Column(String(255), nullable=False)
    prompt_version = Column(String(64), nullable=False)
    dataset_ids = Column(JSONB, nullable=False, default=list)
    probe_id = Column(String(36), ForeignKey("probe_monitors.id", ondelete="SET NULL"), nullable=True)

    status = Column(String(16), nullable=False, default="pending")
    progress = Column(Float, nullable=True)
    celery_task_id = Column(String(155), nullable=True)
    metrics = Column(JSONB, nullable=False, default=dict)
    parse_failures = Column(Integer, nullable=False, default=0)
    # ⚠ THESE TWO WERE ACCEPTED BY THE REQUEST AND DROPPED ON THE FLOOR, and Stage 2 acceptance
    # is how it surfaced. `JudgeRunCreate` declares `max_rows_per_set` (ge=20, le=50000) and
    # `parse_failure_limit`, validates both, and nothing carried them any further: there was no
    # column, the endpoint did not pass them, and `execute_judge_run` read every row and used
    # `judge_rows`' own default limit.
    #
    # MEASURED: a run submitted with `max_rows_per_set: 200` over five sets judged **5,737 rows**
    # instead of 1,000 — 5.7x the work, 77 minutes instead of about 13, against a served 7B model.
    # An operator tightening the parse-failure limit got the module default and no indication.
    #
    # Stored rather than only passed, because a run has to say what it measured: "the judge agreed
    # with the probe on 200 rows per set" and "…on every row" are different claims.
    max_rows_per_set = Column(Integer, nullable=True)
    parse_failure_limit = Column(Float, nullable=True)
    error_message = Column(Text, nullable=True)

    created_at = Column(DateTime(timezone=True), nullable=False, default=utc_now)
    completed_at = Column(DateTime(timezone=True), nullable=True)

    __table_args__ = (
        Index("ix_probe_monitor_judge_runs_probe_id", "probe_id"),
    )
