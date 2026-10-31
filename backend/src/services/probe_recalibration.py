"""Move a probe's operating point without a GPU, from the negatives already on disk.

⚠ **THE DIAL WAS BUILT IN SEPTEMBER AND ATTACHED TO NOTHING.**
`persist_calibration_negatives` has kept every probe's negative scores since then, and its own
docstring says why: *"a threshold is the (1 - target_fpr) quantile of exactly these numbers, so
re-deriving it at another target is arithmetic over an array already computed"*. Nothing ever
read the file back. The only line in this estate that wrote `probe.threshold` was inside the
`calibrating` stage of a full training run, so changing an operating point cost ~2.6 hours of
GPU — which is why every calibration question so far has been settled by argument instead of by
measurement.

⚠ **WHAT THIS DOES NOT DO, STATED HERE BECAUSE A FAST DIAL INVITES THE WRONG USE.** A re-cut bar
is still a quantile of the SAME calibration corpus. It does not make a threshold transfer between
distributions, and `threshold_transfer` — which this module returns on every proposal — exists to
say so: on the first shipped probe, five evaluation sets' own 1% thresholds spanned 24 points.
Lowering a bar until a probe fires on the traffic in front of you is not calibration. The
proposal therefore always carries the per-set consequences and the `caution` string, and the
caller is expected to show them.

Three refusals, each naming the number that caused it:

* **The array is missing.** A probe trained before persistence cannot be re-cut at all, and that
  is a different statement from "its bar cannot move".
* **The target is finer than the sample can afford.** `calibrate` answers an unaffordable budget
  with `threshold=None`, which means FIRE ON NOTHING — a real operating point, and the single
  worst thing to do silently. 2,000 negatives cannot express a rate finer than 1/2000.
* **A derived bar would be left describing a target the probe no longer has.** This is the
  load-bearing one. `window_for` and `threshold_for_length` both PREFER a per-window or
  per-length bar over the global one, so moving only the global would leave an armed probe's
  served behaviour unchanged — a recalibration that reports success and does nothing. Each stale
  entry also carries its own `target_fpr` and would claim a target that is no longer true.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

#: The windows and bands whose arrays this module needs in order to re-cut them. A probe
#: carrying a derived bar whose array is absent is refused rather than partially moved.
_LENGTH_BANDS = "length_bands"


class RecalibrationRefused(Exception):
    """A refusal with the code and HTTP status every caller should use.

    ⚠ RAISED FROM THE SERVICE, NOT THE ENDPOINT, for the same reason the evidence gate lives in
    the definition builder: a re-cut dispatched from the MCP tool must meet the same refusals,
    and a gate in the endpoint is one the other callers bypass.
    """

    def __init__(self, code: str, detail: str, *, status: int = 409,
                 extra: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(detail)
        self.code = code
        self.detail = detail
        self.status = status
        #: Structured fields merged into the HTTP `detail` beside `code` and `message` (039 TI4):
        #: `decision_changed` carries `expected`, `current` and `next`.
        self.extra = dict(extra or {})


def load_calibration_array(path: Optional[str], *, what: str) -> Optional[List[float]]:
    """A persisted `.npy` as a plain list, or `None` when it is not there.

    Returns `None` rather than raising for a missing file: "this probe predates persistence" is
    the caller's decision to make, and three callers make it differently.
    """
    if not path:
        return None
    import numpy as np

    file = Path(path)
    if not file.exists():
        logger.warning("calibration %s recorded at %s is not on disk", what, path)
        return None
    try:
        return [float(v) for v in np.load(file).reshape(-1)]
    except Exception as exc:  # noqa: BLE001 - a corrupt array must not read as an absent one
        raise RecalibrationRefused(
            "calibration_array_unreadable",
            f"the stored calibration {what} at {path} could not be read ({exc})",
        ) from exc


def finest_affordable_fpr(n_negatives: int) -> float:
    """The smallest false-positive rate `n` negatives can express.

    `calibrate` spends `int(target_fpr * n)` negatives, rounded DOWN, so anything below `1/n`
    buys zero of them and the threshold lands above every negative — fire on nothing.
    """
    return 1.0 / float(n_negatives) if n_negatives > 0 else 1.0


def refuse_unaffordable_target(target_fpr: float, n_negatives: int) -> None:
    """Refuse a target the sample cannot express, naming the rate that it can."""
    if int(target_fpr * n_negatives) > 0:
        return
    raise RecalibrationRefused(
        "target_fpr_unaffordable",
        f"{n_negatives} negatives cannot express a false-positive rate of {target_fpr:g}; the "
        f"finest they can express is {finest_affordable_fpr(n_negatives):g}. At {target_fpr:g} "
        f"the threshold would land above every negative, which means the probe fires on "
        f"NOTHING — pass allow_fire_on_nothing=true if that is genuinely what you want",
        status=422,
    )


def recut_windows(probe: Any, *, target_fpr: float) -> Tuple[Optional[Dict[str, Any]], List[str]]:
    """Re-cut each per-window bar from its own stored negatives.

    Returns `(decisions, unrecuttable)`. `decisions` is `None` when the probe has none, which is
    the probe's existing "never attempted" state and must stay distinguishable from `{}`.
    `unrecuttable` names the windows whose negatives were never persisted — the caller refuses on
    a non-empty list rather than shipping a table where some entries moved and some did not.
    """
    from .probe_monitor_metrics import length_band_decisions
    from .probe_monitor_trainer import calibrate

    existing = probe.window_decisions
    if not existing:
        return (existing if existing is None else {}), []

    decisions: Dict[str, Any] = {}
    unrecuttable: List[str] = []
    for window, entry in existing.items():
        scores = load_calibration_array(
            (entry or {}).get("scores_path"), what=f"window {window!r} negatives"
        )
        if scores is None:
            unrecuttable.append(window)
            continue
        refuse_unaffordable_target(target_fpr, len(scores))
        calibration = calibrate(
            scores, target_fpr=target_fpr, source=probe.threshold_source or "calibration_set"
        )
        # ⚠ THE WINDOW'S OWN LENGTH BANDS MOVE WITH ITS BAR (2026-10-04), or a re-cut would copy
        # them through at the OLD target — and the runtime prefers a band to the window's bar, so
        # the window would serve exactly as before while the table showed a new number. A table
        # whose lengths were never kept cannot move, and makes the window unrecuttable.
        bands = (entry or {}).get("length_bands")
        if bands:
            lengths = load_calibration_array(
                (entry or {}).get("lengths_path"), what=f"window {window!r} lengths"
            )
            if lengths is None:
                unrecuttable.append(window)
                continue
            if len(lengths) != len(scores):
                raise RecalibrationRefused(
                    "calibration_arrays_disagree",
                    f"window {window!r}: the stored negatives ({len(scores)}) and lengths "
                    f"({len(lengths)}) are different sizes, so they do not describe the same rows",
                )
            bands = length_band_decisions(
                [float(v) for v in scores],
                [int(round(v)) for v in lengths],
                target_fpr=target_fpr,
                global_threshold=calibration.threshold,
            )
        decisions[window] = {
            **(entry or {}),
            "threshold": calibration.threshold,
            "target_fpr": calibration.target_fpr,
            "realised_fpr": calibration.realised_fpr,
            "n_negatives": calibration.n_negatives,
            "length_bands": bands or None,
        }
    return decisions, unrecuttable


def recut_length_bands(
    probe: Any,
    negatives: Sequence[float],
    *,
    target_fpr: float,
    global_threshold: Optional[float],
) -> Tuple[Optional[List[Dict[str, Any]]], bool]:
    """Re-cut the per-length table from the stored scores and lengths.

    Returns `(bands, unrecuttable)`. `unrecuttable` is True when the probe HAS a table and the
    lengths it was built from were never persisted — the boundaries are quantiles of the lengths
    actually observed, so they cannot be reconstructed from the scores.
    """
    from .probe_monitor_metrics import length_band_decisions

    if not probe.length_bands:
        return (probe.length_bands if probe.length_bands is None else []), False

    lengths = load_calibration_array(probe.calibration_lengths_path, what="lengths")
    if lengths is None:
        return None, True
    if len(lengths) != len(negatives):
        raise RecalibrationRefused(
            "calibration_arrays_disagree",
            f"the stored negatives ({len(negatives)}) and lengths ({len(lengths)}) are different "
            f"sizes, so they do not describe the same rows; re-cutting per length needs a new run",
        )
    bands = length_band_decisions(
        list(negatives),
        [int(round(v)) for v in lengths],
        target_fpr=target_fpr,
        global_threshold=global_threshold,
    )
    return bands, False


def refuse_stale_derived_bars(unrecuttable_windows: Sequence[str], bands_unrecuttable: bool) -> None:
    """Refuse rather than move the global bar and leave the derived ones behind.

    ⚠ THIS IS THE REFUSAL THAT STOPS A SILENT NO-OP. Both lookups prefer a derived bar over the
    global one, so a probe whose windows stayed put would serve exactly as before while the row,
    the report and the UI all showed the new number.
    """
    if not unrecuttable_windows and not bands_unrecuttable:
        return
    missing: List[str] = []
    if unrecuttable_windows:
        missing.append(
            "per-window bars or their own length tables for " + ", ".join(sorted(unrecuttable_windows))
        )
    if bands_unrecuttable:
        missing.append("the per-length table")
    raise RecalibrationRefused(
        "derived_bars_not_recuttable",
        f"this probe carries {' and '.join(missing)}, and the negatives they were cut from were "
        f"never persisted, so they cannot move without the model. Moving only the global "
        f"threshold would change nothing a consumer serves, because a per-window and a "
        f"per-length bar both take precedence over it. Re-cut them with recut_windows=true "
        f"(a GPU job), or recalibrate a probe trained after calibration arrays were kept",
    )


def next_revision(probe: Any) -> int:
    """The revision this re-cut will be. 1 is the bar the probe's own run cut.

    ⚠ DERIVED FROM THE RECORDED REVISIONS, NOT FROM THE LIST'S LENGTH AND NOT FROM A COUNTER.
    A counter can disagree with the entries a reader uses to answer "what was revision 3"; a
    length is wrong the moment an entry is ever dropped or back-filled. The first version of
    this function used the length and returned 4 for a two-entry history, which its own test
    caught — the revision is the one number a consumer joins on, so it has to come from the
    thing being joined to.

    An empty history yields 2, because the first re-cut seeds revision 1 for the bar the run
    cut and then appends its own.
    """
    history = probe.calibration_history or []
    if not history:
        return 2
    return max(int(entry.get("revision") or 0) for entry in history) + 1


def _calibration_for(db: Any, probe: Any, threshold_source: Optional[str]) -> Any:
    """The calibration reference a bar cut under `threshold_source` will carry (TI5), through the
    ONE rule `probe_decision.calibration_for_bar` implements.

    `db=None` (a sessionless preview over a stub) can resolve every case except a source that
    changed to `calibration_set` with no snapshot to copy: that needs the run row. It is then
    reported NOT FORMABLE with that reason — a digest computed without the reference would be one
    the commit never stores.
    """
    from .probe_decision import NotFormable, calibration_for_bar, snapshot_threshold_source

    snapshot = getattr(probe, "decision_snapshot", None)
    if db is None and threshold_source == "calibration_set" and not (
        snapshot and snapshot_threshold_source(snapshot) == threshold_source
    ):
        return NotFormable("no database session was given to resolve the calibration reference")
    return calibration_for_bar(db, probe, threshold_source)


def _bar_row(*, threshold: Any, target_fpr: Any, realised_fpr: Any, threshold_source: Any,
             window_decisions: Any, length_bands: Any) -> Any:
    """A row-shaped view of bar VALUES, so a candidate bar is formed by the same assembly as a
    stored one (`probe_decision.form_snapshot`) — never by a second one (FR-8)."""
    from types import SimpleNamespace

    return SimpleNamespace(threshold=threshold, target_fpr=target_fpr, realised_fpr=realised_fpr,
                           threshold_source=threshold_source, window_decisions=window_decisions,
                           length_bands=length_bands)


def _blocks_equal_without_calibration(current_row: Any, proposed_row: Any) -> bool:
    """D-T5: the internal fallback for `unchanged` when a digest is null (a bar that is not
    formable). The two blocks are formed WITHOUT `calibration` and compared; never published."""
    from .probe_decision import form_snapshot

    current = form_snapshot(current_row, None)
    proposed = form_snapshot(proposed_row, None)
    if current["decision"] is None or proposed["decision"] is None:
        return False
    return current["decision"] == proposed["decision"]


def decision_unchanged(current_digest: Optional[str], proposed_digest: Optional[str],
                       current_row: Any, proposed_row: Any) -> bool:
    """FR-12: is the candidate decision the one already in force?

    By digest when both exist; by the blocks without `calibration` when either is null (D-T5) —
    so a bar that is not formable can still be recognised as re-cut to itself.
    """
    if current_digest is not None and proposed_digest is not None:
        return current_digest == proposed_digest
    if (current_digest is None) != (proposed_digest is None):
        # One side forms and the other does not: the decision a consumer could hold changed.
        return False
    return _blocks_equal_without_calibration(current_row, proposed_row)


def propose(probe: Any, evaluations: Sequence[Any], *, target_fpr: float,
            allow_fire_on_nothing: bool = False, db: Any = None) -> Dict[str, Any]:
    """What this probe's operating point would become, writing nothing.

    ⚠ THE PROPOSAL CARRIES `threshold_transfer` FOR THE CANDIDATE BAR, NOT THE SHIPPED ONE, and
    that is the whole reason a preview is worth having. `threshold_transfer` already looks any
    threshold up against each evaluation set's stored ROC and returns the recall, the realised
    FPR, whether the set is unreachable, and a `caution` — so the per-set consequence of a move
    is free, with no GPU and no new metric code. Without it a dial is a number with no feedback.
    """
    from .probe_monitor_metrics import threshold_transfer
    from .probe_monitor_trainer import calibrate

    negatives = load_calibration_array(probe.calibration_scores_path, what="negatives")
    if negatives is None:
        raise RecalibrationRefused(
            "no_calibration_array",
            f"probe {probe.id} has no persisted calibration negatives, so its bar cannot be "
            f"re-cut; it was trained before the arrays were kept, and moving its operating "
            f"point needs a new run",
        )
    if not allow_fire_on_nothing:
        refuse_unaffordable_target(target_fpr, len(negatives))

    try:
        calibration = calibrate(
            negatives, target_fpr=target_fpr, source=probe.threshold_source or "calibration_set"
        )
    except ValueError as exc:
        raise RecalibrationRefused("target_fpr_invalid", str(exc), status=422) from exc

    windows, unrecuttable_windows = recut_windows(probe, target_fpr=target_fpr)
    bands, bands_unrecuttable = recut_length_bands(
        probe, negatives, target_fpr=target_fpr, global_threshold=calibration.threshold
    )
    refuse_stale_derived_bars(unrecuttable_windows, bands_unrecuttable)

    rows = [{"metrics": e.metrics} for e in evaluations]
    # ── Feature 039 FR-10: the identity of both ends ──────────────────────────────────────────
    # ⚠ THE CANDIDATE IS FORMED BY THE SAME ASSEMBLY AS EVERY STORED BAR (`form_snapshot`), with the
    # calibration reference the commit will store (TI5). A preview that formed its own block — say,
    # without `calibration` — would promise a `decision_digest` the build never carries, and an agent
    # holding it would join on nothing (FR-8).
    from .probe_decision import form_snapshot

    proposed_row = _bar_row(
        threshold=calibration.threshold, target_fpr=calibration.target_fpr,
        realised_fpr=calibration.realised_fpr, threshold_source=calibration.source,
        window_decisions=windows, length_bands=bands,
    )
    proposed_snapshot = form_snapshot(
        proposed_row, _calibration_for(db, probe, calibration.source)
    )
    current_digest = getattr(probe, "decision_digest", None)
    # The preview's `unchanged`. (The GPU arm runs this over its swapped table; its result's
    # `unchanged` is `apply`'s, decided on the LOCKED row — never this one.)
    current_row = _bar_row(
        threshold=probe.threshold, target_fpr=probe.target_fpr, realised_fpr=probe.realised_fpr,
        threshold_source=probe.threshold_source, window_decisions=probe.window_decisions,
        length_bands=probe.length_bands,
    )
    current_revision = next_revision(probe) - 1
    return {
        "probe_id": probe.id,
        "current": {
            "threshold": probe.threshold,
            "target_fpr": probe.target_fpr,
            "realised_fpr": probe.realised_fpr,
            "threshold_source": probe.threshold_source,
            "revision": current_revision,
            "decision_digest": current_digest,
            "decision_revision": current_revision,
        },
        "proposed": {
            "threshold": calibration.threshold,
            "target_fpr": calibration.target_fpr,
            "realised_fpr": calibration.realised_fpr,
            "threshold_source": calibration.source,
            "revision": next_revision(probe),
            "n_negatives": calibration.n_negatives,
            "fires_on_nothing": calibration.threshold is None,
            "decision_digest": proposed_snapshot["digest"],
            "decision_revision": next_revision(probe),
        },
        # A re-cut moves the bar, never the detector: the last built detector digest, unchanged.
        "detector_digest": getattr(probe, "built_detector_digest", None),
        "detector_unchanged": True,
        # FR-12 on the preview: committing this would append nothing.
        "unchanged": decision_unchanged(
            current_digest, proposed_snapshot["digest"], current_row, proposed_row
        ),
        "window_decisions": windows,
        "length_bands": bands,
        # The consequence of the CANDIDATE bar on every set that was evaluated, beside the
        # consequence of the one being served, so a move can be read as a change.
        "transfer_current": threshold_transfer(probe.threshold, probe.threshold_source, rows),
        "transfer_proposed": threshold_transfer(calibration.threshold, calibration.source, rows),
        # A published copy is append-only and cannot be reached. Stated on every proposal
        # rather than only on the commit, because it belongs in the decision.
        "published_copies_go_stale": bool(probe.published),
    }


def _lock_row(db: Any, probe: Any) -> Any:
    """The probe row, re-read under a row lock (Feature 039 C2).

    `populate_existing` because the probe is already in the session's identity map: without it the
    query would hand back the in-memory object — values read before another session's commit — and
    the comparison below would compare against nothing new. A seam so stub-session tests can supply
    the row; the real lock is proven only by the two-session database test
    (`test_probe_recalibration_locks.py`).
    """
    from ..models.probe_monitor import ProbeMonitor

    return (
        db.query(ProbeMonitor)
        .filter(ProbeMonitor.id == probe.id)
        .with_for_update()
        .populate_existing()
        .one()
    )


def _millm_copies_field(copies: Any) -> Any:
    """`imported_copies`' answer as the result field: ids, or why there are none to name."""
    from .millm_read import NotConfigured, Unknown

    if isinstance(copies, NotConfigured):
        return {"not_configured": copies.reason}
    if isinstance(copies, Unknown):
        return {"unknown": copies.reason}
    return list(copies)


def apply(
    db: Any, probe: Any, proposal: Dict[str, Any], *, reason: str = "",
    seed_windows: Optional[Dict[str, Any]] = None,
    expect_decision_digest: Optional[str] = None,
    bookkeeping: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Write a proposal onto the probe, and stale anything that stated the old bar.

    `bookkeeping` (the GPU arm's array paths and counts — `calibration_lengths_path`,
    `calibration_positions_path`, `calibration_n_negatives`) is written onto the LOCKED row inside the
    SAME commit as the bar (Feature 040 S3, review R1-6): it used to be re-applied in a second
    transaction after this one, so a worker death between the two left the row naming the pre-re-cut
    files beside a bar cut from the new ones.

    In this order (FTDD §6.4), all under ONE row lock and ONE commit:

    1. lock the row (`_lock_row`);
    2. refuse `decision_changed` when the locked decision is not the one the proposal started from,
       or not the one the caller said it read (`expect_decision_digest`, C3);
    3. an `unchanged` decision appends nothing (FR-12) — on the GPU arm only the re-scored arrays'
       bookkeeping is kept (D-T6);
    4. the revision is computed UNDER the lock; the history entry carries the new decision digest,
       and the seeded revision 1 the digest of the bar it replaces (FR-13);
    5. the row, its snapshot, and the invalidation of any cached definition (FR-11) are written
       together and committed once;
    6. AFTER the commit, outside the lock: miLLM's imported copies and the `next` list (FR-14).
    """
    from ..schemas.probe_identity import current_revision
    from . import millm_read
    from .next_calls import after_recut, recalibrate_preview
    from .probe_decision import calibration_for_bar, form_snapshot, write_decision_snapshot
    from .probe_definition_builder import mark_invalidated

    # ⚠ 1. THE LOCK (C2, a pre-existing race). Two re-cuts committed at once both appended "the
    # next revision" from the history each had read, so one revision number named two bars.
    locked = _lock_row(db, probe)
    started_from = (proposal.get("current") or {}).get("decision_digest")
    current_digest = locked.decision_digest
    changed = current_digest != started_from
    expected = started_from
    if not changed and expect_decision_digest is not None and current_digest != expect_decision_digest:
        # C3: an agent committing over a preview it did not read states the decision it read.
        changed, expected = True, expect_decision_digest
    started_revision = (proposal.get("current") or {}).get("decision_revision")
    locked_revision = current_revision(locked.calibration_history)
    if (not changed and current_digest is None and started_revision is not None
            and started_revision != locked_revision):
        # ⚠ A NOT-FORMABLE DECISION HAS NO DIGEST, SO `None == None` CANNOT SEE A MOVE (review R1).
        # Two re-cuts of such a probe both "started from None": the second overwrote the first's
        # bar with no refusal. Only a commit appends a revision (an identical re-cut appends
        # nothing), so a revision that moved under the lock is a decision that moved.
        changed = True
    if changed:
        db.rollback()
        logger.info(
            "decision_changed probe=%s expected=%s current=%s", probe.id, expected, current_digest
        )
        target = (proposal.get("proposed") or {}).get("target_fpr")
        raise RecalibrationRefused(
            "decision_changed",
            f"probe {probe.id}'s decision is now {current_digest} at revision {locked_revision}, "
            f"not {expected} at revision {started_revision} — it was re-cut "
            f"after this proposal was formed (or the decision you read is not the current one). "
            f"Nothing was written; preview again and commit against the decision in force",
            status=409,
            extra={
                "expected": expected,
                "current": current_digest,
                "next": [recalibrate_preview(
                    probe.id, float(target) if target is not None else 0.01,
                    "The decision moved since this proposal; a fresh preview shows the move from "
                    "the decision now in force.",
                )],
            },
        )

    for name, value in (bookkeeping or {}).items():
        if value is not None:
            setattr(locked, name, value)

    proposed = proposal["proposed"]
    windows = proposal.get("window_decisions")
    bands = proposal.get("length_bands")
    calibration = calibration_for_bar(db, locked, proposed["threshold_source"])
    proposed_row = _bar_row(
        threshold=proposed["threshold"], target_fpr=proposed["target_fpr"],
        realised_fpr=proposed["realised_fpr"], threshold_source=proposed["threshold_source"],
        window_decisions=windows, length_bands=bands,
    )
    candidate = form_snapshot(proposed_row, calibration)
    old_revision = current_revision(locked.calibration_history)
    identity_now = {"decision_digest": current_digest, "decision_revision": old_revision}

    # ⚠ 3. AN IDENTICAL RE-CUT APPENDS NOTHING (FR-12). It used to add a revision naming the same
    # bar as the one before, so "revision 3" and "revision 4" were one decision under two numbers —
    # and every consumer holding revision 3 was told its bar had moved when it had not.
    if decision_unchanged(current_digest, candidate["digest"], locked, proposed_row):
        unchanged: Dict[str, Any] = {
            **proposal, "committed": False, "reason": "unchanged", "unchanged": True,
            "definition_invalidated": False, "current": {**proposal["current"], **identity_now},
            "next": [],
        }
        if seed_windows is not None and windows is not None:
            # D-T6: the GPU arm re-scored the windows' arrays; their bookkeeping (`scores_path`,
            # `lengths_path`) is kept so the next re-cut needs no GPU. The contract-form block is by
            # definition unchanged, so the snapshot, the digest and the history stay as they are.
            locked.window_decisions = windows
            db.commit()
            unchanged["bookkeeping_updated"] = True
        elif any(v is not None for v in (bookkeeping or {}).values()):
            db.commit()  # the GPU arm's arrays were rewritten: their bookkeeping is kept (R1-6)
            unchanged["bookkeeping_updated"] = True
        else:
            db.rollback()
        logger.info("probe %s re-cut to the decision already in force; nothing committed", probe.id)
        return unchanged

    # 4. The revision, UNDER the lock.
    revision = next_revision(locked)
    previous = {
        "threshold": locked.threshold,
        "target_fpr": locked.target_fpr,
        "realised_fpr": locked.realised_fpr,
        "threshold_source": locked.threshold_source,
        "revision": old_revision,
        **identity_now,
    }
    entry = {
        "at": datetime.now(timezone.utc).isoformat(),
        "revision": revision,
        "from": {
            "threshold": locked.threshold,
            "target_fpr": locked.target_fpr,
            "realised_fpr": locked.realised_fpr,
        },
        "to": {
            "threshold": proposed["threshold"],
            "target_fpr": proposed["target_fpr"],
            "realised_fpr": proposed["realised_fpr"],
        },
        "n_negatives": proposed["n_negatives"],
        "threshold_source": proposed["threshold_source"],
        "windows_recut": sorted((windows or {}).keys()),
        "length_bands_recut": len(bands or []),
        "reason": reason,
    }
    # ⚠ SEEDED WITH REVISION 1 ON THE FIRST RE-CUT. Without it an event stamped `revision 1`
    # would be unanswerable: the row carries only the current bar, so the history has to record
    # the one the run cut before it is replaced.
    history = list(locked.calibration_history or [])
    if not history:
        history.append({
            # ⚠ `created_at`, AND ONLY `created_at`. `ProbeMonitor` HAS NO `updated_at`.
            # The first version read `(probe.updated_at or probe.created_at)` behind a
            # `getattr(..., None) or getattr(..., None)` condition — which the EXISTING
            # `created_at` satisfied, so the value expression then touched the attribute that
            # does not exist and the first live commit returned a 500. A defensive condition in
            # front of a direct access is not a fallback; it is a guard that cannot fire.
            "at": locked.created_at.isoformat() if locked.created_at else None,
            "revision": 1,
            "from": None,
            "to": {
                "threshold": locked.threshold,
                "target_fpr": locked.target_fpr,
                "realised_fpr": locked.realised_fpr,
            },
            "n_negatives": None,
            "threshold_source": locked.threshold_source,
            # The table revision 1 actually served — the GPU arm passes the one from before its
            # re-score, so a window it drops is still recorded as judged at revision 1
            # (review round 3, L1).
            "windows_recut": sorted(
                (seed_windows if seed_windows is not None else locked.window_decisions or {}).keys()
            ),
            "length_bands_recut": len(locked.length_bands or []),
            "reason": "cut by the training run",
            # FR-13: the digest of the bar revision 1 IS — the one being replaced (null when it is
            # not formable). Entries written before 039 carry no key and read null; they are never
            # back-filled, because a digest re-derived from an old entry's thresholds would name a
            # block nobody ever stored.
            "decision_digest": current_digest,
        })

    # 5. The row and its snapshot, in the same transaction as the bar.
    locked.threshold = proposed["threshold"]
    locked.target_fpr = proposed["target_fpr"]
    locked.realised_fpr = proposed["realised_fpr"]
    locked.threshold_source = proposed["threshold_source"]
    locked.window_decisions = windows
    locked.length_bands = bands
    # ⚠ THE BAR'S CONTRACT FORM IS STORED IN THE SAME COMMIT AS THE BAR (Feature 039 FR-8). The next
    # build copies this block, and refuses `decision_snapshot_stale` if it no longer matches — so a
    # re-cut that skipped this line could never be built, rather than silently published.
    snapshot = write_decision_snapshot(locked, calibration)
    entry["decision_digest"] = snapshot["digest"]
    history.append(entry)
    locked.calibration_history = history

    # ⚠ FR-11: ANY APPENDED REVISION STALES A CACHED DEFINITION, INSIDE THIS COMMIT. It used to be
    # invalidated only when the GLOBAL threshold moved, after the commit — so a window-only re-cut
    # left a "current" definition whose `identity.decision_digest` no longer named the row's
    # decision, and a crash between the two commits left a moved bar beside a definition claiming
    # the old one (review R1 debt).
    had_definition = bool(getattr(locked, "definition_path", None))
    invalidated = False
    if had_definition:
        invalidated = mark_invalidated(locked, reason="threshold recalibrated", cause="decision")
    db.commit()

    logger.info(
        "probe %s recalibrated: threshold %s -> %s at target_fpr %s (revision %s, decision %s -> "
        "%s, %s)",
        probe.id, entry["from"]["threshold"], entry["to"]["threshold"],
        proposed["target_fpr"], revision, current_digest, snapshot["digest"],
        reason or "no reason given",
    )

    # ⚠ 6. miLLM IS ASKED AFTER THE COMMIT, OUTSIDE THE LOCK. A read of another service inside the
    # lock would hold this row for up to its timeout, stalling every build commit and re-cut of it.
    #
    # ⚠ NOTHING AFTER THE COMMIT MAY RAISE (review R1). The GPU arm wraps this call in a rollback
    # that RESTORES the pre-re-score arrays on any exception; one escaping from here would put the
    # old arrays back on disk under a committed row whose bars were cut from the new ones. The bar
    # is committed: a failure to work out the follow-up calls is reported, never raised.
    # ⚠ ENH-006 D82/D84: THE RUNG IS RE-TAKEN AT THE NEW BAR. Rung 2 needs few false alarms on everyday and
    # nonsense text, and a catch rate on unseen tasks, all AT THE SHIPPED BAR, so a bar that moved can move
    # the rung either way; left alone, `probe.rung`
    # would state a check measured at a bar no longer in force. After the commit, and never raising
    # (see 6 below): the bar stands whatever the recompute does.
    try:
        from .probe_monitor_run import recompute_rung

        recompute_rung(db, locked.id)
    except Exception:  # noqa: BLE001 - the re-cut is committed; the rung is recomputed on the next evaluation
        logger.exception("probe %s: re-cut committed; its rung could not be recomputed at the new bar",
                         probe.id)

    identity_new = {"decision_digest": snapshot["digest"], "decision_revision": revision}
    try:
        copies = _millm_copies_field(millm_read.imported_copies(locked.id))
        next_list = after_recut(
            locked, committed=True, had_definition=had_definition,
            decision=snapshot.get("decision"), decision_digest=snapshot["digest"],
            decision_revision=revision, millm_copies=copies,
        )
    except Exception as exc:  # noqa: BLE001 - the commit stands; the follow-ups are reported as unknown
        logger.exception("probe %s: re-cut committed (revision %s); its follow-up calls could not "
                         "be formed", probe.id, revision)
        copies = {"unknown": f"the follow-up calls could not be formed ({type(exc).__name__})"}
        next_list = []
    return {
        **proposal,
        "committed": True,
        "unchanged": False,
        "definition_invalidated": invalidated,
        "previous": previous,
        "current": {
            "threshold": locked.threshold,
            "target_fpr": locked.target_fpr,
            "realised_fpr": locked.realised_fpr,
            "threshold_source": locked.threshold_source,
            "revision": revision,
            **identity_new,
        },
        "proposed": {**proposed, "revision": revision, **identity_new},
        "detector_digest": getattr(locked, "built_detector_digest", None),
        "millm_copies": copies,
        "next": next_list,
    }


def recalibrate_probe(
    db: Any,
    probe_id: str,
    *,
    target_fpr: float,
    commit: bool,
    allow_fire_on_nothing: bool = False,
    reason: str = "",
    expect_decision_digest: Optional[str] = None,
) -> Dict[str, Any]:
    """Propose, and write only when asked. Synchronous: no model, no GPU lease, microseconds."""
    from ..models.probe_monitor import ProbeMonitor, ProbeMonitorEvaluation
    from .probe_decision import ensure_decision_snapshot

    probe = db.query(ProbeMonitor).filter(ProbeMonitor.id == probe_id).one_or_none()
    if probe is None:
        raise RecalibrationRefused(
            "probe_not_found", f"probe {probe_id} not found", status=404
        )
    # A probe cut before 039 has no stored decision yet; form it (once, compare-and-set) so the
    # preview can state the digest of the decision in force.
    ensure_decision_snapshot(db, probe)
    evaluations = (
        db.query(ProbeMonitorEvaluation)
        .filter(ProbeMonitorEvaluation.probe_id == probe_id)
        .all()
    )
    proposal = propose(
        probe, evaluations, target_fpr=target_fpr, allow_fire_on_nothing=allow_fire_on_nothing,
        db=db,
    )
    # ⚠ A COMMIT INVALIDATES THE CACHED DEFINITION, and a probe whose run predates precision
    # recording (2026-10-03) can never rebuild one — so for every such probe, moving the bar ends
    # its exportability. Said on the PREVIEW, where it can still change the decision (review
    # round 1, M3); not refused, because re-cutting the float16 negatives on disk is itself sound.
    from ..models.probe_monitor import ProbeMonitorRun
    from .probe_monitor_run import probe_precision

    run = db.query(ProbeMonitorRun).filter(ProbeMonitorRun.id == probe.run_id).first()
    rebuild_refusal = probe_precision(db, run)["refusal"] if run is not None else None
    proposal["definition_rebuildable"] = rebuild_refusal is None
    proposal["definition_rebuild_refusal"] = rebuild_refusal
    # ⚠ NOT REFUSED FOR THE RENDER EITHER (2026-10-08), for the same reason: this re-cuts the
    # probe's OWN stored negatives, rendered as it was trained, so the new bar is consistent with
    # its head. But a probe rendered without the generation prompt is SERVED with it, so any bar
    # it carries — the old one or this one — is applied by miLLM to a form it was not cut under.
    # Said here, where the operator chooses; and such a probe cannot rebuild its definition.
    from .probe_monitor_run import probe_render

    render_refusal = probe_render(run, probe)["refusal"]
    proposal["render_note"] = render_refusal
    if rebuild_refusal is None and render_refusal is not None:
        rebuild_refusal = render_refusal
        proposal["definition_rebuildable"] = False
        proposal["definition_rebuild_refusal"] = rebuild_refusal
    proposal["commit_ends_exportability"] = bool(probe.definition_path) and rebuild_refusal is not None
    if not commit:
        return {**proposal, "committed": False, "definition_invalidated": False}
    return apply(db, probe, proposal, reason=reason, expect_decision_digest=expect_decision_digest)
