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

    def __init__(self, code: str, detail: str, *, status: int = 409) -> None:
        super().__init__(detail)
        self.code = code
        self.detail = detail
        self.status = status


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
        decisions[window] = {
            **(entry or {}),
            "threshold": calibration.threshold,
            "target_fpr": calibration.target_fpr,
            "realised_fpr": calibration.realised_fpr,
            "n_negatives": calibration.n_negatives,
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
        missing.append("per-window bars for " + ", ".join(sorted(unrecuttable_windows)))
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


def propose(probe: Any, evaluations: Sequence[Any], *, target_fpr: float,
            allow_fire_on_nothing: bool = False) -> Dict[str, Any]:
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
    return {
        "probe_id": probe.id,
        "current": {
            "threshold": probe.threshold,
            "target_fpr": probe.target_fpr,
            "realised_fpr": probe.realised_fpr,
            "threshold_source": probe.threshold_source,
            "revision": next_revision(probe) - 1,
        },
        "proposed": {
            "threshold": calibration.threshold,
            "target_fpr": calibration.target_fpr,
            "realised_fpr": calibration.realised_fpr,
            "threshold_source": calibration.source,
            "revision": next_revision(probe),
            "n_negatives": calibration.n_negatives,
            "fires_on_nothing": calibration.threshold is None,
        },
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


def apply(db: Any, probe: Any, proposal: Dict[str, Any], *, reason: str = "") -> Dict[str, Any]:
    """Write a proposal onto the probe, and stale anything that stated the old bar.

    Order matters: the history entry is appended from the values still on the row, so it is
    built BEFORE they are overwritten.
    """
    from .probe_definition_builder import invalidate_definition

    proposed = proposal["proposed"]
    entry = {
        "at": datetime.now(timezone.utc).isoformat(),
        "revision": proposed["revision"],
        "from": {
            "threshold": probe.threshold,
            "target_fpr": probe.target_fpr,
            "realised_fpr": probe.realised_fpr,
        },
        "to": {
            "threshold": proposed["threshold"],
            "target_fpr": proposed["target_fpr"],
            "realised_fpr": proposed["realised_fpr"],
        },
        "n_negatives": proposed["n_negatives"],
        "threshold_source": proposed["threshold_source"],
        "windows_recut": sorted((proposal.get("window_decisions") or {}).keys()),
        "length_bands_recut": len(proposal.get("length_bands") or []),
        "reason": reason,
    }
    # ⚠ SEEDED WITH REVISION 1 ON THE FIRST RE-CUT. Without it an event stamped `revision 1`
    # would be unanswerable: the row carries only the current bar, so the history has to record
    # the one the run cut before it is replaced.
    history = list(probe.calibration_history or [])
    if not history:
        history.append({
            # ⚠ `created_at`, AND ONLY `created_at`. `ProbeMonitor` HAS NO `updated_at`.
            # The first version read `(probe.updated_at or probe.created_at)` behind a
            # `getattr(..., None) or getattr(..., None)` condition — which the EXISTING
            # `created_at` satisfied, so the value expression then touched the attribute that
            # does not exist and the first live commit returned a 500. A defensive condition in
            # front of a direct access is not a fallback; it is a guard that cannot fire.
            "at": probe.created_at.isoformat() if probe.created_at else None,
            "revision": 1,
            "from": None,
            "to": {
                "threshold": probe.threshold,
                "target_fpr": probe.target_fpr,
                "realised_fpr": probe.realised_fpr,
            },
            "n_negatives": None,
            "threshold_source": probe.threshold_source,
            "windows_recut": sorted((probe.window_decisions or {}).keys()),
            "length_bands_recut": len(probe.length_bands or []),
            "reason": "cut by the training run",
        })
    history.append(entry)

    moved = probe.threshold != proposed["threshold"]
    probe.threshold = proposed["threshold"]
    probe.target_fpr = proposed["target_fpr"]
    probe.realised_fpr = proposed["realised_fpr"]
    probe.threshold_source = proposed["threshold_source"]
    probe.window_decisions = proposal.get("window_decisions")
    probe.length_bands = proposal.get("length_bands")
    probe.calibration_history = history
    db.commit()

    # ⚠ THE SAME REASONING AS THE RUN'S OWN CALL, AND THE SAME REASON STRING. An exported
    # definition carries this operating point and a consumer has no way to tell it has moved.
    # Unconditional on `moved` would be wrong only in that it churns; unconditional on a
    # definition EXISTING would be wrong in the direction that matters.
    invalidated = False
    if moved and getattr(probe, "definition_path", None):
        invalidated = invalidate_definition(db, probe.id, reason="threshold recalibrated")

    logger.info(
        "probe %s recalibrated: threshold %s -> %s at target_fpr %s (revision %s, %s)",
        probe.id, entry["from"]["threshold"], entry["to"]["threshold"],
        proposed["target_fpr"], proposed["revision"], reason or "no reason given",
    )
    return {**proposal, "committed": True, "definition_invalidated": invalidated}


def recalibrate_probe(
    db: Any,
    probe_id: str,
    *,
    target_fpr: float,
    commit: bool,
    allow_fire_on_nothing: bool = False,
    reason: str = "",
) -> Dict[str, Any]:
    """Propose, and write only when asked. Synchronous: no model, no GPU lease, microseconds."""
    from ..models.probe_monitor import ProbeMonitor, ProbeMonitorEvaluation

    probe = db.query(ProbeMonitor).filter(ProbeMonitor.id == probe_id).one_or_none()
    if probe is None:
        raise RecalibrationRefused(
            "probe_not_found", f"probe {probe_id} not found", status=404
        )
    evaluations = (
        db.query(ProbeMonitorEvaluation)
        .filter(ProbeMonitorEvaluation.probe_id == probe_id)
        .all()
    )
    proposal = propose(
        probe, evaluations, target_fpr=target_fpr, allow_fire_on_nothing=allow_fire_on_nothing
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
    proposal["commit_ends_exportability"] = bool(probe.definition_path) and rebuild_refusal is not None
    if not commit:
        return {**proposal, "committed": False, "definition_invalidated": False}
    return apply(db, probe, proposal, reason=reason)
