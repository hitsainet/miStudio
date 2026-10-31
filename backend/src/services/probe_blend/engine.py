"""The blend engine: pure NumPy over members' stored per-row arrays (Feature 041 FR-8 – FR-15; FTID §3.5).

No database, no file, no global random state: inputs are arrays and plain dicts, every RNG is seeded
(`random_state=seed`), and every fit runs under `threadpool_limits(1)` — so two runs of one request give
the same figures to the bit (FR-15), whichever worker runs them.

⚠ ROWS ARE PAIRED BY SPLIT POSITION, NEVER BY ARRAY INDEX (T23). Each member's evaluation kept its own
rows (a render that failed, a filter); `align` inner-joins on positions with `searchsorted` and refuses a
member that kept too few of the shared rows (`Misaligned`) or two members that disagree on a label at one
position (`LabelsDisagree`).

⚠ THE SHARED HELPERS ARE CALLED, NEVER COPIED: `calibrate` (the order statistic a shipped bar is cut with,
not a quantile — miStudio memory `calibrate-is-an-order-statistic`), `auroc`, `wilson_interval` and
`decide_many` (each member's verdict at its RECORDED decision, the rule miLLM serves; OQ-3 (a)), and
`probe_breakdown.groups_of` for per-subgroup figures. `test_blend_engine.py` asserts each call by spy.

⚠ RULES ∪ CANDIDATES (`rules_or_*`) ARE 040'S UNION, NEVER A COPY: the version's cached rule verdicts are read BY
SPLIT POSITION (`rules_at`), made the union's rows by 040's `union_rows`, cut by 040's `union_cut` (the probe's bar
on the negatives the rules leave, the rules' own false positives charged first) and flagged by 040's `detectors`.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import candidates as cand

#: Recorded in the report (T27): the stack's regularisation and iteration budget are fixed.
STACK_C = 1.0
STACK_MAX_ITER = 2000


# ── errors (the service turns each into its refusal code) ────────────────────────────────────


class EngineRefused(Exception):
    code = "engine_error"


class Misaligned(EngineRefused):
    code = "members_misaligned"

    def __init__(self, member: str, view: str, n: int, aligned: int, fraction: float):
        super().__init__(f"member {member} kept {aligned} of its {n} rows on view {view} that every member "
                         f"scored ({aligned / max(n, 1):.3f} < {fraction})")
        self.details = {"member": member, "view": view, "rows": n, "aligned": aligned, "min_fraction": fraction}


class LabelsDisagree(EngineRefused):
    code = "labels_disagree"

    def __init__(self, view: str, position: int):
        super().__init__(f"two members' labels differ on view {view} at split position {position}")
        self.details = {"view": view, "position": int(position)}


class MemberConstant(EngineRefused):
    code = "member_scores_constant"

    def __init__(self, member: str):
        super().__init__(f"member {member}'s blend-fit scores have zero spread; standardisation is undefined")
        self.details = {"member": member}


class PositionsMissing(EngineRefused):
    code = "calibration_positions_not_recorded"

    def __init__(self, member: str, view: str):
        super().__init__(f"member {member}'s evaluation on view {view} has no row positions")
        self.details = {"member": member, "view": view}


# ── inputs ───────────────────────────────────────────────────────────────────────────────────


@dataclass
class MemberArrays:
    scores: np.ndarray
    positions: Optional[np.ndarray]
    labels: np.ndarray
    n_scored: np.ndarray


@dataclass
class Aligned:
    """One view after the inner join: shared positions (ascending), labels, and each member's columns."""

    view: str
    positions: np.ndarray
    labels: np.ndarray
    scores: np.ndarray          # rows × members, float64
    n_scored: np.ndarray        # rows × members, int64
    counts: Dict[str, Dict[str, int]] = field(default_factory=dict)


def align(view: str, members: Sequence[str], arrays: Dict[str, MemberArrays], min_fraction: float) -> Aligned:
    """Inner-join the members' rows of one view ON SPLIT POSITION (T23)."""
    sorted_positions: List[np.ndarray] = []
    orders: List[np.ndarray] = []
    for member in members:
        a = arrays[member]
        if a.positions is None:
            raise PositionsMissing(member, view)
        positions = np.asarray(a.positions, dtype=np.int64).reshape(-1)
        order = np.argsort(positions, kind="stable")
        sorted_p = positions[order]
        if sorted_p.size > 1 and np.any(sorted_p[1:] == sorted_p[:-1]):
            raise EngineRefused(f"member {member}'s positions on view {view} repeat a row")
        sorted_positions.append(sorted_p)
        orders.append(order)
    common = sorted_positions[0]
    for p in sorted_positions[1:]:
        common = np.intersect1d(common, p, assume_unique=True)
    scores = np.empty((common.size, len(members)), dtype=np.float64)
    n_scored = np.empty((common.size, len(members)), dtype=np.int64)
    labels: Optional[np.ndarray] = None
    counts: Dict[str, Dict[str, int]] = {}
    for j, member in enumerate(members):
        a = arrays[member]
        sorted_p, order = sorted_positions[j], orders[j]
        n = int(sorted_p.size)
        if n and common.size / n < min_fraction:
            raise Misaligned(member, view, n, int(common.size), min_fraction)
        index = order[np.searchsorted(sorted_p, common)]          # ⚠ BY POSITION
        scores[:, j] = np.asarray(a.scores, dtype=np.float64).reshape(-1)[index]
        n_scored[:, j] = np.asarray(a.n_scored).reshape(-1)[index]
        member_labels = np.asarray(a.labels).reshape(-1)[index].astype(np.int64)
        if labels is None:
            labels = member_labels
        else:
            differ = np.nonzero(labels != member_labels)[0]
            if differ.size:
                raise LabelsDisagree(view, int(common[differ[0]]))
        counts[member] = {"n": n, "aligned": int(common.size), "excluded": n - int(common.size)}
    return Aligned(view=view, positions=common, labels=labels if labels is not None else np.zeros(0, np.int64),
                   scores=scores, n_scored=n_scored, counts=counts)


# ── fitting ──────────────────────────────────────────────────────────────────────────────────


def standardise_fit(X: np.ndarray, members: Sequence[str]) -> Tuple[np.ndarray, np.ndarray]:
    """Mean and standard deviation (ddof 0) per member, float64; zero spread → `MemberConstant`."""
    X = np.asarray(X, dtype=np.float64)
    mean = X.mean(axis=0)
    std = X.std(axis=0, ddof=0)
    for j, s in enumerate(std.tolist()):
        if not s > 0:
            raise MemberConstant(members[j])
    return mean, std


def fit_stack(Z: np.ndarray, y: np.ndarray, seed: int) -> Tuple[np.ndarray, float]:
    """L2 logistic regression (lbfgs, C 1.0, 2000 iterations; L2 is scikit-learn's default), seeded, ONE BLAS
    thread (T27)."""
    from sklearn.linear_model import LogisticRegression
    from threadpoolctl import threadpool_limits

    with threadpool_limits(limits=1):
        # L2 is the default (scikit-learn 1.8 deprecated the `penalty` argument; naming it warns).
        model = LogisticRegression(C=STACK_C, solver="lbfgs", max_iter=STACK_MAX_ITER, random_state=seed)
        model.fit(np.asarray(Z, dtype=np.float64), np.asarray(y, dtype=np.int64))
    return np.asarray(model.coef_[0], dtype=np.float64), float(model.intercept_[0])


def apply(candidate: cand.Candidate, params: Dict[str, Any], scores: np.ndarray, n_fired: np.ndarray) -> np.ndarray:
    """A candidate's per-row value: the blended score (score kinds) or the members firing (verdict kinds)."""
    if candidate.produces == "verdict":
        return np.asarray(n_fired, dtype=np.float64)
    mean = np.asarray(params["mean"], dtype=np.float64)
    std = np.asarray(params["std"], dtype=np.float64)
    Z = (np.asarray(scores, dtype=np.float64) - mean) / std
    if candidate.kind == "stack":
        return Z @ np.asarray(params["weights"], dtype=np.float64) + float(params["intercept"])
    if candidate.kind == "max":
        return Z.max(axis=1)
    if candidate.kind == "best_single":
        return Z[:, int(params["member_index"])]
    raise EngineRefused(f"no value rule for {candidate.kind}")


def fires_of(candidate: cand.Candidate, values: np.ndarray, threshold: Any) -> np.ndarray:
    """Score kinds: `value >= threshold` (no threshold fires nothing); verdict kinds: `n_fired >= k`; a rules ∪
    candidate's value IS its final verdict (0/1)."""
    values = np.asarray(values, dtype=np.float64)
    if candidate.rules:
        return values >= 0.5
    if candidate.produces == "verdict":
        return values >= float(candidate.k)
    if threshold is None:
        return np.zeros(values.shape[0], dtype=bool)
    return values >= float(threshold)


def member_verdicts(decision: Dict[str, Any], scores: np.ndarray, n_scored: np.ndarray) -> np.ndarray:
    """One member's verdict per row at its RECORDED decision (OQ-3 (a)), through 040's `decide_many`."""
    from ..probe_monitor_metrics import decide_many

    decisions = decide_many(decision["probe_bar"], decision["window_entry"], decision["probe_bands"],
                            decision["probe_scope"], decision["window"], n_scored)
    return decisions.fires(scores)


# ── rules ∪ candidates (040's union) ──────────────────────────────────────────────────────────


@dataclass
class RulesAt:
    """One view's rule verdicts on the ALIGNED rows: the union's match and veto (040's `union_rows`) and the
    unknown codes (0 known)."""

    match: np.ndarray
    veto: np.ndarray
    unknown: np.ndarray


def rules_at(rules: Any, positions: np.ndarray) -> RulesAt:
    """A view's cached verdicts (indexed by split position) at the aligned rows — ⚠ BY POSITION (FR-20)."""
    from ..rules_evaluation import union_rows

    positions = np.asarray(positions, dtype=np.int64)
    unknown = np.asarray(rules.unknown)[positions]
    match, veto = union_rows(np.asarray(rules.rule_match, dtype=bool)[positions],
                             np.asarray(rules.rule_veto, dtype=bool)[positions], unknown)
    return RulesAt(match=match, veto=veto, unknown=unknown)


def union_fires(rules: RulesAt, fires: np.ndarray) -> np.ndarray:
    """`rule_match OR (fires AND NOT rule_veto)` — 040's `detectors`, an unknown verdict letting the blend decide."""
    from ..rules_evaluation import detectors

    return detectors(rules.match, rules.veto, rules.unknown, np.asarray(fires, dtype=bool))["rules_or_probe"]


def _subset(rules: RulesAt, mask: np.ndarray) -> RulesAt:
    return RulesAt(match=rules.match[mask], veto=rules.veto[mask], unknown=rules.unknown[mask])


def n_fired_of(decisions: Sequence[Dict[str, Any]], aligned: Aligned) -> np.ndarray:
    total = np.zeros(aligned.positions.size, dtype=np.int64)
    for j, decision in enumerate(decisions):
        total += member_verdicts(decision, aligned.scores[:, j], aligned.n_scored[:, j]).astype(np.int64)
    return total


# ── folds and cross-validation ───────────────────────────────────────────────────────────────


def folds(groups: Optional[np.ndarray], y: np.ndarray, k: int, seed: int,
          min_rows_per_class: int) -> Tuple[np.ndarray, List[str], Dict[str, Any]]:
    """Fold id per row (int32), flags and counts. Grouped (`StratifiedGroupKFold`) when group values
    exist — a group never straddles folds; else `StratifiedKFold` and `ungrouped_folds` (T25: stated,
    never silent). `few_groups` / `few_rows` flag; NEVER refuse (FR-11)."""
    from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

    y = np.asarray(y, dtype=np.int64)
    flags: List[str] = []
    counts: Dict[str, Any] = {"rows": int(y.size), "positives": int((y == 1).sum()), "negatives": int((y == 0).sum())}
    fold_ids = np.full(y.size, -1, dtype=np.int32)
    n_classes_min = min(counts["positives"], counts["negatives"])
    if groups is not None:
        n_groups = int(np.unique(groups).size)
        counts["groups"] = n_groups
        if n_groups < k:
            flags.append("few_groups")
            return fold_ids, flags, counts
        splitter = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
        splits = splitter.split(np.zeros(y.size), y, groups)
    else:
        flags.append("ungrouped_folds")
        if n_classes_min < k:
            flags.append("few_rows")
            return fold_ids, flags, counts
        splitter = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
        splits = splitter.split(np.zeros(y.size), y)
    for i, (_train, test) in enumerate(splits):
        fold_ids[test] = i
    per_fold = []
    for i in range(k):
        held = y[fold_ids == i]
        per_fold.append({"positives": int((held == 1).sum()), "negatives": int((held == 0).sum())})
    counts["folds"] = per_fold
    if any(min(f["positives"], f["negatives"]) < min_rows_per_class for f in per_fold):
        flags.append("few_rows")
    return fold_ids, flags, counts


def assignment_digest(fold_ids: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(fold_ids, dtype=np.int32).tobytes()).hexdigest()


def _recall_at(values_pos: np.ndarray, values_neg: np.ndarray, target_fpr: float) -> Dict[str, Any]:
    from ..probe_monitor_trainer import calibrate

    cut = calibrate(values_neg.tolist(), target_fpr=target_fpr, source="blend_cv")
    threshold = cut.threshold
    recall = float((values_pos >= threshold).mean()) if (threshold is not None and values_pos.size) else 0.0
    return {"recall": recall, "threshold": threshold, "realised_fpr": float(cut.realised_fpr)}


def cv(candidate: cand.Candidate, fit: Aligned, n_fired: np.ndarray, fold_ids: np.ndarray, k: int, seed: int,
       target_fpr: float, members: Sequence[str], best_index: Optional[int],
       should_stop: Callable[[], None], beat: Callable[[], None]) -> Dict[str, Any]:
    """In-fold figures on `blend_fit` (labelled `in_fold`): standardisation AND stack refitted inside each
    training fold, out-of-fold values pooled; AUROC by `auroc`, recall at the out-of-fold operating point
    by `calibrate` on the pooled out-of-fold negatives."""
    from ..probe_monitor_metrics import auroc

    y = fit.labels
    if candidate.produces == "verdict":
        fires = n_fired >= int(candidate.k)
        pos, neg = fires[y == 1], fires[y == 0]
        return {"in_fold": True, "auroc": None, "auroc_reason": "verdict_only",
                "recall": float(pos.mean()) if pos.size else None,
                "realised_fpr": float(neg.mean()) if neg.size else None}
    oof = out_of_fold(candidate, fit, fold_ids, k, seed, members, best_index, should_stop, beat)
    at = _recall_at(oof[y == 1], oof[y == 0], target_fpr)
    return {"in_fold": True, "auroc": auroc(oof.tolist(), y.tolist()), "recall": at["recall"],
            "realised_fpr": at["realised_fpr"]}


def cv_union(candidate: cand.Candidate, fit: Aligned, n_fired: np.ndarray, rules: RulesAt, fold_ids: np.ndarray,
             k: int, seed: int, target_fpr: float, members: Sequence[str], best_index: Optional[int],
             should_stop: Callable[[], None], beat: Callable[[], None]) -> Dict[str, Any]:
    """In-fold figures of a rules ∪ candidate on `blend_fit`: the base's out-of-fold values (refitted inside each
    training fold), the union cut by 040's `union_cut` on the pooled out-of-fold negatives, the union's recall."""
    from ..detector_next import DetectorRefused
    from ..rules_evaluation import refuse_rules_over_budget, union_cut

    base = cand.base_of(candidate)
    y = fit.labels
    neg = y == 0
    if base.produces == "verdict":
        fires = union_fires(rules, n_fired >= int(base.k))
    else:
        oof = out_of_fold(base, fit, fold_ids, k, seed, members, best_index, should_stop, beat)
        try:
            refuse_rules_over_budget(rules.match[neg], target_fpr)
        except DetectorRefused as refusal:
            return {"in_fold": True, "not_assessable": True, "reason": [refusal.code], **refusal.details}
        cut = union_cut(oof[neg], rules.match[neg], rules.veto[neg], target_fpr)
        base_fires = (oof >= cut["threshold"]) if cut["threshold"] is not None else np.zeros(y.size, dtype=bool)
        fires = union_fires(rules, base_fires)
    pos_f, neg_f = fires[y == 1], fires[neg]
    return {"in_fold": True, "auroc": None, "auroc_reason": "verdict_only",
            "recall": float(pos_f.mean()) if pos_f.size else None,
            "realised_fpr": float(neg_f.mean()) if neg_f.size else None}


def out_of_fold(candidate: cand.Candidate, fit: Aligned, fold_ids: np.ndarray, k: int, seed: int,
                members: Sequence[str], best_index: Optional[int], should_stop: Callable[[], None],
                beat: Callable[[], None]) -> np.ndarray:
    """A score kind's out-of-fold values: standardisation AND stack refitted inside each training fold."""
    y = fit.labels
    oof = np.empty(y.size, dtype=np.float64)
    for i in range(k):
        should_stop()
        beat()
        train, test = fold_ids != i, fold_ids == i
        mean, std = standardise_fit(fit.scores[train], members)
        Z_train = (fit.scores[train] - mean) / std
        Z_test = (fit.scores[test] - mean) / std
        if candidate.kind == "stack":
            weights, intercept = fit_stack(Z_train, y[train], seed)
            oof[test] = Z_test @ weights + intercept
        elif candidate.kind == "max":
            oof[test] = Z_test.max(axis=1)
        else:
            oof[test] = Z_test[:, int(best_index)]
    return oof


def best_single(fit: Aligned, fold_ids: np.ndarray, k: int, target_fpr: float, members: Sequence[str],
                cv_ok: bool) -> Tuple[int, List[Dict[str, Any]]]:
    """P1: the member with the highest `blend_fit` recall at the target FPR (out-of-fold when the folds are
    assessable, else in-sample), ties by member probe id."""
    y = fit.labels
    rows: List[Dict[str, Any]] = []
    for j, member in enumerate(members):
        values = fit.scores[:, j]
        if cv_ok:
            oof = np.empty(y.size, dtype=np.float64)
            for i in range(k):
                train, test = fold_ids != i, fold_ids == i
                mean, std = values[train].mean(), values[train].std(ddof=0)
                oof[test] = (values[test] - mean) / (std if std > 0 else 1.0)
            values = oof
        at = _recall_at(values[y == 1], values[y == 0], target_fpr)
        rows.append({"probe_id": member, "recall": at["recall"]})
    order = sorted(range(len(members)), key=lambda j: (-rows[j]["recall"], members[j]))
    return order[0], rows


# ── calibration and test figures ─────────────────────────────────────────────────────────────


def calibrate_candidate(candidate: cand.Candidate, values: np.ndarray, labels: np.ndarray, target_fpr: float,
                        floor: int, rules: Optional[RulesAt] = None) -> Dict[str, Any]:
    """Score kinds: one global threshold by `calibrate` on the calibration NEGATIVES. Verdict kinds: the
    realised false-positive rate of `n_fired >= k` on them. Each records its count and `few_negatives`.

    Rules ∪ candidates (`values` are the BASE's values): the union through 040 — `union_cut` for a score base
    (the rules' false positives charged first, vetoed negatives removed, the base's bar cut on what is left;
    `rules_spend_budget` recorded, never a bar), the union's realised rate for a verdict base."""
    from ..probe_monitor_trainer import calibrate

    is_negative = np.asarray(labels) == 0
    negatives = np.asarray(values, dtype=np.float64)[is_negative]
    n = int(negatives.size)
    out: Dict[str, Any] = {"n_negatives": n, "few_negatives": n < int(floor), "floor": int(floor),
                           "target_fpr": target_fpr}
    if n == 0:
        raise EngineRefused("the calibration view has no negatives on the rows every member scored")
    if candidate.rules:
        if rules is None:
            raise EngineRefused(f"{candidate.name} needs the calibration view's rule verdicts")
        return _calibrate_union(candidate, negatives, _subset(rules, is_negative), target_fpr, out)
    if candidate.produces == "verdict":
        out.update(threshold=None, k=candidate.k, source="realised",
                   realised_fpr=float((negatives >= float(candidate.k)).mean()))
        return out
    cut = calibrate(negatives.tolist(), target_fpr=target_fpr, source="calibration_set")
    out.update(threshold=cut.threshold, k=None, source="calibration_set", realised_fpr=float(cut.realised_fpr))
    return out


def _calibrate_union(candidate: cand.Candidate, negatives: np.ndarray, rules: RulesAt, target_fpr: float,
                     out: Dict[str, Any]) -> Dict[str, Any]:
    from ..detector_next import DetectorRefused
    from ..rules_evaluation import refuse_rules_over_budget, union_cut

    out["rule_false_positives"] = int(rules.match.sum())
    out["vetoed"] = int(rules.veto.sum())
    out["unknown"] = int((rules.unknown != 0).sum())
    if candidate.produces == "verdict":
        fires = union_fires(rules, negatives >= float(candidate.k))
        out.update(threshold=None, k=candidate.k, source="realised_union", realised_fpr=float(fires.mean()))
        return out
    try:
        refuse_rules_over_budget(rules.match, target_fpr)
    except DetectorRefused as refusal:
        out.update(threshold=None, k=None, source="union_calibration", realised_fpr=None, refused=refusal.code,
                   **refusal.details)
        return out
    cut = union_cut(negatives, rules.match, rules.veto, target_fpr)                 # ⚠ 040's cut, never a copy
    out.update(threshold=cut["threshold"], k=None, source="union_calibration",
               realised_fpr=float(cut["realised_union_fpr"]),
               union={key: cut[key] for key in ("budget", "rule_false_positives", "vetoed",
                                                "probe_false_positives_allowed", "probe_false_positives",
                                                "rule_share", "probe_alone_fpr_at_bar")})
    return out


def test_figures(candidate: Optional[cand.Candidate], values: np.ndarray, fires: np.ndarray,
                 labels: np.ndarray) -> Dict[str, Any]:
    """Recall with Wilson's 95% interval, AUROC (`null` + `verdict_only` for a verdict), counts, FPR."""
    from ..probe_monitor_metrics import auroc, wilson_interval

    labels = np.asarray(labels)
    pos, neg = labels == 1, labels == 0
    n_pos, n_neg = int(pos.sum()), int(neg.sum())
    hits = int((fires & pos).sum())
    lo, hi = wilson_interval(hits, n_pos)
    verdict = candidate is not None and (candidate.produces == "verdict" or candidate.rules)
    return {
        "recall": hits / n_pos if n_pos else None,
        "wilson": [lo, hi] if n_pos else None,
        "auroc": None if verdict else (auroc(np.asarray(values, dtype=np.float64).tolist(), labels.tolist())
                                       if n_pos and n_neg else None),
        "auroc_reason": "verdict_only" if verdict else None,
        "n_pos": n_pos, "n_neg": n_neg,
        "realised_fpr": float((fires & neg).sum()) / n_neg if n_neg else None,
    }


def rank(per_candidate: Dict[str, Dict[str, Dict[str, Any]]], held_out: Sequence[str],
         in_distribution: Sequence[str]) -> Tuple[List[str], str]:
    """FR-13 / L8: worst held-out recall first (descending), then the worst Wilson lower bound, then the
    name. No held-out view (OQ-2 (a)): the same over in-distribution views, labelled so."""
    basis_views, basis = (list(held_out), "held_out_min") if held_out else (list(in_distribution),
                                                                            "in_distribution_only")

    def key(name: str) -> Tuple[float, float, str]:
        figures = [per_candidate[name][v] for v in basis_views if v in per_candidate[name]]
        recalls = [f["recall"] if f["recall"] is not None else 0.0 for f in figures] or [0.0]
        lows = [f["wilson"][0] if f["wilson"] else 0.0 for f in figures] or [0.0]
        return (-min(recalls), -min(lows), name)

    return sorted(per_candidate, key=key), basis


# ── the whole run ────────────────────────────────────────────────────────────────────────────


@dataclass
class EngineInputs:
    members: List[str]
    #: Per member, `decide_many`'s arguments (`probe_breakdown.decision_inputs`).
    decisions: List[Dict[str, Any]]
    blend_fit: str
    tests: List[Dict[str, Any]]                         # [{id, held_out}]
    calibration: str
    arrays: Dict[str, Dict[str, MemberArrays]]          # view → member → arrays
    candidates: List[cand.Candidate]
    folds: int
    seed: int
    target_fpr: float
    min_rows_per_class: int
    min_aligned_fraction: float
    floor: int
    #: The blend_fit view's group code per SPLIT POSITION (None: no group column → ungrouped folds).
    group_codes: Optional[np.ndarray] = None
    #: test view → column → (codes per split position, values) for per-subgroup figures.
    columns: Dict[str, Dict[str, Tuple[np.ndarray, List[Any]]]] = field(default_factory=dict)
    min_subgroup_rows: int = 30
    #: Blend criteria to judge each candidate against: [{id, detector, held_out_test, subset, min_recall,
    #: target_fpr}] — `blend_alone` on the probe-only candidates, `rules_or_blend` on the rules ∪ ones; subset a
    #: column filter, "all", or a rule subset (`rule_invisible_positives` / `rule_visible_positives`).
    criteria: List[Dict[str, Any]] = field(default_factory=list)
    #: The version names a rule set (whether or not its verdicts were loaded).
    has_rule_set: bool = False
    #: view → the version's cached rule verdicts indexed by SPLIT POSITION (`detector_success_report.RuleRows`),
    #: loaded when a rules ∪ candidate or a rule subset needs them; None otherwise.
    rules: Optional[Dict[str, Any]] = None
    #: {rule_set_digest, engine_digest}: recorded in each rules ∪ candidate's parameters.
    rules_identity: Optional[Dict[str, Any]] = None
    rule_unknown_share_max: float = 1.0


def _noop() -> None:
    return None


def run(inputs: EngineInputs, *, should_stop: Callable[[], None] = _noop,
        beat: Callable[[], None] = _noop) -> Dict[str, Any]:
    """The pipeline: align → standardise and fit on `blend_fit` → grouped CV → calibrate on the shared
    calibration sample → figures on every test view → rank. `should_stop()` and `beat()` are called
    between folds and between candidates (T32)."""
    from ..probe_breakdown import groups_of

    members = list(inputs.members)
    views = [inputs.blend_fit, *[t["id"] for t in inputs.tests], inputs.calibration]
    aligned = {v: align(v, members, inputs.arrays[v], inputs.min_aligned_fraction) for v in views}
    fit = aligned[inputs.blend_fit]
    mean, std = standardise_fit(fit.scores, members)
    weights, intercept = fit_stack((fit.scores - mean) / std, fit.labels, inputs.seed)
    n_fired = {v: n_fired_of(inputs.decisions, aligned[v]) for v in views}
    rule_at = ({v: rules_at(inputs.rules[v], aligned[v].positions) for v in views}
               if inputs.rules is not None else None)

    groups = None
    if inputs.group_codes is not None:
        groups = np.asarray(inputs.group_codes)[fit.positions]           # ⚠ BY POSITION
    fold_ids, fold_flags, fold_counts = folds(groups, fit.labels, inputs.folds, inputs.seed, inputs.min_rows_per_class)
    cv_ok = not ({"few_groups", "few_rows"} & set(fold_flags))
    best_index, best_rows = best_single(fit, fold_ids, inputs.folds, inputs.target_fpr, members, cv_ok)

    params_base = {"mean": mean.tolist(), "std": std.tolist(), "C": STACK_C, "max_iter": STACK_MAX_ITER}
    out_candidates: List[Dict[str, Any]] = []
    values_by_view: Dict[str, Dict[int, np.ndarray]] = {v: {} for v in views}
    per_candidate_tests: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for index, candidate in enumerate(inputs.candidates):
        should_stop()
        beat()
        if candidate.rules and rule_at is None:
            raise EngineRefused(f"{candidate.name} needs the version's rule verdicts, which were not loaded")
        base = cand.base_of(candidate)
        params: Dict[str, Any] = dict(params_base)
        if base.kind == "stack":
            params.update(weights=weights.tolist(), intercept=intercept)
        if base.kind == "best_single":
            params.update(member_index=best_index, member=members[best_index])
        if base.produces == "verdict":
            params = {"k": base.k}
        if candidate.rules:
            params["rules"] = dict(inputs.rules_identity or {})
        base_values = {v: apply(base, {**params_base, **params}, aligned[v].scores, n_fired[v]) for v in views}
        if cv_ok:
            if candidate.rules:
                cv_figures = cv_union(candidate, fit, n_fired[inputs.blend_fit], rule_at[inputs.blend_fit], fold_ids,
                                      inputs.folds, inputs.seed, inputs.target_fpr, members, best_index,
                                      should_stop, beat)
            else:
                cv_figures = cv(candidate, fit, n_fired[inputs.blend_fit], fold_ids, inputs.folds, inputs.seed,
                                inputs.target_fpr, members, best_index, should_stop, beat)
        else:
            cv_figures = {"in_fold": True, "not_assessable": True, "reason": list(fold_flags), **fold_counts}
        calibration = calibrate_candidate(candidate, base_values[inputs.calibration],
                                          aligned[inputs.calibration].labels, inputs.target_fpr, inputs.floor,
                                          rules=rule_at[inputs.calibration] if candidate.rules else None)
        if candidate.rules:
            # The final verdict per row: rules ∪ (the base fires at the union bar) — stored as 0/1 (FTDD §5).
            values = {v: union_fires(rule_at[v], fires_of(base, base_values[v], calibration["threshold"]))
                      .astype(np.float64) for v in views}
        else:
            values = base_values
        for v in views:
            values_by_view[v][index] = values[v]
        tests: Dict[str, Dict[str, Any]] = {}
        for test in inputs.tests:
            v = test["id"]
            fires = fires_of(candidate, values[v], calibration["threshold"])
            figures = test_figures(candidate, values[v], fires, aligned[v].labels)
            breakdown = {}
            for column, (codes, column_values) in sorted((inputs.columns.get(v) or {}).items()):
                breakdown[column] = groups_of(np.asarray(codes)[aligned[v].positions], aligned[v].labels == 1, fires,
                                              values[v], np.ones(fires.shape[0], dtype=bool), column_values,
                                              inputs.min_subgroup_rows)
            figures["breakdown"] = breakdown
            tests[v] = figures
        per_candidate_tests[candidate.name] = tests
        # `blend_alone` is judged on the probe-only candidates, `rules_or_blend` on the rules ∪ ones.
        form = "rules_or_blend" if candidate.rules else "blend_alone"
        rule_match = {v: rule_at[v].match for v in views} if rule_at is not None else None
        criteria = [judge_criterion(c, candidate, tests, inputs, values, aligned, calibration, rule_match)
                    for c in inputs.criteria if c.get("detector", "blend_alone") == form]
        out_candidates.append({
            "index": index, "name": candidate.name, "kind": candidate.kind,
            "produces": "verdict" if candidate.rules else candidate.produces,
            "params": params if (candidate.produces == "score" or candidate.rules) else {"k": candidate.k},
            "calibration": calibration, "cv": cv_figures, "tests": tests, "criteria": criteria,
        })

    members_own: Dict[str, Dict[str, Any]] = {}
    for test in inputs.tests:
        v = test["id"]
        members_own[v] = {}
        for j, member in enumerate(members):
            fires = member_verdicts(inputs.decisions[j], aligned[v].scores[:, j], aligned[v].n_scored[:, j])
            members_own[v][member] = test_figures(None, aligned[v].scores[:, j], fires, aligned[v].labels)
    held = [t["id"] for t in inputs.tests if t.get("held_out")]
    ind = [t["id"] for t in inputs.tests if not t.get("held_out")]
    order, basis = rank(per_candidate_tests, held, ind)
    rows: Dict[str, Dict[str, Any]] = {
        v: {"positions": aligned[v].positions, "labels": aligned[v].labels, "values": values_by_view[v]} for v in views}
    rules_summary = None
    if rule_at is not None:
        rules_summary = {}
        for v in views:
            rows[v]["rule_match"] = rule_at[v].match
            n_rows = int(rule_at[v].unknown.shape[0])
            unknown = int((rule_at[v].unknown != 0).sum())
            share = unknown / n_rows if n_rows else 0.0
            rules_summary[v] = {"rows": n_rows, "rule_match": int(rule_at[v].match.sum()),
                                "rule_veto": int(rule_at[v].veto.sum()), "unknown": unknown, "share": share,
                                "over_share": share > float(inputs.rule_unknown_share_max),
                                "pending": int((rule_at[v].unknown == 2).sum()),
                                "producer_computed": bool(getattr(inputs.rules[v], "producer_computed", False))}
    return {
        "alignment": {v: {"rows": int(aligned[v].positions.size), "members": aligned[v].counts} for v in views},
        "standardisation": {m: {"mean": float(mean[j]), "std": float(std[j])} for j, m in enumerate(members)},
        "folds": {"k": inputs.folds, "seed": inputs.seed, "assignment_digest": assignment_digest(fold_ids),
                  "flags": fold_flags, "counts": fold_counts},
        "best_single": {"member": members[best_index], "by_member": best_rows},
        "candidates": out_candidates,
        "members_own_figures": members_own,
        "ranking": {"order": order, "basis": basis},
        "rows": rows,
        "rules": rules_summary,
    }


def judge_criterion(criterion: Dict[str, Any], candidate: cand.Candidate, tests: Dict[str, Dict[str, Any]],
                    inputs: EngineInputs, values: Dict[str, np.ndarray], aligned: Dict[str, Aligned],
                    calibration: Dict[str, Any], rule_match: Optional[Dict[str, np.ndarray]] = None) -> Dict[str, Any]:
    """A pure blend criterion (`blend_alone`) judged on this candidate as 040's success report judges a probe:
    the criterion's test views (held-out → out-of-distribution, else in-distribution), its subset, the
    point estimate against `min_recall` (FR-22 of 040), with Wilson's interval beside it."""
    return blend_criterion_verdict(criterion, candidate, inputs.tests, inputs.columns, values, aligned, calibration,
                                   rule_match=rule_match, has_rule_set=inputs.has_rule_set)


def blend_criterion_verdict(criterion: Dict[str, Any], candidate: cand.Candidate, tests: Sequence[Dict[str, Any]],
                            columns: Dict[str, Dict[str, Tuple[np.ndarray, List[Any]]]],
                            values: Dict[str, np.ndarray], aligned: Dict[str, Any],
                            calibration: Dict[str, Any], *, rule_match: Optional[Dict[str, np.ndarray]] = None,
                            has_rule_set: bool = False) -> Dict[str, Any]:
    """THE ONE blend-criterion rule — the report and 040's success report (FR-20 branch) both call it.

    A rule subset (`rule_invisible_positives` / `rule_visible_positives`, 040 FR-5) selects the positives the
    version's rules do not / do see, from `rule_match` on the aligned rows; with no rule set every rule verdict is
    false (040 FR-33), so the invisible positives are all of them and the visible ones cannot be judged."""
    from ..detector_success_report import RULE_SUBSETS
    from ..probe_monitor_metrics import wilson_interval
    from ..probe_breakdown import _value_key

    entry: Dict[str, Any] = {"id": criterion["id"], "detector": criterion.get("detector", "blend_alone"),
                             "min_recall": float(criterion["min_recall"]), "reasons": []}
    target = criterion.get("target_fpr")
    if target is not None and abs(float(target) - float(calibration["target_fpr"])) > 1e-12:
        entry["reasons"].append({"code": "target_fpr_mismatch", "criterion_target_fpr": target,
                                 "blend_target_fpr": calibration["target_fpr"]})
    if calibration.get("refused"):
        entry["reasons"].append({"code": calibration["refused"]})
    realised = calibration.get("realised_fpr")
    if candidate.rules and realised is not None and float(realised) > float(calibration["target_fpr"]) + 1e-12:
        # 040-T27: a union above the target on the calibration negatives cannot be judged at that target.
        entry["reasons"].append({"code": "union_over_budget", "realised_union_fpr": float(realised),
                                 "target_fpr": calibration["target_fpr"]})
    held = criterion.get("held_out_test")
    chosen = [t["id"] for t in tests if bool(t.get("held_out")) == bool(held)]
    entry["test_view_ids"] = chosen
    hits = n = 0
    subset = criterion.get("subset", "all")
    for v in chosen:
        fires = fires_of(candidate, values[v], calibration["threshold"])
        positives = np.asarray(aligned[v].labels) == 1
        mask = np.ones(positives.shape[0], dtype=bool)
        if isinstance(subset, dict):
            column = (columns.get(v) or {}).get(subset["column"])
            if column is None:
                entry["reasons"].append({"code": "column_not_declared", "column": subset["column"], "view_id": v})
                continue
            codes, column_values = column
            wanted = [i for i, value in enumerate(column_values) if _value_key(value) == _value_key(subset["value"])]
            mask &= np.isin(np.asarray(codes)[aligned[v].positions], wanted)
        elif isinstance(subset, str) and subset in RULE_SUBSETS:
            if rule_match is not None and v in rule_match:
                seen = np.asarray(rule_match[v], dtype=bool)
                mask &= ~seen if subset == "rule_invisible_positives" else seen
            elif not has_rule_set:
                if "no_rule_set" not in entry.setdefault("notes", []):
                    entry["notes"].append("no_rule_set")
                if subset == "rule_visible_positives":
                    entry["reasons"].append({"code": "no_rule_set"})
                    continue
            else:
                entry["reasons"].append({"code": "missing_evaluation", "missing": "rules_evaluation", "view_id": v})
                continue
        n += int((positives & mask).sum())
        hits += int((positives & mask & fires).sum())
    if not chosen:
        entry["reasons"].append({"code": "missing_evaluation", "missing": "test_view",
                                 "wanted": "out_of_distribution" if held else "in_distribution"})
    recall = hits / n if n else None
    lo, hi = wilson_interval(hits, n) if n else (None, None)
    if n == 0 and chosen:
        entry["reasons"].append({"code": "missing_evaluation", "missing": "positives"})
    entry["figures"] = {"positives": n, "flagged_positives": hits, "recall": recall,
                        "recall_ci": [lo, hi] if n else None}
    if entry["reasons"]:
        entry["verdict"] = "not_assessable"
    else:
        entry["verdict"] = "met" if recall >= float(criterion["min_recall"]) else "not_met"
    return entry
