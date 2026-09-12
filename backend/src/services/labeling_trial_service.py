"""Prompt-template trials: run one template over a fixed panel, write no label.

A trial answers "would THIS template label these features better?" without
touching the labels being compared against. Running five variants over a panel
would otherwise overwrite the user's real labels five times, and the fifth
variant would be scored against features the first four had rewritten.

Deliberately built ALONGSIDE `LabelingService.label_features_for_extraction`
rather than inside it. That method is 900 lines with three near-identical
persistence branches, pinned by AST- and source-scraping tests, and it exists to
WRITE labels. Threading a "do not write" flag through it would put the apply
path one boolean away from silently not persisting — the highest-cost failure
available in this subsystem. A trial reuses the pieces that generate a label and
owns its own, much shorter, non-persisting path.

The no-write property is asserted at runtime, not assumed: after each batch the
session must hold no dirty `Feature`. Skipping `commit()` is not enough on its
own, because these are ORM attribute assignments and any later commit in the
same session would flush them.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Session

from ..models.extraction_job import ExtractionJob, ExtractionStatus
from ..models.feature import Feature
from ..models.labeling_job import LabelingJob, LabelingMode, LabelingStatus
from ..models.labeling_prompt_template import LabelingPromptTemplate
from ..models.labeling_trial_run import LabelingTrialRun
from .labeling_fingerprint import prompt_fingerprint_fields

logger = logging.getLogger(__name__)

MAX_PANEL_SIZE = 200
LABEL_BATCH_SIZE = 10

#: How many top-K passages the detection scorer grades a label on.
#:
#: FIXED, and independent of the template's own `max_examples`. If the scoring
#: set moved with the arm, two arms would be graded on different passages and
#: the delta between them would measure the instrument rather than the label.
SCORING_POSITIVES_K = 10

#: The template fields the RENDERER reads. One definition, imported by every
#: site that builds a `template_config`.
#:
#: There were three copies of this list — here, and twice in `labeling_service`
#: (the bulk path and its no-template fallback) — and they had already drifted:
#: the bulk copies omit `include_nlp_analysis`, so a template configured for NLP
#: analysis silently ran without it. A key added to one copy and not the others
#: produces a trial that does not predict the bulk run it is trialling.
#:
#: NOT the same set as the fingerprint. The fingerprint covers everything that
#: changes what the judge is SENT, which includes `system_message`,
#: `user_prompt_template` and the sampling parameters consumed elsewhere. This
#: is the narrower set the formatter itself reads.
TEMPLATE_CONFIG_KEYS = (
    "template_type",
    "max_examples",
    "include_prefix",
    "include_suffix",
    "prime_token_marker",
    "include_logit_effects",
    "top_promoted_tokens_count",
    "top_suppressed_tokens_count",
    "include_negative_examples",
    "num_negative_examples",
    "is_detection_template",
    "include_nlp_analysis",
    "example_sampling",
    "activation_display",
)
_TEMPLATE_CONFIG_KEYS = TEMPLATE_CONFIG_KEYS


class TrialError(Exception):
    """A trial cannot be started or run."""


class TrialWroteToFeatures(RuntimeError):
    """The guard that makes 'writes nothing' a fact rather than an intention."""


def panel_id_for(extraction_job_id: str, feature_ids: Sequence[str]) -> str:
    """Content-addressed panel identity.

    Equal ids PROVE an identical, order-independent, extraction-bound feature
    set, so `compare` can refuse a mismatched pair outright instead of trusting
    that two runs happened to cover the same features.
    """
    payload = f"{extraction_job_id}|{','.join(sorted(feature_ids))}"
    return "pnl_" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def freeze_template(template: LabelingPromptTemplate) -> Dict[str, Any]:
    """Copy every field of the template that affects the prompt.

    A stored `prompt_template_id` is NOT enough: templates are editable through
    `PATCH /labeling-prompt-templates/{id}`, so a run holding only a foreign key
    would silently re-describe itself the moment someone tuned the template
    mid-experiment, and two runs would claim to differ by a change neither
    actually used.

    DERIVED FROM THE FINGERPRINT'S OWN FIELD SET, not hand-listed.

    This listed seventeen fields by hand, next to a fingerprint module whose
    entire design note is that hand-listing is the bug. `prompt_fingerprint`
    is computed over `__table__.columns` minus an identity denylist, so a
    hand-written freeze list can only ever drift BEHIND it — and the drift is
    silent, because a frozen run still looks complete.

    That drift is not cosmetic here. A frozen template is what lets a trial say
    what it actually ran; a field missing from the freeze is a variable that
    moved without being recorded.
    """
    frozen: Dict[str, Any] = {
        "template_id": template.id,
        "template_name": template.name,
    }
    frozen.update(prompt_fingerprint_fields(template))
    frozen.update({
        "body_sha256": hashlib.sha256(
            f"{template.system_message}\x1f{template.user_prompt_template}".encode()
        ).hexdigest()[:16],
    })
    return frozen


class LabelingTrialService:
    def __init__(self, db):
        self.db = db

    # ── start (async, from the endpoint) ─────────────────────────────────────

    async def start_trial(
        self,
        extraction_job_id: str,
        feature_ids: List[str],
        config: Dict[str, Any],
    ) -> LabelingTrialRun:
        if not isinstance(self.db, AsyncSession):
            raise TrialError("start_trial requires an AsyncSession")
        if not feature_ids:
            raise TrialError("a trial needs at least one feature")
        if len(feature_ids) > MAX_PANEL_SIZE:
            raise TrialError(
                f"panel of {len(feature_ids)} exceeds the maximum {MAX_PANEL_SIZE}"
            )

        extraction = (await self.db.execute(
            select(ExtractionJob).where(ExtractionJob.id == extraction_job_id)
        )).scalar_one_or_none()
        if not extraction:
            raise TrialError(f"Extraction job {extraction_job_id} not found")
        if extraction.status != ExtractionStatus.COMPLETED.value:
            raise TrialError(
                f"Extraction {extraction_job_id} is {extraction.status}, not completed"
            )

        # Bind child to parent. A foreign or unknown id is a hard error, not a
        # warning: silently dropping it would shrink the panel, change its
        # panel_id, and destroy comparability with every earlier run — the exact
        # property the panel exists to provide.
        rows = (await self.db.execute(
            select(Feature.id).where(
                Feature.id.in_(feature_ids),
                Feature.extraction_job_id == extraction_job_id,
            )
        )).scalars().all()
        missing = sorted(set(feature_ids) - set(rows))
        if missing:
            raise TrialError(
                f"{len(missing)} feature(s) are not in extraction "
                f"{extraction_job_id}: {missing[:5]}"
                + ("…" if len(missing) > 5 else "")
            )

        template_id = config.get("prompt_template_id")
        template = None
        if template_id:
            template = (await self.db.execute(
                select(LabelingPromptTemplate).where(
                    LabelingPromptTemplate.id == template_id)
            )).scalar_one_or_none()
            if not template:
                raise TrialError(f"Prompt template {template_id} not found")
        else:
            template = (await self.db.execute(
                select(LabelingPromptTemplate).where(
                    LabelingPromptTemplate.is_default.is_(True))
            )).scalars().first()
            if not template:
                raise TrialError(
                    "no prompt_template_id given and no default template exists"
                )
        if template.is_detection_template:
            raise TrialError(
                f"template {template.id} is a detection/scoring template, not a "
                f"labeling template; it cannot be the variable under test"
            )

        panel = panel_id_for(extraction_job_id, feature_ids)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        job_id = f"trial_{extraction_job_id}_{stamp}_{uuid.uuid4().hex[:6]}"
        run_id = f"ltr_{uuid.uuid4().hex[:12]}"

        # A trial neither takes nor is blocked by the apply-path 409: it writes
        # no Feature row, so the invariant that lock protects does not apply.
        # It IS blocked by another in-flight trial on the same panel, so a
        # double-click cannot burn the budget twice.
        inflight = (await self.db.execute(
            select(LabelingTrialRun).where(
                LabelingTrialRun.panel_id == panel,
                LabelingTrialRun.status.in_(["queued", "running"]),
            )
        )).scalars().first()
        if inflight:
            raise TrialError(
                f"panel {panel[:20]}… already has an in-flight trial: {inflight.id}"
            )

        job = LabelingJob(
            id=job_id,
            extraction_job_id=extraction_job_id,
            labeling_method=config.get("labeling_method", "openai_compatible"),
            openai_model=config.get("openai_model"),
            openai_compatible_endpoint=config.get("openai_compatible_endpoint"),
            openai_compatible_model=config.get("openai_compatible_model"),
            prompt_template_id=template.id,
            mode=LabelingMode.TRIAL.value,
            feature_ids=list(feature_ids),
            trial_run_id=run_id,
            status=LabelingStatus.QUEUED.value,
            progress=0.0,
            features_labeled=0,
            # Scoped, never the extraction-wide count. An unfiltered denominator
            # would make a 30-feature trial report progress against ~30k and jump
            # to 1.0 after the first batch.
            total_features=len(feature_ids),
            max_tokens=config.get("max_tokens", 300),
            api_timeout=config.get("api_timeout", 120.0),
        )
        run = LabelingTrialRun(
            id=run_id,
            panel_id=panel,
            extraction_job_id=extraction_job_id,
            labeling_job_id=job_id,
            prompt_template_id=template.id,
            name=config.get("name"),
            status="queued",
            payload={
                "panel": {
                    "panel_id": panel,
                    "extraction_job_id": extraction_job_id,
                    "feature_ids": sorted(feature_ids),
                    "size": len(feature_ids),
                },
                "prompt": freeze_template(template),
                "config": {
                    "labeling_method": config.get("labeling_method", "openai_compatible"),
                    "model": config.get("openai_compatible_model")
                             or config.get("openai_model"),
                    "batch_size": config.get("batch_size", LABEL_BATCH_SIZE),
                },
                "results": [],
                "stats": {},
            },
        )
        self.db.add(job)
        self.db.add(run)
        await self.db.commit()
        await self.db.refresh(run)
        return run

    # ── compare (pure) ───────────────────────────────────────────────────────

    @staticmethod
    def compare(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
        """Compare two trial payloads. Refuses more often than it answers.

        ONE SHAPE FOR EVERY RETURN PATH — `detection_delta` is present on all of
        them. `compare_panels` learned this the hard way (see its own comment:
        the dropout counters were added to the success branch only, so a
        consumer got a KeyError precisely when the comparison was refused). The
        refusal branches are exactly when a caller most needs to know whether
        the detection scores had anything to say — a failed judge gate, for
        instance, produces no scored features at all and would otherwise be
        indistinguishable from an empty panel.
        """
        pa = (a.get("panel") or {}).get("panel_id")
        pb = (b.get("panel") or {}).get("panel_id")
        if not pa or not pb or pa != pb:
            return {
                "comparable": False,
                "verdict": None,
                "detection_delta": None,
                "reason": "the two runs used different panels; their labels are "
                          "not measurements of the same thing",
            }

        ra = {r["feature_id"]: r for r in a.get("results", [])}
        rb = {r["feature_id"]: r for r in b.get("results", [])}
        overlap = sorted(set(ra) & set(rb))
        if not overlap:
            return {
                "comparable": True, "verdict": None, "compared": 0,
                "detection_delta": LabelingTrialService._detection_delta(a, b),
                "reason": "no overlapping features to compare; comparing nothing "
                          "is not comparing",
            }

        ok = [f for f in overlap
              if ra[f].get("status") == "ok" and rb[f].get("status") == "ok"]
        if not ok:
            # Failed labels stringify identically, so a fully-failed pair would
            # otherwise read as perfect agreement.
            return {
                "comparable": True, "verdict": "inconclusive", "compared": 0,
                "errors": {"a": len(overlap) - len(ok), "b": len(overlap) - len(ok)},
                "detection_delta": LabelingTrialService._detection_delta(a, b),
                "reason": "every overlapping feature errored in at least one arm",
            }

        per_feature = []
        changed = cat_changed = 0
        for f in ok:
            la, lb = ra[f], rb[f]
            name_changed = la.get("specific") != lb.get("specific")
            category_changed = la.get("category") != lb.get("category")
            changed += bool(name_changed)
            cat_changed += bool(category_changed)
            per_feature.append({
                "feature_id": f,
                "a": {"specific": la.get("specific"), "category": la.get("category")},
                "b": {"specific": lb.get("specific"), "category": lb.get("category")},
                "label_changed": name_changed,
                "category_changed": category_changed,
            })

        # COMPUTED ONCE. Each `_detection_delta` runs a bootstrap over the
        # panel; recomputing them inside the verdict would quadruple that work
        # and let the reported deltas drift from the verdict derived from them.
        _top_delta = LabelingTrialService._detection_delta(a, b)
        _mid_delta = LabelingTrialService._detection_delta(
            a, b, key="detection_midrange")

        return {
            "comparable": True,
            "panel_id": pa,
            "compared": len(ok),
            "overlap": len(overlap),
            "errored": len(overlap) - len(ok),
            "label_change_rate": changed / len(ok),
            "category_change_rate": cat_changed / len(ok),
            "per_feature": per_feature,
            "verdict": "b_differs" if changed else "identical",
            "detection_delta": _top_delta,
            # THE SECOND RULER, AND THE VERDICT THAT USES BOTH.
            "detection_delta_midrange": _mid_delta,
            "two_ruler_verdict": LabelingTrialService._two_ruler_verdict(
                _top_delta, _mid_delta
            ),
            "reason": None,
        }

    @staticmethod
    def panel_resolution(null_comparison: Dict[str, Any]) -> Optional[float]:
        """The smallest effect this panel can actually resolve.

        Read off a NULL comparison — the same template run twice. The true
        delta is zero by construction, so the spread of per-feature differences
        is pure measurement noise, and the MDE computed from it is the panel's
        real resolution rather than an assumed one.

        WHY RUN IT FIRST. A pre-registered threshold that sits below the panel's
        resolution is unreachable: the experiment cannot return a positive
        result at all, and a null cannot be distinguished from "underpowered".
        On ~80 features a plausible resolution is nearer 0.047 than the 0.02 a
        plan might name. Eight minutes of judge time buys the difference between
        an interpretable null and a meaningless one.

        It also measures judge stability, which is otherwise not computable:
        two identical runs that disagree are disagreeing about nothing.

        NOTE WHAT THIS IS NOT FOR. The resolution is reported alongside a
        verdict, never used AS the test — `detection_metrics.compare_panels`
        records why: MDE is 2.80 SE while the interval excludes zero at about
        1.96 SE, so testing an observed effect against it discarded every
        genuine result in that 43% band, including a measured +0.046 with an
        interval of [0.011, 0.082]. Use it to decide whether a threshold is
        worth pre-registering, not to accept or reject an arm.
        """
        delta = (null_comparison or {}).get("detection_delta") or {}
        return delta.get("minimum_detectable_effect")

    #: How much a top-K loss may be tolerated when mid-range improves.
    #:
    #: Zero would refuse any arm that trades top-decile specificity at all,
    #: which is what a broader label necessarily does. `MIN_MEANINGFUL_DELTA` is
    #: the floor below which `compare_panels` refuses to call a change real, so
    #: a loss under it is by that instrument's own standard not a loss.
    MAX_TOLERATED_TOPK_LOSS = 0.02

    @staticmethod
    def _two_ruler_verdict(
        top: Optional[Dict[str, Any]],
        mid: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Ship only on a mid-range win with no material top-K loss.

        WHY TWO RULERS AT ALL.

        Detection asks whether a label lets a judge pick the feature's passages
        out of a pool. Over TOP-K positives that is maximised by the narrowest
        label still covering the top decile — and a wider example spread exists
        to produce labels broader than the top decile. Scored on one ruler, the
        instrument and the hypothesis point in opposite directions, and the
        experiment can only return "worse" or "no difference".

        Worked case: a feature whose top decile is legal boilerplate and whose
        ranks 20-100 are formal institutional prose. `legal_contract_language`
        is narrower and false of the feature; `formal_institutional_register` is
        truer and fires on more of a general corpus, so it scores ~0.175 WORSE
        on top-K. One ruler says "do not ship" about the better label.

        Pinning to mid-range instead would just invert the bias. Both, with
        asymmetric thresholds, is the only formulation where "truer across the
        range" is something the instrument can reward while an arm that has
        stopped describing the strongest evidence is still caught.
        """
        top = top or {}
        mid = mid or {}

        if not top or not mid:
            return {
                "verdict": None,
                "reason": "one or both rulers produced no comparison",
            }
        for name, delta in (("top_k", top), ("midrange", mid)):
            if delta.get("verdict") is None:
                return {
                    "verdict": None,
                    "reason": (
                        f"the {name} ruler returned no verdict: "
                        f"{delta.get('reason') or 'unstated'}"
                    ),
                }

        mid_delta = mid.get("mean_delta")
        top_delta = top.get("mean_delta")
        if mid_delta is None or top_delta is None:
            return {"verdict": None, "reason": "a ruler reported no mean delta"}

        improved = mid.get("verdict") == "candidate_better"
        tolerated = top_delta >= -LabelingTrialService.MAX_TOLERATED_TOPK_LOSS

        if improved and tolerated:
            verdict = "ship"
        elif improved and not tolerated:
            verdict = "traded_top_k"
        elif mid.get("verdict") == "baseline_better":
            verdict = "worse"
        else:
            verdict = "indistinguishable"

        return {
            "verdict": verdict,
            "midrange_mean_delta": mid_delta,
            "top_k_mean_delta": top_delta,
            "max_tolerated_top_k_loss": LabelingTrialService.MAX_TOLERATED_TOPK_LOSS,
            "reason": {
                "ship": "improved across the range without materially losing "
                        "the top decile",
                "traded_top_k": "improved across the range but gave up more of "
                                "the top decile than the tolerance allows — a "
                                "broader label is not automatically a better "
                                "one",
                "worse": "the wider spread produced labels that describe the "
                         "feature less well on its own mid-range",
                "indistinguishable": "no effect either ruler can resolve; read "
                                     "minimum_detectable_effect before "
                                     "concluding there is none",
            }[verdict],
        }

    #: Coverage may differ between arms by at most this fraction of the panel
    #: before the comparison is refused.
    #:
    #: A template that refuses MORE features is graded on the subset it chose to
    #: answer, and an easier subset scores higher. Without this the headline
    #: delta silently rewards refusing hard features — which is precisely how a
    #: change could look like an improvement while making labels worse.
    MAX_COVERAGE_GAP = 0.15

    @staticmethod
    def _detection_delta(
        a: Dict[str, Any],
        b: Dict[str, Any],
        *,
        key: str = "detection",
    ) -> Optional[Dict[str, Any]]:
        """The PAIRED detection-score comparison — the only real verdict here.

        WHY THIS EXISTS
        ---------------
        `compare` above answers "did the label STRING change", which is not a
        quality measurement: a template that renames everything scores the same
        as one that fixes everything. The real instrument is
        `detection_metrics.compare_panels` — a paired bootstrap with a minimum
        meaningful delta and a minimum detectable effect.

        It had **no production caller**. Only a unit test imported it, so the
        entire statistical apparatus was dead code while `compare` shipped a
        string diff. That is this repo's signature failure mode: a mechanism
        fully implemented, tested, and unreachable.

        PAIRING IS THE POINT. Both arms saw identical features, passages and
        judge, so everything except the prompt cancels — which is what makes a
        panel of ~30 sufficient.

        Returns None when neither arm carries detection scores at all (scoring
        is optional), so a caller can tell "not measured" from "measured and
        inconclusive".
        """
        det_a = a.get(key) or {}
        det_b = b.get(key) or {}
        if not det_a and not det_b:
            return None

        _present_a = set(det_a.get("per_feature") or {})
        _present_b = set(det_b.get("per_feature") or {})
        _scored_a = {
            k for k, v in (det_a.get("per_feature") or {}).items()
            if (v or {}).get("balanced_accuracy") is not None
        }
        _scored_b = {
            k for k, v in (det_b.get("per_feature") or {}).items()
            if (v or {}).get("balanced_accuracy") is not None
        }

        def _refused(reason: str) -> Dict[str, Any]:
            """A refusal carrying the SAME KEYS a verdict carries.

            `compare_panels` learned this about its own dropout counters: keys
            added to the success branch alone produce a KeyError precisely when
            the comparison is refused, which is when a caller most needs them.
            The MCP tool advertises mean_delta / ci / minimum_detectable_effect
            as part of this payload, so a refusal that omits them breaks every
            reader on exactly the cases it exists to report.
            """
            return {
                "verdict": None,
                "compared": 0,
                "mean_delta": None,
                "ci": None,
                "minimum_detectable_effect": None,
                "wins": 0, "losses": 0, "ties": 0,
                # COMPARE_PANELS' DEFINITIONS, CHARACTER FOR CHARACTER.
                #
                # A refusal that reports "nothing was discarded" while an entire
                # arm went unscored is a false statement inside the payload that
                # exists to be honest. But the first correction of it used the
                # SCORED sets on both sides, which undercounts: a feature that
                # was present in `per_feature` and ungradeable in BOTH arms was
                # on the panel and did not make the comparison, and
                # `compare_panels` counts it. Since scored ⊆ present, the error
                # ran in the flattering direction — the same shape as the
                # filter_suppressed_neurons undercount in this very arc.
                #
                # `baseline_total` had the same split: the PRESENT count on a
                # verdict, the SCORED count on a refusal. One key, two
                # quantities, which is worse than either.
                "baseline_total": len(_present_a),
                "candidate_total": len(_present_b),
                "dropped": len(_present_a | _present_b) - len(_scored_a & _scored_b),
                "coverage_a": det_a.get("coverage") or None,
                "coverage_b": det_b.get("coverage") or None,
                "reason": reason,
            }

        # FINDING 5: "never scored" and "scored under a different ruler" are
        # different facts. `_score_detection`'s early returns carry no
        # `prompt_version` at all, so a comparison against an unscorable arm
        # would otherwise report a version mismatch — blaming a moved ruler for
        # an arm that was never measured.
        for name, det in (("a", det_a), ("b", det_b)):
            if det and not det.get("per_feature") and not det.get("prompt_version"):
                return _refused(
                    f"arm {name} produced no detection scores at all "
                    f"({det.get('reason') or 'no reason recorded'}); it was "
                    f"never measured, so there is nothing to compare"
                )

        # A MOVED RULER IS NOT A COMPARISON.
        #
        # The detection prompt is deliberately pinned (PADR IDL-48: "an editable
        # ruler silently invalidates every prior score"). Two arms scored under
        # different prompt versions are two different measurements, and
        # subtracting them produces a number with no meaning.
        va, vb = det_a.get("prompt_version"), det_b.get("prompt_version")
        if va != vb:
            return _refused(
                f"scored under different rulers ({va!r} vs {vb!r}); the "
                f"scores are not measurements of the same thing"
            )

        # A FAILED GATE MEANS THERE ARE NO SCORES TO COMPARE. `score_panel`
        # already refuses in that case, so per_feature is empty and the delta
        # would silently read as "no overlap" rather than "the judge could not
        # grade anything".
        for name, det in (("a", det_a), ("b", det_b)):
            gate = det.get("gate") or {}
            if gate and gate.get("passed") is False:
                return _refused(
                    f"arm {name}'s judge failed the sanity gate, so its "
                    f"scores were withheld; the template was never measured"
                )

        def _ba(det: Dict[str, Any]) -> Dict[str, Optional[float]]:
            return {
                fid: (row or {}).get("balanced_accuracy")
                for fid, row in (det.get("per_feature") or {}).items()
            }

        from .detection_metrics import compare_panels

        delta = dict(compare_panels(_ba(det_a), _ba(det_b)))

        cov_a = det_a.get("coverage") or {}
        cov_b = det_b.get("coverage") or {}
        delta["coverage_a"] = cov_a or None
        delta["coverage_b"] = cov_b or None

        panel_size = cov_a.get("panel_size") or cov_b.get("panel_size") or 0
        scored_a = cov_a.get("scored")
        scored_b = cov_b.get("scored")
        if panel_size and scored_a is not None and scored_b is not None:
            gap = abs(scored_a - scored_b) / panel_size
            delta["coverage_gap"] = gap
            if gap > LabelingTrialService.MAX_COVERAGE_GAP:
                delta["confounded_by_coverage"] = True
                delta["verdict"] = None
                delta["reason"] = (
                    f"the arms scored different numbers of features "
                    f"({scored_a} vs {scored_b} of {panel_size}); the arm that "
                    f"answered fewer was graded on the subset it chose, so the "
                    f"delta measures selection, not quality"
                )
            else:
                delta["confounded_by_coverage"] = False

        return delta

    # ── run (sync, from Celery) ──────────────────────────────────────────────

    # Features whose label is a refusal carry no claim to test, so they are not
    # scored. That makes COVERAGE part of the result rather than a footnote: a
    # template that refuses 30 of 31 features would otherwise be scored on the
    # one it kept and report a near-perfect number. Measured on this panel — the
    # substitution-test candidate labelled 1 of 31 and would have reported a
    # single-feature score against the baseline's eighteen.
    _REFUSAL_LABELS = {"uninterpretable", "noise", "none", "unknown", ""}

    def _score_detection(self, *, run, features, results, examples_by_feature,
                         prompt_sample_indices=None,
                         labeler) -> Dict[str, Any]:
        """Run the judge sanity gate, then score every labelled feature.

        Never raises into the trial: a scoring failure must not discard labels
        that took real GPU time to produce. It records why instead, because
        "no score" and "a bad score" are different facts and collapsing them
        is how a broken judge gets read as a bad template.
        """
        from src.services import labeling_detection_scorer as scorer

        try:
            by_id = {f.id: f for f in features}
            panel_id = run.panel_id
            scorable, skipped = [], []

            for r in results:
                label = (r.get("specific") or "").strip().lower()
                if r["status"] != "ok" or label in self._REFUSAL_LABELS:
                    skipped.append(r["feature_id"])
                    continue
                positives = examples_by_feature.get(r["feature_id"]) or []
                if not positives:
                    skipped.append(r["feature_id"])
                    continue
                hard, easy = scorer.sample_negatives(
                    self.db,
                    feature_id=r["feature_id"],
                    extraction_id=by_id[r["feature_id"]].extraction_job_id,
                    # EXCLUDE BOTH SETS, not just the graded one.
                    #
                    # `positives` here is the PINNED scoring set (top-K). Under
                    # a stratified arm the judge was shown ranks 1, 11, 21 … 91
                    # — so ranks 11 upward are passages the label was literally
                    # derived from, and they stay drawable as negatives via any
                    # donor sharing that sample_index. The judge would then be
                    # asked whether the label describes its own evidence, answer
                    # yes, and be scored wrong for it.
                    #
                    # That biases the stratified arm DOWNWARD by construction,
                    # which is the same class of artifact the pinned ruler
                    # above exists to prevent — the ruler fixed the passages and
                    # missed the exclusion. The baseline arm has no such
                    # passages (its prompt set is a subset of its scoring set),
                    # so the bias is one-sided.
                    #
                    # Excluding more can only remove contamination; it cannot
                    # manufacture an advantage for either arm.
                    exclude_samples=sorted({
                        *(p.get("sample_index") for p in positives),
                        *((prompt_sample_indices or {}).get(r["feature_id"], ())),
                    } - {None}),
                )
                if not (hard or easy):
                    skipped.append(r["feature_id"])
                    continue

                # RENDER before assembling. score_feature reads item["text"],
                # and the rows here — from _retrieve_top_examples_batch_sync and
                # from the negative-sampling SQL alike — carry
                # prefix_tokens/prime_token/suffix_tokens and no "text" at all.
                # assemble_items copies the row through unchanged, so without
                # this every scoring run died on KeyError('text') and the
                # blanket except below reported it as {"scored": false} — a
                # broken measurement that looked like an absent one.
                #
                # Positives and negatives MUST go through the SAME renderer:
                # render_passage adds nothing and truncates symmetrically, so
                # neither class can be identified by formatting or by length.
                positives = [
                    {**row, "text": scorer.render_passage(row)} for row in positives
                ]
                hard = [{**row, "text": scorer.render_passage(row)} for row in hard]
                easy = [{**row, "text": scorer.render_passage(row)} for row in easy]

                scorable.append({
                    "feature_id": r["feature_id"],
                    # The label AND its description: the label alone is what a
                    # human sees, but a two-word snake_case string is a thinner
                    # claim than the template actually made.
                    "explanation": f'{r["specific"]}: {r.get("description") or ""}'.strip(),
                    "items": scorer.assemble_items(
                        positives, hard, easy,
                        panel_id=panel_id, feature_id=r["feature_id"],
                    ),
                    "negative_ceiling": scorer.negative_ceiling(positives),
                })

            if not scorable:
                return {
                    "scored": False,
                    "reason": "no feature carried a testable label",
                    "coverage": {"scored": 0, "skipped": len(skipped),
                                 "panel_size": len(features)},
                }

            judge = self._build_judge(labeler)

            # The gate runs FIRST and on a handful of calls. A judge that cannot
            # find a token the passages literally contain cannot grade anything
            # subtler, and scoring anyway would blame the template for the
            # judge's incapacity.
            controls = [{
                "feature_id": c["feature_id"],
                "items": c["items"],
                "literal_explanation": (
                    "Passages that contain the token "
                    f'"{self._rank1_token(examples_by_feature, c["feature_id"])}".'
                ),
                "mismatched_explanation": (
                    "Passages about eighteenth-century Baltic maritime insurance law."
                ),
            } for c in scorable[:2]]

            gate = scorer.run_gate(controls, judge)
            out = scorer.score_panel(scorable, judge, gate=gate)
            out["coverage"] = {
                "scored": len(scorable),
                "skipped": len(skipped),
                "panel_size": len(features),
            }
            return out

        except (KeyError, AttributeError, TypeError) as exc:
            # A SHAPE error, not a judge failure. Called out separately because
            # the two are not the same fact and collapsing them is how a
            # KeyError('text') spent its life being read as "the judge could not
            # score this panel".
            logger.error(
                "detection scoring hit a data-shape error for %s: %s — this is a "
                "BUG in the scoring wiring, not a judge problem",
                run.id, exc, exc_info=True,
            )
            return {
                "scored": False,
                "reason": f"scoring wiring error: {type(exc).__name__}: {exc}"[:300],
                "wiring_error": True,
            }
        except Exception as exc:
            logger.warning("detection scoring failed for %s: %s",
                           run.id, exc, exc_info=True)
            return {"scored": False, "reason": f"{type(exc).__name__}: {exc}"[:300]}

    @staticmethod
    def _rank1_token(examples_by_feature, feature_id: str) -> str:
        rows = examples_by_feature.get(feature_id) or []
        for r in rows:
            t = (r.get("prime_token") or "").strip().lstrip("\u2581").strip()
            if t:
                return t
        return "the"

    @staticmethod
    def _build_judge(labeler):
        """Adapt the labeling client into the scorer's JudgeFn (str -> str).

        Deliberately the SAME endpoint and model that produced the labels. A
        judge on a different model would fold that model's competence into every
        comparison, and two templates measured against different judges are not
        comparable at all.
        """
        import asyncio as _asyncio

        def _judge(prompt: str) -> str:
            loop = _asyncio.new_event_loop()
            try:
                _asyncio.set_event_loop(loop)
                resp = loop.run_until_complete(labeler._call_openai(
                    messages=[{"role": "user", "content": prompt}]
                ))
                return (resp.choices[0].message.content or "") if resp.choices else ""
            finally:
                loop.close()
                _asyncio.set_event_loop(None)

        return _judge

    def _assert_wrote_nothing(self) -> None:
        """The load-bearing guard.

        `_collect` never assigns a mapped attribute, so the session should hold
        no dirty Feature and any later commit has nothing to flush. But "we did
        not assign" is a discipline, not a guarantee — one careless line in a
        future edit turns a measurement into a destructive write over the very
        labels the trial exists to compare against. This turns that discipline
        into something a mutation can kill.
        """
        leaked = [o for o in self.db.dirty if isinstance(o, Feature)]
        if leaked:
            self.db.rollback()
            raise TrialWroteToFeatures(
                f"a trial modified {len(leaked)} Feature row(s): "
                f"{[getattr(f, 'id', '?') for f in leaked][:5]}"
            )

    def run_trial(self, labeling_job_id: str) -> Dict[str, Any]:
        if not isinstance(self.db, Session):
            raise TrialError("run_trial requires a sync Session")

        from .labeling_service import LabelingService
        from .openai_labeling_service import OpenAILabelingService

        job = self.db.query(LabelingJob).filter(
            LabelingJob.id == labeling_job_id).first()
        if not job:
            raise TrialError(f"labeling job {labeling_job_id} not found")
        if job.mode != LabelingMode.TRIAL.value:
            raise TrialError(
                f"job {labeling_job_id} has mode {job.mode!r}; run_trial refuses "
                f"anything but a trial, so it can never write labels by accident"
            )
        run = self.db.query(LabelingTrialRun).filter(
            LabelingTrialRun.id == job.trial_run_id).first()
        if not run:
            raise TrialError(f"trial run {job.trial_run_id} not found")

        payload = dict(run.payload or {})
        frozen = payload.get("prompt") or {}
        panel_ids = (payload.get("panel") or {}).get("feature_ids") or job.feature_ids

        # Scoped selection. Both predicates are kept: the extraction bound must
        # survive alongside the id filter, or a stale id from another extraction
        # could widen the panel.
        features = (
            self.db.query(Feature)
            .filter(Feature.extraction_job_id == job.extraction_job_id)
            .filter(Feature.id.in_(panel_ids))
            .order_by(Feature.neuron_index)
            .all()
        )
        if len(features) != len(panel_ids):
            raise TrialError(
                f"panel resolved to {len(features)} of {len(panel_ids)} features; "
                f"a shrunken panel is not the panel that was requested"
            )

        run.status = "running"
        job.status = LabelingStatus.LABELING.value
        self.db.commit()

        svc = LabelingService(self.db)
        max_examples = frozen.get("max_examples") or 10
        feature_ids = [f.id for f in features]

        # TWO RETRIEVALS, AND KEEPING THEM APART IS THE POINT.
        #
        # THE VARIABLE — what the judge is SHOWN, under the arm's own sampling
        # strategy. This is the thing under test.
        examples_by_feature = svc._retrieve_top_examples_batch_sync(
            self.db, feature_ids, max_examples=max_examples,
            sampling=frozen.get("example_sampling"),
        )

        # THE RULER — what the resulting label is TESTED on. Pinned to top_k at
        # a fixed K, independent of the arm.
        #
        # `_score_detection` takes its positives, its `negative_ceiling` and the
        # gate's oracle token from whatever dict it is handed. Passing the
        # prompt's examples would mean a stratified arm is graded on DIFFERENT,
        # intrinsically harder passages than the baseline — biasing it downward
        # by construction, so a null result would be uninterpretable and a
        # negative result an artifact of the instrument.
        #
        # This is PADR IDL-48 one level deeper. The scoring PROMPT is pinned
        # because an editable ruler silently invalidates every prior score; the
        # scoring PASSAGES are the other half of the same ruler.
        # The contrast block the arm's template asks for, excluding what the
        # prompt already shows — the same construction the bulk path uses, so a
        # trial predicts it.
        from .labeling_service import resolve_num_negative

        # From `frozen`, not from `template_config` — that dict is built forty
        # lines below, and the retrieval has to happen alongside the others.
        n_negative = resolve_num_negative(frozen)
        negatives_by_feature = {}
        if n_negative:
            negatives_by_feature = svc._retrieve_bottom_examples_batch_sync(
                self.db, feature_ids, num_negative_examples=n_negative,
                exclude_sample_indices_by_feature={
                    fid: [e.get("sample_index") for e in rows]
                    for fid, rows in examples_by_feature.items()
                },
            )

        scoring_examples = svc._retrieve_top_examples_batch_sync(
            self.db, feature_ids, max_examples=SCORING_POSITIVES_K,
            sampling="top_k",
        )

        # THE SECOND RULER, AND THE REASON THE FIRST IS NOT ENOUGH.
        #
        # The detection score asks: does this label let a judge pick the
        # feature's passages out of a pool? On a TOP-K positive set that is
        # maximised by the NARROWEST label still covering the top decile — and
        # spreading the examples exists precisely to produce labels broader than
        # the top decile. The instrument and the hypothesis point in opposite
        # directions.
        #
        # Worked case: a feature whose top decile is legal boilerplate and whose
        # ranks 20-100 are formal institutional prose generally.
        # `legal_contract_language` is narrower and FALSE of the feature;
        # `formal_institutional_register` is truer and fires on more of a general
        # corpus, so it scores ~0.175 WORSE on the top-K ruler. A single-ruler
        # experiment returns "do not ship", with confidence, on the arm that
        # produced the better label.
        #
        # So: two pinned rulers, both identical across arms. Top-K catches an
        # arm that has stopped describing the strongest evidence; mid-range
        # catches one that describes the range better. Ship only on a win at
        # mid-range with no material loss at top-K.
        #
        # Drawn from bands 3-9 so it is disjoint from top-K, and pinned exactly
        # as the first ruler is — an arm must never be graded on its own
        # sampling.
        midrange_examples = svc._retrieve_midrange_examples_batch_sync(
            self.db, feature_ids, max_examples=SCORING_POSITIVES_K,
        )

        cfg = payload.get("config") or {}
        labeler = OpenAILabelingService(
            api_key=cfg.get("api_key") or "unused",
            base_url=job.openai_compatible_endpoint,
            model=cfg.get("model") or job.openai_model or "gpt-4o-mini",
            temperature=frozen.get("temperature", 0.3),
            max_tokens=job.max_tokens or frozen.get("max_tokens", 300),
            top_p=frozen.get("top_p", 0.9),
        )
        template_config = {
            k: frozen.get(k) for k in _TEMPLATE_CONFIG_KEYS
        }

        results: List[Dict[str, Any]] = []
        batch_size = int(cfg.get("batch_size") or LABEL_BATCH_SIZE)
        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            for start in range(0, len(features), batch_size):
                batch = features[start:start + batch_size]
                tasks = [
                    labeler.generate_label_from_examples(
                        examples=examples_by_feature.get(f.id, []),
                        template_config=template_config,
                        user_prompt_template=frozen.get("user_prompt_template", ""),
                        system_message=frozen.get("system_message", ""),
                        feature_id=f.id,
                        neuron_index=f.neuron_index,
                        # A TRIAL MUST SEND THE PROMPT THE BULK RUN WILL SEND.
                        #
                        # `include_negative_examples` is in the frozen
                        # `template_config` and the formatter renders the block
                        # only when it is handed a LIST — which this never
                        # passed. So a template configured for contrast produced
                        # a prompt byte-identical to one without it, `compare`
                        # reported `identical`, and the paired delta reported ~0
                        # with a CI containing zero, while the two frozen
                        # fingerprints DID differ. A confidently measured null,
                        # manufactured: the arm was never a distinct arm.
                        negative_examples=negatives_by_feature.get(f.id, []),
                    )
                    for f in batch
                ]
                labels = loop.run_until_complete(
                    asyncio.gather(*tasks, return_exceptions=True))

                # COLLECT. No attribute on `f` is assigned anywhere below.
                for f, label in zip(batch, labels):
                    if isinstance(label, Exception):
                        results.append({
                            "feature_id": f.id, "neuron_index": f.neuron_index,
                            "status": "error", "error": str(label)[:300],
                            "category": None, "specific": None, "description": None,
                            "fit_count": None, "confidence": None,
                        })
                        continue
                    results.append({
                        "feature_id": f.id,
                        "neuron_index": f.neuron_index,
                        "status": "ok",
                        "category": label.get("category"),
                        "specific": label.get("specific"),
                        "description": label.get("description", ""),
                        # The model's SELF-ASSESSMENT. Templates ask for these and
                        # they were parsed and discarded, which left no way to tell
                        # a confident label from a hedged one — the exact signal a
                        # trial exists to compare. None when the template does not
                        # ask for them.
                        "fit_count": label.get("fit_count"),
                        "confidence": label.get("confidence"),
                        # Recorded so a reader can see the label was protected in
                        # the apply path, without the trial skipping it — skipping
                        # would punch a hole in the panel and break comparability
                        # with a run taken before the star existed.
                        "protected": f.star_color == "aqua",
                    })

                self._assert_wrote_nothing()
                job.progress = min(1.0, (start + len(batch)) / len(features))
                job.features_labeled = len(results)
                self.db.commit()
        finally:
            loop.close()
            asyncio.set_event_loop(None)

        ok = [r for r in results if r["status"] == "ok"]
        payload["results"] = results
        payload["stats"] = {
            "panel_size": len(features),
            "labeled": len(ok),
            "errors": len(results) - len(ok),
            "protected": sum(1 for r in ok if r.get("protected")),
        }

        # The measurement this whole apparatus exists for. Without it a trial
        # yields labels and someone reads them, which is the method the trial
        # was built to replace.
        payload["scoring"] = {
            "positive_sampling": "top_k",
            "n_positive": SCORING_POSITIVES_K,
            "midrange_positive_sampling": "bands_3_9",
            "midrange_n_positive": SCORING_POSITIVES_K,
        }
        payload["detection"] = self._score_detection(
            run=run, features=features, results=results,
            examples_by_feature=scoring_examples,
            # BOTH HALVES OF WHAT THE JUDGE WAS SHOWN.
            #
            # The contrast passages were omitted, so a donor sharing one of
            # their sample_indexes put it back in as a negative — and the judge
            # had been told explicitly that those passages "also activate the
            # feature". At scoring time it is asked whether the label describes
            # one, says yes, and is marked wrong.
            #
            # One-sided: only a contrast arm has such passages, so it biases the
            # answer to "contrast does not help" — a manufactured negative in
            # place of the manufactured null this same round removed.
            prompt_sample_indices={
                fid: (
                    [e.get("sample_index") for e in rows]
                    + [
                        e.get("sample_index")
                        for e in negatives_by_feature.get(fid, [])
                    ]
                )
                for fid, rows in examples_by_feature.items()
            },
            labeler=labeler,
        )

        # THE SAME LABELS, GRADED AGAINST THE OTHER RULER.
        #
        # Only the positive passages differ; the exclusion is the union of both
        # rulers and both halves of the prompt, so neither arm can be scored
        # against text it was shown.
        payload["detection_midrange"] = self._score_detection(
            run=run, features=features, results=results,
            examples_by_feature=midrange_examples,
            prompt_sample_indices={
                fid: (
                    [e.get("sample_index") for e in rows]
                    + [
                        e.get("sample_index")
                        for e in negatives_by_feature.get(fid, [])
                    ]
                    + [
                        e.get("sample_index")
                        for e in scoring_examples.get(fid, [])
                    ]
                )
                for fid, rows in examples_by_feature.items()
            },
            labeler=labeler,
        )
        run.payload = payload
        run.status = "completed"
        run.completed_at = datetime.now(timezone.utc).replace(tzinfo=None)
        job.status = LabelingStatus.COMPLETED.value
        job.progress = 1.0
        job.completed_at = datetime.now(timezone.utc)
        self._assert_wrote_nothing()
        self.db.commit()
        return {"trial_run_id": run.id, "stats": payload["stats"]}
