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

logger = logging.getLogger(__name__)

MAX_PANEL_SIZE = 200
LABEL_BATCH_SIZE = 10


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
    """
    return {
        "template_id": template.id,
        "template_name": template.name,
        "system_message": template.system_message,
        "user_prompt_template": template.user_prompt_template,
        "template_type": template.template_type,
        "temperature": template.temperature,
        "max_tokens": template.max_tokens,
        "top_p": template.top_p,
        "max_examples": template.max_examples,
        "include_prefix": template.include_prefix,
        "include_suffix": template.include_suffix,
        "prime_token_marker": template.prime_token_marker,
        "include_logit_effects": template.include_logit_effects,
        "top_promoted_tokens_count": template.top_promoted_tokens_count,
        "top_suppressed_tokens_count": template.top_suppressed_tokens_count,
        "include_negative_examples": template.include_negative_examples,
        "num_negative_examples": template.num_negative_examples,
        "include_nlp_analysis": template.include_nlp_analysis,
        "is_detection_template": template.is_detection_template,
        "body_sha256": hashlib.sha256(
            f"{template.system_message}\x1f{template.user_prompt_template}".encode()
        ).hexdigest()[:16],
    }


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
        """Compare two trial payloads. Refuses more often than it answers."""
        pa = (a.get("panel") or {}).get("panel_id")
        pb = (b.get("panel") or {}).get("panel_id")
        if not pa or not pb or pa != pb:
            return {
                "comparable": False,
                "verdict": None,
                "reason": "the two runs used different panels; their labels are "
                          "not measurements of the same thing",
            }

        ra = {r["feature_id"]: r for r in a.get("results", [])}
        rb = {r["feature_id"]: r for r in b.get("results", [])}
        overlap = sorted(set(ra) & set(rb))
        if not overlap:
            return {
                "comparable": True, "verdict": None, "compared": 0,
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
            "reason": None,
        }

    # ── run (sync, from Celery) ──────────────────────────────────────────────

    # Features whose label is a refusal carry no claim to test, so they are not
    # scored. That makes COVERAGE part of the result rather than a footnote: a
    # template that refuses 30 of 31 features would otherwise be scored on the
    # one it kept and report a near-perfect number. Measured on this panel — the
    # substitution-test candidate labelled 1 of 31 and would have reported a
    # single-feature score against the baseline's eighteen.
    _REFUSAL_LABELS = {"uninterpretable", "noise", "none", "unknown", ""}

    def _score_detection(self, *, run, features, results, examples_by_feature,
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
                    exclude_samples=[p.get("sample_index") for p in positives],
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
        examples_by_feature = svc._retrieve_top_examples_batch_sync(
            self.db, [f.id for f in features], max_examples=max_examples
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
            k: frozen.get(k) for k in (
                "template_type", "max_examples", "include_prefix", "include_suffix",
                "prime_token_marker", "include_logit_effects",
                "top_promoted_tokens_count", "top_suppressed_tokens_count",
                "include_negative_examples", "num_negative_examples",
            )
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
        payload["detection"] = self._score_detection(
            run=run, features=features, results=results,
            examples_by_feature=examples_by_feature, labeler=labeler,
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
