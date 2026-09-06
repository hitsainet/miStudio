"""Circuit strength calibration service (Feature 20 / IDL-37).

Orchestrates: generate neutral falsifiable probes → search the usable band
(onset by drift, cliff by judge) → clamp the served dial to it → persist a
reproducible manifest and the band onto the circuit.

Design split, so the orchestration is testable without a GPU or a live LLM:
  * the SEARCH is a pure function of injected callbacks (circuit_calibration_search);
  * the JUDGE and PROBE generator wrap the enhanced-labeling LLM client;
  * the GPU GENERATION (multi-layer circuit steering at a given dial) is isolated
    in `_build_generation_fns`, mirroring CircuitFaithfulnessService's model
    loading. `run(...)` accepts injected `gen_at`/`baseline_at`/`judge` so the
    whole orchestration — clamp, manifest, write-back, reproduce — is covered by
    unit tests; production passes the GPU/LLM-backed callables.

Like faithfulness, calibration runs ON the circuit and holds the GPU, so its
in-flight lifecycle (calibration_status/task_id) lives on the circuit row.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional

from .circuit_calibration_search import calibrate
from .circuit_probe_generator import generate_probes

logger = logging.getLogger(__name__)

DEFAULT_STEP_BUDGET = 10
DEFAULT_PROBE_COUNT = 3
DEFAULT_MARGIN = 0.15
JUDGE_METRIC_ID = "neutral-fact-correctness/v1"


class CalibrationConfigError(ValueError):
    """Bad calibration request — surfaces as a 422."""


class CalibrationRunError(RuntimeError):
    """Calibration failed at run time."""


class CircuitCalibrationService:
    """Find and ship a circuit's usable steering band."""

    @staticmethod
    def create_config(config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        cfg = dict(config or {})
        step_budget = int(cfg.get("step_budget", DEFAULT_STEP_BUDGET))
        probe_count = int(cfg.get("probe_count", DEFAULT_PROBE_COUNT))
        seed = int(cfg.get("seed", 0))
        margin = float(cfg.get("margin", DEFAULT_MARGIN))
        if step_budget < 2:
            raise CalibrationConfigError("step_budget must be ≥ 2")
        if probe_count < 1:
            raise CalibrationConfigError("probe_count must be ≥ 1")
        if not (0.0 <= margin <= 1.0):
            raise CalibrationConfigError("margin must be in [0, 1]")
        # The judge + probe-generation LLM endpoint/model, carried on the request
        # like an enhanced-labeling job (labeling config is per-request, not a
        # global setting). Optional: without them the search still runs with the
        # neutral static probes but cannot judge a real cliff — the endpoint
        # gates whether run() can do a real GPU pass.
        return {"step_budget": step_budget, "probe_count": probe_count,
                "seed": seed, "margin": margin,
                "judge_endpoint": cfg.get("judge_endpoint"),
                "judge_model": cfg.get("judge_model"),
                "judge_api_key": cfg.get("judge_api_key")}

    @staticmethod
    def _member_labels(circuit) -> List[str]:
        labels: List[str] = []
        for m in (circuit.members or []):
            feat = m.get("feature") or {}
            lb = feat.get("label")
            if lb:
                labels.append(str(lb))
        return labels

    #: CircuitBudget.intensity is bounded le=2.0, so a served dial (and thus a
    #: calibrated cliff/sweet_spot) can never exceed this. The search must not
    #: look above it, or the clamp would produce a budget the contract rejects.
    MAX_DIAL = 2.0

    @classmethod
    def _dial_bounds(cls, circuit) -> tuple[float, float]:
        """The dial search range = the circuit's AUTHORED intensity_range,
        CAPPED at MAX_DIAL (the servable ceiling). Defaults to [0, 2]."""
        budget = circuit.budget or {}
        rng = budget.get("intensity_range") or [0.0, cls.MAX_DIAL]
        try:
            lo, hi = float(rng[0]), float(rng[1])
        except (TypeError, ValueError, IndexError):
            lo, hi = 0.0, cls.MAX_DIAL
        if hi <= lo:
            lo, hi = 0.0, cls.MAX_DIAL
        # An authored lo ABOVE the servable ceiling can't be calibrated — capping
        # it would collapse the range to a point and every run would silently
        # return no-usable-band with no explanation. Fail loudly instead (R3).
        if lo >= cls.MAX_DIAL:
            raise CalibrationConfigError(
                f"intensity_range lower bound {lo} is at/above the servable "
                f"ceiling {cls.MAX_DIAL}; nothing to calibrate. Lower the "
                "authored intensity_range.")
        # Never search (or clamp) above what the budget can hold — an authored
        # range like [0, 3.0] would otherwise yield a cliff the contract rejects.
        hi = min(hi, cls.MAX_DIAL)
        lo = max(0.0, min(lo, hi))
        return lo, hi

    @classmethod
    def build_band(
        cls,
        circuit,
        *,
        gen_at: Callable[[float, str], str],
        baseline_at: Callable[[str], str],
        judge: Callable[[str, str], str],
        divergence: Callable[[str, str], float],
        config: Optional[Dict[str, Any]] = None,
        llm_call: Optional[Callable[[str, int], str]] = None,
    ) -> Dict[str, Any]:
        """The pure orchestration: probes → search → band + manifest payload.

        Returns {"band": <CircuitCalibration dict>, "manifest_payload": {...}}.
        All generation/judging is injected — no GPU, no network — so this is the
        unit-tested seam. `run()` wires the real GPU/LLM callables into here.
        """
        cfg = cls.create_config(config)
        labels = cls._member_labels(circuit)
        probes = generate_probes(labels, llm_call=llm_call,
                                 n=cfg["probe_count"])
        lo, hi = cls._dial_bounds(circuit)

        result = calibrate(
            gen_at, baseline_at, judge, divergence,
            probes=probes, lo=lo, hi=hi,
            max_steps=cfg["step_budget"], margin=cfg["margin"],
        )

        band = {
            "onset": result.onset,
            "sweet_spot": result.sweet_spot,
            "cliff": result.cliff,
            "provisional": True,   # generated probes + discovery-plane measurement
            "probe_set": probes,
            "judge_metric_id": JUDGE_METRIC_ID,
            "step_budget": cfg["step_budget"],
            "non_monotone": result.non_monotone,
        }
        # First-class transcripts: every (dial, prompt) generation the search
        # produced, so the manifest is analysis-ready for an Opus meaning pass
        # (not just verdict-only). Lifted from the trace entries that carry a
        # `generation`.
        transcripts = [
            {"dial": t.get("dial"), "prompt": t.get("probe"),
             "phase": t.get("phase"), "verdict": t.get("verdict"),
             "generation": t.get("generation")}
            for t in result.trace if t.get("generation") is not None
        ]
        manifest_payload = {
            "probes": probes,
            "config": cfg,
            "band": {k: band[k] for k in ("onset", "sweet_spot", "cliff",
                                          "non_monotone")},
            "transcripts": transcripts,
            "usable_band": result.usable_band,
            "judge_reliable": result.judge_reliable,
            "trace": result.trace,
            "converged": result.converged,
            "steps_used": result.steps_used,
            "floor": result.floor,
            "dial_bounds": [lo, hi],
        }
        # `usable_band` is an ORCHESTRATION signal, not a contract field: when
        # False the search found no correct dial above onset, so the caller must
        # NOT clamp the served dial to this degenerate band (badge, not gate —
        # a failed measurement must not disable the circuit).
        return {"band": band, "usable_band": result.usable_band,
                "judge_reliable": result.judge_reliable,
                "manifest_payload": manifest_payload}

    @classmethod
    def run(cls, db, circuit_id: str, config: Dict[str, Any], *,
            progress_cb: Optional[Callable[[int], None]] = None) -> Dict[str, Any]:
        """GPU orchestrator (sync, called from the Celery task).

        Loads the circuit, builds the GPU-backed generation + LLM judge, searches
        the band via build_band, persists a `calibration` manifest, and clamps
        the served dial + writes the band onto the circuit through the contract.

        The GPU generation wiring (`_build_generation_fns`) is the calibration
        close-out step (tracked debt, like 017's Phase-6 GPU acceptance); the
        orchestration below is otherwise complete and unit-tested via build_band.
        """
        from ..models.circuit import Circuit

        circuit = db.query(Circuit).filter(Circuit.id == circuit_id).first()
        if circuit is None:
            raise CalibrationRunError(f"Circuit {circuit_id} not found")
        if not circuit.members:
            raise CalibrationRunError("Circuit has no members")
        cfg = cls.create_config(config)

        if progress_cb:
            progress_cb(5)
        gen_at, baseline_at, judge, divergence, llm_call = \
            cls._build_generation_fns(circuit, db, cfg)

        out = cls.build_band(
            circuit, gen_at=gen_at, baseline_at=baseline_at, judge=judge,
            divergence=divergence, config=cfg, llm_call=llm_call)
        if progress_cb:
            progress_cb(80)

        # Persist the manifest FIRST so the measurement is never lost — but the
        # order below (manifest → write) is safe because _write_calibration
        # validates the band and can only fail on an inverted band, which
        # build_band cannot produce. The no-usable-band case is NOT a failure:
        # we record the manifest and mark the run completed WITHOUT clamping.
        manifest_id = cls._persist_manifest(db, circuit_id, out["manifest_payload"])

        if not out.get("judge_reliable", True):
            # The judge called the UNSTEERED output broken — it can't grade this
            # circuit's model. Inconclusive, NOT a claim about the circuit.
            cls._mark_status(db, circuit_id, "judge_unreliable")
            if progress_cb:
                progress_cb(100)
            return {"circuit_id": circuit_id, "band": None,
                    "usable_band": False, "judge_reliable": False,
                    "manifest_ref": manifest_id,
                    "reason": "the judge graded the UNSTEERED baseline as not "
                    "correct — it cannot reliably grade this circuit's model; "
                    "use a stronger judge model"}

        if not out["usable_band"]:
            # No correct dial above onset — do NOT clamp (badge, not gate: a
            # failed measurement must not disable the circuit). Record the
            # outcome so an agent can see WHY there is no band.
            cls._mark_no_usable_band(db, circuit_id, manifest_id)
            if progress_cb:
                progress_cb(100)
            return {"circuit_id": circuit_id, "band": None,
                    "usable_band": False, "manifest_ref": manifest_id,
                    "reason": "no dial above onset was judged correct"}

        band = dict(out["band"])
        band["manifest_ref"] = manifest_id
        cls._write_calibration(db, circuit_id, band)
        if progress_cb:
            progress_cb(100)
        return {"circuit_id": circuit_id, "band": band,
                "usable_band": True, "manifest_ref": manifest_id}

    @classmethod
    def reproduce(cls, db, manifest_id: str, *,
                  progress_cb: Optional[Callable[[int], None]] = None
                  ) -> Dict[str, Any]:
        """Re-run calibration from a stored manifest's config + probes and store
        a `reproduction` manifest with the band-delta verdict (FPRD §8.5). Uses
        the SAME probes and seed as the original, so a reproducible measurement
        lands within tolerance. Does NOT clamp the circuit — reproduction only
        checks the number, it doesn't re-ship the band.
        """
        from ..models.circuit import Circuit
        from ..models.validation_manifest import ValidationManifest
        from .manifest_service import ManifestService

        original = db.query(ValidationManifest).filter(
            ValidationManifest.id == manifest_id).first()
        if original is None:
            raise CalibrationRunError(f"Manifest {manifest_id} not found")
        if original.kind != "calibration":
            raise CalibrationRunError(
                f"Manifest {manifest_id} is {original.kind!r}, not calibration")
        payload = original.payload or {}
        circuit = db.query(Circuit).filter(
            Circuit.id == original.circuit_id).first()
        if circuit is None:
            raise CalibrationRunError(
                "Original circuit is gone — cannot reproduce")

        cfg = cls.create_config(payload.get("config") or {})
        stored_probes = payload.get("probes") or []
        if progress_cb:
            progress_cb(5)
        gen_at, baseline_at, judge, divergence, _llm = \
            cls._build_generation_fns(circuit, db, cfg)

        # Reuse the EXACT stored probes — reproduction must not re-generate them.
        lo, hi = cls._dial_bounds(circuit)
        result = calibrate(
            gen_at, baseline_at, judge, divergence,
            probes=stored_probes, lo=lo, hi=hi,
            max_steps=cfg["step_budget"], margin=cfg["margin"])
        if progress_cb:
            progress_cb(80)

        repro_payload = {
            "probes": stored_probes,
            "config": cfg,
            "band": {"onset": result.onset, "sweet_spot": result.sweet_spot,
                     "cliff": result.cliff, "non_monotone": result.non_monotone},
            "usable_band": result.usable_band,
            "trace": result.trace,
        }
        verdict = ManifestService.calibration_reproduction_verdict(
            payload, repro_payload)
        repro_payload["reproduction_verdict"] = verdict
        repro_payload["reproduces"] = manifest_id

        from .manifest_service import validate_payload
        validate_payload("reproduction", repro_payload)  # path-safety
        man = ValidationManifest(kind="reproduction", payload=repro_payload,
                                 circuit_id=circuit.id,
                                 parent_manifest_id=manifest_id)
        db.add(man)
        db.commit()
        db.refresh(man)
        if progress_cb:
            progress_cb(100)
        return {"reproduces": manifest_id, "reproduction_manifest": man.id,
                "verdict": verdict}

    @staticmethod
    def _mark_no_usable_band(db, circuit_id: str, manifest_id: str) -> None:
        """A run that found no usable band: clear the in-flight marker WITHOUT
        clamping the dial or writing a calibration block. Distinct "no_band"
        status so a consumer can tell the latest measurement found nothing from a
        fresh successful calibration — and any PRIOR band on the circuit is not
        misrepresented as this run's result (R3). The circuit serves as before."""
        CircuitCalibrationService._mark_status(db, circuit_id, "no_band")

    @staticmethod
    def _mark_status(db, circuit_id: str, status: str) -> None:
        """Set the circuit's calibration_status without clamping or writing a
        band. Used for the non-clamping terminal outcomes (no_band,
        judge_unreliable)."""
        from ..models.circuit import Circuit
        circuit = db.query(Circuit).filter(Circuit.id == circuit_id).first()
        if circuit is not None:
            circuit.calibration_status = status
            db.commit()

    @staticmethod
    def _persist_manifest(db, circuit_id: str, payload: Dict[str, Any]) -> str:
        """Persist a self-contained `calibration` manifest (sync path, mirrors
        faithfulness). Returns the manifest id."""
        from ..models.validation_manifest import ValidationManifest
        from .manifest_service import validate_payload

        validate_payload("calibration", payload)
        man = ValidationManifest(kind="calibration", payload=payload,
                                 circuit_id=circuit_id)
        db.add(man)
        db.commit()
        db.refresh(man)
        return man.id

    @staticmethod
    def _write_calibration(db, circuit_id: str, band: Dict[str, Any]) -> None:
        """Write the band onto the circuit AND clamp the dial, through the
        CircuitDefinitionV1 contract (validators + version bump) — never a raw
        JSONB mutation (mirrors _write_circuit_faithfulness). Clamps
        intensity_range to [onset, cliff] and intensity to sweet_spot."""
        from ..models.circuit import Circuit
        from ..schemas.circuit_definition import CircuitDefinitionV1

        circuit = db.query(Circuit).filter(Circuit.id == circuit_id).first()
        if circuit is None:
            return
        onset, sweet, cliff = band["onset"], band["sweet_spot"], band["cliff"]
        if not (onset <= sweet <= cliff):
            raise CalibrationRunError(
                f"refusing to clamp to an inverted band "
                f"(onset={onset}, sweet_spot={sweet}, cliff={cliff})")
        budget = dict(circuit.budget or {})
        budget["intensity_range"] = [onset, cliff]
        budget["intensity"] = sweet
        try:
            defn = CircuitDefinitionV1(
                name=circuit.name, narrative=circuit.narrative,
                saes=circuit.saes, members=circuit.members, edges=circuit.edges,
                budget=budget, faithfulness=circuit.faithfulness, calibration=band)
        except Exception as e:
            raise CalibrationRunError(
                f"calibration produced an invalid circuit: {e}") from e
        circuit.budget = defn.budget.model_dump(mode="json")
        circuit.calibration = defn.calibration.model_dump(mode="json")
        circuit.version = (circuit.version or 1) + 1
        # REFUSE TO OVERWRITE A CANCELLATION — WITHOUT DESTROYING THE RESULT.
        #
        # R1 used `db.refresh(circuit)` here. `Session.refresh()` EXPIRES the
        # instance BEFORE autoflush, so every un-flushed attribute set just
        # above — the band, the clamp, the version bump — was erased and the
        # commit wrote only the status. On EVERY SUCCESSFUL RUN the feature
        # silently produced nothing while reporting "completed". Verified
        # against this repo's SQLAlchemy; the test fake's no-op `refresh`
        # is what hid it.
        #
        # A scalar column select reads the committed status without touching
        # the identity map or the pending writes.
        from sqlalchemy import select as _select
        
        from ..core.cancellation import is_cancelled as _is_cancelled
        
        _fresh = db.execute(
            _select(Circuit.calibration_status).where(Circuit.id == circuit.id)
        ).scalar()
        if _fresh is None:
            # The circuit was deleted mid-run. Writing "completed"
            # here flushes an UPDATE matching zero rows, which is a
            # StaleDataError out of a clean finish — the same shape
            # as the ObjectDeletedError removed from the extraction
            # path. There is nothing left to record to.
            logger.info("Circuit %s was deleted mid-run", circuit.id)
            return
        if _is_cancelled("circuit_calibration", _fresh):
            # A CANCELLED RUN MUST NOT MUTATE THE PRODUCTION CIRCUIT.
            #
            # The result fields above were assigned before this check
            # and the commit below is unconditional, so a cancel during
            # the tail left the row cancelled WITH the new band applied
            # and the version bumped — which is what export and
            # millm_import then ship. R1's refresh() hid this by
            # wiping everything; R2 fixed the success path and inverted
            # the cancel path.
            logger.info(
                "Circuit %s was cancelled during the tail of this run; "
                "discarding the result rather than applying it",
                circuit.id,
            )
            db.rollback()
            return
        circuit.calibration_status = "completed"
        db.commit()

    #: A short neutral prompt the model completes for onset/cliff generation.
    #: The probe's own prompt is what actually gets generated; this is the
    #: system-style lead-in kept minimal so the circuit's tint, not the prompt,
    #: drives divergence.
    GEN_MAX_TOKENS = 80

    @classmethod
    def _build_generation_fns(cls, circuit, db, cfg: Dict[str, Any]):
        """Build the GPU-backed (gen_at, baseline_at, judge, divergence, llm_call)
        for a real run. Mirrors CircuitFaithfulnessService's model loading; the
        hook is ADDITIVE steering (dial × strength × W_dec on the residual), not
        suppression.

        Executes only on a GPU host with the model + SAEs present — the same
        envelope as faithfulness's run() (which is likewise exercised on hardware,
        not in unit tests). The orchestration that consumes these callables
        (build_band, the clamp, the manifest) is fully unit-tested via injection.
        """
        seed = int(cfg.get("seed", 0))
        judge_endpoint = cfg.get("judge_endpoint")
        judge_model = cfg.get("judge_model")
        if not judge_endpoint or not judge_model:
            raise CalibrationRunError(
                "calibration needs a judge LLM: pass judge_endpoint + judge_model "
                "(an OpenAI-compatible endpoint, e.g. the miLLM instance). Without "
                "a judge the correctness cliff cannot be found.")

        from .enhanced_labeling_service import EnhancedLabelingService
        from .steering_core import (SteeringCoreError, build_steer_generator,
                                    load_model_and_structure,
                                    resolve_circuit_members)

        # The GENERATION half (model + SAEs + additive residual hook + greedy
        # gen_at/baseline_at) is the shared steering core — the same code the
        # recorder uses. Only member RESOLUTION differs per artifact type; for
        # calibration it is the circuit resolver. SteeringCoreError (missing
        # model/SAE, silent-drop refusal) surfaces as a CalibrationRunError so
        # the task's status/lifecycle is unchanged.
        try:
            # Load the model FIRST so member resolution puts the decoder weights
            # on the model's real device (the hook needs W_dec and hidden on the
            # same device).
            model, tokenizer, structure, disable_cache, device = \
                load_model_and_structure(circuit.model_id, db)
            _model_id, resolved = resolve_circuit_members(circuit, db, device)
            gen_at, baseline_at = build_steer_generator(
                model, tokenizer, structure, resolved,
                disable_cache=disable_cache, max_tokens=cls.GEN_MAX_TOKENS)
        except SteeringCoreError as e:
            raise CalibrationRunError(str(e)) from e

        # Judge + probe-generation LLM: the enhanced-labeling client, same
        # endpoint/key config as labeling (reasoning-model quirks handled there).
        labeler = EnhancedLabelingService(
            endpoint=judge_endpoint, model=judge_model,
            api_key=cfg.get("judge_api_key"))

        def llm_call(prompt, max_tokens):
            return labeler._call_llm(prompt, max_tokens)

        def judge(generation, expected):
            raw = labeler._call_llm(
                "You are grading whether a model response is factually CORRECT.\n"
                f"Question context: the correct answer is: {expected}\n"
                f"Model response: {generation}\n"
                "Reply with EXACTLY one word: CORRECT (the facts are right), "
                "DEGRADING (partly wrong/rambling), or BROKEN (confidently false "
                "or incoherent).", 8)
            return _parse_verdict(raw)

        divergence = cosine_text_divergence(embed=None)  # Jaccard; no embed dep

        return gen_at, baseline_at, judge, divergence, llm_call


def _parse_verdict(raw: str) -> str:
    """Map a judge LLM reply to correct | degrading | broken, robustly.

    Guards against negation ("not broken", "correct, not broken") which a naive
    `"broken" in text` misclassifies as broken and truncates the band (Feature 20
    R2). Strategy: find the FIRST verdict WORD (whole-token), ignoring one leading
    negation, and prefer an explicit CORRECT/DEGRADING before falling to BROKEN.
    """
    import re

    text = (raw or "").strip().lower()
    tokens = re.split(r"[^a-z]+", text)
    # Only UNAMBIGUOUS negators count. "no" is deliberately EXCLUDED (R3): a terse
    # judge reply "No, broken." means "no[t correct], broken" — treating the "no"
    # as negating the following "broken" would ship a broken dial as usable, the
    # exact inverse of the invariant. "not"/"isn't"/"aren't" directly before a
    # verdict word are the only safe negations ("not broken" = correct).
    verdict_words = {"correct", "degrading", "degrade", "degraded", "broken"}
    for i, tok in enumerate(tokens):
        if tok in verdict_words:
            negated = i > 0 and tokens[i - 1] in ("not", "isnt", "arent")
            if tok == "correct":
                return "correct"
            if tok.startswith("degrad"):
                return "degrading"
            # tok == "broken"
            return "correct" if negated else "broken"
    # No recognizable verdict word: default to the CONSERVATIVE reading so an
    # unparseable judge reply does not silently pass a dial as usable.
    return "broken"


def cosine_text_divergence(embed: Callable[[str], List[float]]):
    """Return a divergence(a, b) = 1 − cos(embed(a), embed(b)) using an injected
    embedder. Falls back to a token-Jaccard distance when no embedder is given —
    coarse but deterministic and dependency-free."""
    if embed is not None:
        def _div(a: str, b: str) -> float:
            va, vb = embed(a), embed(b)
            dot = sum(x * y for x, y in zip(va, vb))
            na = sum(x * x for x in va) ** 0.5
            nb = sum(y * y for y in vb) ** 0.5
            if na == 0 or nb == 0:
                return 1.0
            return 1.0 - dot / (na * nb)
        return _div

    def _jaccard(a: str, b: str) -> float:
        sa, sb = set(a.lower().split()), set(b.lower().split())
        if not sa and not sb:
            return 0.0
        return 1.0 - len(sa & sb) / len(sa | sb)
    return _jaccard
