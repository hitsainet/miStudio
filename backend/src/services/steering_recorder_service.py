"""Steered-transcript recorder (Steered Transcript Recorder increment).

Records `(dial, prompt, unsteered_output, steered_output)` transcripts for ANY
steerable artifact — a circuit, a cluster profile, or an ad-hoc feature set — on
the GPU, and persists them as a `steering_samples` manifest. The transcripts are
the raw material for a SEPARATE, post-run LLM meaning-analysis pass (hand them to
Opus 4.8): "what did steering this artifact semantically DO to the outputs across
the dial?". This is NOT a correctness judge — the path is judge-free.

Built on the unified `steering_core` (the same generation core calibration uses),
so a recorded transcript matches the calibrated band (same hook target, greedy
generation, dial-owned-here multiply).
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional

from .steering_core import (SteeringCoreError, build_steer_generator,
                            load_model_and_structure, resolve_circuit_members,
                            resolve_cluster_members, resolve_feature_members)

logger = logging.getLogger(__name__)

# Caps that bound the dials × prompts × tokens product on the single GPU.
MAX_RECORD_DIALS = 8
MAX_RECORD_PROMPTS = 8
MAX_RECORD_TOKENS = 200
MAX_RECORD_GENERATIONS = 64   # hard ceiling on prompts × (1 + dials)
DEFAULT_RECORD_TOKENS = 80
MAX_DIAL = 2.0                # servable ceiling (matches calibration MAX_DIAL)


class RecordConfigError(ValueError):
    """Bad record request — surfaces as a 422."""


class RecordRunError(RuntimeError):
    """Recording failed at run time."""


class _RecorderMisconfigured(BaseException):
    """A programming error discovered ON THE CANCELLATION PATH.

    BaseException for the same reason `OperatorCancelled` is: everything that
    catches `Exception` between here and the task boundary would record this as
    a failed run, which is the one outcome the cancellation work exists to stop
    producing. It is a bug either way — but it must arrive as a bug, not as a
    crash report about the user's job.
    """


class SteeringRecorderService:
    """Generate + record steered transcripts for a circuit / cluster / features."""

    @staticmethod
    def create_config(config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        cfg = dict(config or {})
        artifact = cfg.get("artifact")
        if not isinstance(artifact, dict) or artifact.get("kind") not in (
                "circuit", "cluster", "features"):
            raise RecordConfigError(
                "artifact must be {kind: 'circuit'|'cluster'|'features', ...}")
        # Per-kind required refs — validate HERE so a malformed artifact 422s at
        # the endpoint, BEFORE the GPU lock is taken and a job dispatched (R1: a
        # missing ref otherwise surfaced as an opaque KeyError deep in the task,
        # after the guard was held and — for features — a model was loaded).
        kind = artifact["kind"]
        if kind == "circuit" and not artifact.get("circuit_id"):
            raise RecordConfigError("circuit artifact needs a circuit_id")
        if kind == "cluster" and not artifact.get("cluster_profile_id"):
            raise RecordConfigError("cluster artifact needs a cluster_profile_id")
        if kind == "features":
            if not artifact.get("model_id"):
                raise RecordConfigError("features artifact needs a model_id")
            feats = artifact.get("features")
            if not isinstance(feats, list) or not feats:
                raise RecordConfigError("features artifact needs a non-empty features list")
            sae_by_layer: Dict[int, str] = {}
            for f in feats:
                if not isinstance(f, dict) or f.get("layer") is None \
                        or f.get("feature_idx") is None or not f.get("sae_id"):
                    raise RecordConfigError(
                        "each feature needs {layer, feature_idx, strength, sae_id}")
                # One SAE per layer — the resolver enforces this too, but check
                # it HERE so a mismatched sae_id 422s before the GPU lock + model
                # load, not deep in the task (R3: same class as the presence
                # check).
                L = int(f["layer"])
                if L in sae_by_layer and sae_by_layer[L] != f["sae_id"]:
                    raise RecordConfigError(
                        f"layer {L} names two different SAEs "
                        f"({sae_by_layer[L]} vs {f['sae_id']}); one SAE per layer")
                sae_by_layer[L] = f["sae_id"]

        dials = cfg.get("dials") or []
        prompts = cfg.get("prompts") or []
        if not isinstance(dials, list) or not dials:
            raise RecordConfigError("dials must be a non-empty list")
        if not isinstance(prompts, list) or not prompts:
            raise RecordConfigError("prompts must be a non-empty list")
        # Dedupe dials preserving order; validate range. A 0.0 dial is dropped —
        # the baseline is ALWAYS recorded separately, so a "steered" sample at
        # dial 0 would just duplicate the baseline and waste a budget slot (R1).
        seen, uniq = set(), []
        for d in dials:
            try:
                fd = float(d)
            except (TypeError, ValueError):
                raise RecordConfigError(f"dial {d!r} is not a number")
            if not (0.0 <= fd <= MAX_DIAL):
                raise RecordConfigError(
                    f"dial {fd} outside [0, {MAX_DIAL}] (the servable ceiling)")
            if fd == 0.0:
                continue   # baseline covers it
            if fd not in seen:
                seen.add(fd)
                uniq.append(fd)
        dials = uniq
        if not dials:
            raise RecordConfigError(
                "dials must include at least one steering strength above 0 "
                "(the unsteered baseline is always recorded)")
        prompts = [str(p) for p in prompts]
        if any(not p.strip() for p in prompts):
            raise RecordConfigError("prompts must be non-empty strings")
        if len(dials) > MAX_RECORD_DIALS:
            raise RecordConfigError(f"at most {MAX_RECORD_DIALS} dials")
        if len(prompts) > MAX_RECORD_PROMPTS:
            raise RecordConfigError(f"at most {MAX_RECORD_PROMPTS} prompts")

        try:
            max_tokens = int(cfg.get("max_tokens") or DEFAULT_RECORD_TOKENS)
        except (TypeError, ValueError):
            raise RecordConfigError("max_tokens must be an integer")
        if not (1 <= max_tokens <= MAX_RECORD_TOKENS):
            raise RecordConfigError(f"max_tokens must be in [1, {MAX_RECORD_TOKENS}]")
        try:
            seed = int(cfg.get("seed", 0))
        except (TypeError, ValueError):
            raise RecordConfigError("seed must be an integer")

        # Hard product cap: baseline (1) + one per dial, per prompt.
        generations = len(prompts) * (1 + len(dials))
        if generations > MAX_RECORD_GENERATIONS:
            raise RecordConfigError(
                f"{generations} generations requested (prompts × (1+dials)) "
                f"exceeds the {MAX_RECORD_GENERATIONS} ceiling; reduce dials or "
                "prompts")

        return {"artifact": artifact, "dials": dials, "prompts": prompts,
                "max_tokens": max_tokens, "seed": seed}

    @classmethod
    def _resolve(cls, artifact: Dict[str, Any], db, device):
        """Dispatch to the per-type resolver → (model_id, resolved_members)."""
        kind = artifact["kind"]
        if kind == "circuit":
            from ..models.circuit import Circuit
            circ = db.query(Circuit).filter(
                Circuit.id == artifact["circuit_id"]).first()
            if circ is None:
                raise RecordRunError(f"Circuit {artifact['circuit_id']} not found")
            return resolve_circuit_members(circ, db, device)
        if kind == "cluster":
            return resolve_cluster_members(artifact["cluster_profile_id"], db, device)
        if kind == "features":
            return resolve_feature_members(
                artifact["features"], artifact["model_id"], db, device)
        raise RecordRunError(f"unknown artifact kind {kind!r}")

    @classmethod
    def record_samples(cls, db, config: Dict[str, Any], *,
                       progress_cb: Optional[Callable[[int], None]] = None,
                       cancel_check: Optional[Callable[[], bool]] = None,
                       run_id: Optional[str] = None,
                       ) -> Dict[str, Any]:
        """GPU orchestrator (sync, called from the Celery task). Loads the model
        once, resolves the artifact's members, generates the baseline + each dial
        per prompt, and persists a `steering_samples` manifest with the text."""
        cfg = cls.create_config(config)
        artifact = cfg["artifact"]

        if progress_cb:
            progress_cb(5)
        try:
            # Model id from the artifact → load the model → resolve members on
            # the model's real device (W_dec + hidden must share a device).
            model_id = cls._artifact_model_id(artifact, db)
            model, tokenizer, structure, disable_cache, device = \
                load_model_and_structure(model_id, db)
            resolved_mid, resolved = cls._resolve(artifact, db, device)
            # The two resolution paths (model load vs member resolve) MUST agree —
            # they used to diverge silently (one fixed, one stale). Steer only
            # against the model we actually loaded.
            if resolved_mid != model_id:
                raise RecordRunError(
                    f"model id mismatch: loaded {model_id!r} but members resolve "
                    f"to {resolved_mid!r} — the two cluster resolution paths "
                    "disagree")
            gen_at, baseline_at = build_steer_generator(
                model, tokenizer, structure, resolved,
                disable_cache=disable_cache, max_tokens=cfg["max_tokens"])
        except SteeringCoreError as e:
            raise RecordRunError(str(e)) from e

        transcripts: List[Dict[str, Any]] = []
        total = len(cfg["prompts"]) * (1 + len(cfg["dials"]))
        done = 0
        for pi, prompt in enumerate(cfg["prompts"]):
            # THE CHECKPOINT. One prompt is 1 + len(dials) generations, each of
            # which is indivisible; the transcripts are accumulated in memory
            # and persisted as one manifest at the end, so abandoning here
            # writes nothing partial.
            if cancel_check is not None and cancel_check():
                from ..core.cancellation import OperatorCancelled

                # PAIRED, NOT MERELY CONVENTIONAL, AND NOT AN `assert`.
                #
                # `run_id` defaults to None so the signature stays additive, but
                # a checker without an id raises an OperatorCancelled whose
                # target is None — reported as a cancellation of nothing, and
                # `record_progress` writes nowhere.
                #
                # This was an `assert`, stripped under `python -O`. R2 made it a
                # ValueError, which fixed the -O half and NOT the other one:
                # ValueError is an ordinary Exception, so
                # `run_circuit_record`'s handler still turns it into a FAILED
                # run — the crash-report-instead-of-cancel outcome the comment
                # claimed to have removed. It has to be a BaseException to
                # travel the same path the cancellation itself does.
                if run_id is None:
                    raise _RecorderMisconfigured(
                        "record_samples was given a cancel_check but no run_id; "
                        "the cancellation would name no row"
                    )
                raise OperatorCancelled(
                    "steering_record", run_id,
                    detail=f"stopped at prompt {pi} of {len(cfg['prompts'])}",
                )
            unsteered = baseline_at(prompt, cfg["seed"])
            done += 1
            if progress_cb:
                progress_cb(5 + int(90 * done / total))
            samples = []
            for dial in cfg["dials"]:
                # dials never contains 0.0 (create_config drops it; baseline is
                # recorded separately as unsteered_output), so every dial here is
                # a real steered generation.
                samples.append({"dial": dial,
                                "steered_output": gen_at(dial, prompt)})
                done += 1
                if progress_cb:
                    progress_cb(5 + int(90 * done / total))
            transcripts.append({"prompt_index": pi, "prompt": prompt,
                                "unsteered_output": unsteered, "samples": samples})

        payload = {
            "artifact": {"kind": artifact["kind"], **cls._artifact_ref(artifact)},
            "dials": cfg["dials"], "prompts": cfg["prompts"],
            "config": {"max_tokens": cfg["max_tokens"], "seed": cfg["seed"],
                       "model_hf_id": cls._model_hf_id(model_id, db)},
            "transcripts": transcripts,
        }
        manifest_id = cls._persist(db, artifact, payload)
        if progress_cb:
            progress_cb(100)
        return {"manifest_ref": manifest_id, "artifact": payload["artifact"],
                "counts": {"prompts": len(cfg["prompts"]),
                           "dials": len(cfg["dials"]), "generations": total}}

    # ── helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _artifact_model_id(artifact, db):
        kind = artifact["kind"]
        if kind == "features":
            return artifact["model_id"]
        if kind == "circuit":
            from ..models.circuit import Circuit
            c = db.query(Circuit).filter(
                Circuit.id == artifact["circuit_id"]).first()
            if c is None:
                raise RecordRunError(f"Circuit {artifact['circuit_id']} not found")
            return c.model_id
        if kind == "cluster":
            from ..models.cluster_profile import ClusterProfile
            from .steering_core import (SteeringCoreError, _cluster_model_id,
                                        _cluster_sae_ids_by_layer)
            p = db.query(ClusterProfile).filter(
                ClusterProfile.id == artifact["cluster_profile_id"]).first()
            if p is None:
                raise RecordRunError(
                    f"Cluster profile {artifact['cluster_profile_id']} not found")
            # SAME authority the resolver uses: CLUSTERS-arc profiles store
            # sae_id but NOT model_id, so derive it from the SAE. (Fixing only
            # steering_core left this parallel copy returning None — the recorder
            # loads the model HERE, before _resolve, so a stale copy = "Model
            # None not found".)
            try:
                return _cluster_model_id(p, _cluster_sae_ids_by_layer(p, db), db)
            except SteeringCoreError as e:
                raise RecordRunError(str(e)) from e
        raise RecordRunError(f"unknown artifact kind {kind!r}")

    @staticmethod
    def _artifact_ref(artifact) -> Dict[str, Any]:
        kind = artifact["kind"]
        if kind == "circuit":
            return {"circuit_id": artifact["circuit_id"]}
        if kind == "cluster":
            return {"cluster_profile_id": artifact["cluster_profile_id"]}
        # features: store the specs (no file paths) so the record is self-contained.
        return {"features": artifact["features"], "model_id": artifact["model_id"]}

    @staticmethod
    def _model_hf_id(model_id, db):
        from ..models.model import Model
        m = db.query(Model).filter(Model.id == model_id).first()
        return getattr(m, "repo_id", None) if m else None

    @staticmethod
    def _persist(db, artifact, payload) -> str:
        from ..models.validation_manifest import ValidationManifest
        from .manifest_service import validate_payload

        validate_payload("steering_samples", payload)
        # A circuit artifact links the manifest to its circuit for retrieval;
        # cluster/feature manifests carry the artifact in the payload only.
        circuit_id = (artifact["circuit_id"] if artifact["kind"] == "circuit"
                      else None)
        man = ValidationManifest(kind="steering_samples", payload=payload,
                                 circuit_id=circuit_id)
        db.add(man)
        db.commit()
        db.refresh(man)
        return man.id
