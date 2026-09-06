"""
Circuit capture service (Feature 016, BR-005/BR-006/BR-023 — FTDD §1).

Sync (Celery-worker-facing) orchestration of a capture run:

  probe (~PROBE_SAMPLES docs) → cost estimate → [stop if not confirmed]
  → full batch loop: ONE forward per batch, multi-layer residual hooks →
    per-layer SAE encode on-GPU → threshold max(θ_floor, ε·max_act_i)
    (per-feature max from the probe; missing ⇒ floor-only, never skip) →
    event/errnorm append (+ optional attention top-k sidecar)
  → per-document 80/20 split (seeded, recorded) → manifest.json (atomic)

The DB row (circuit_capture_runs.manifest) mirrors manifest.json exactly so
listings never touch disk. Stale-flagging: capture records SAE fingerprints;
`mark_stale_for_sae` flips `stale` when a referenced SAE changes.
"""

import hashlib
import json
import logging
import os
import shutil
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from ..core.config import settings
from ..models.circuit_runs import CircuitCaptureRun
from ..models.dataset import Dataset
from ..models.dataset_tokenization import DatasetTokenization, TokenizationStatus
from ..models.external_sae import ExternalSAE
from .circuit_capture_store import open_writers
from ..core.clock import utc_now

logger = logging.getLogger(__name__)

PROBE_SAMPLES = 32
MAX_SEQ_LENGTH = 512  # capture cap — also keeps token_pos comfortably in u16
DEFAULT_EPSILON = 0.1
DEFAULT_THETA_FLOOR = 0.01
DEFAULT_SPLIT_RATIO = 0.8
DEFAULT_SAMPLE_CAP = 2000
EVENT_BYTES = 12  # sizeof EVENT_DTYPE row (u32 feature_idx widened it from 10)
STORE_SIZE_MULTIPLIER = 5.0   # abort a capture exceeding 5× its own estimate
MIN_FREE_DISK_BYTES = 5 * 2**30  # refuse/abort if <5 GB free on the data volume


class CaptureConfigError(ValueError):
    """Invalid capture configuration — surfaces as a 422."""


class CaptureConflictError(RuntimeError):
    """A GPU circuit task is already active — surfaces as a 409."""


def captures_dir() -> Path:
    return settings.data_dir / "circuit_captures"


def size_ceiling_bytes(events_est: int) -> float:
    """The store-size abort threshold (R1 QA-P1): 5× the probe estimate, with
    a 64 MB floor so a tiny estimate doesn't abort a legitimate small capture."""
    return max(events_est * EVENT_BYTES * STORE_SIZE_MULTIPLIER, 64 * 2**20)


def exceeds_size_ceiling(buffered_events: int, events_est: int) -> bool:
    return buffered_events * EVENT_BYTES > size_ceiling_bytes(events_est)


class CircuitCaptureService:
    # ── concurrency guard ────────────────────────────────────────────────

    # Postgres advisory-lock key: serializes the check-then-insert so two
    # concurrent requests can't both pass the guard (R2 B2).
    _GPU_LOCK_KEY = 0x1C1C_C0DE

    @staticmethod
    def assert_no_active_gpu_run(db) -> None:
        """One GPU circuit task at a time on the single 3090 (R1 QA-P1 / R2
        Q1). Covers BOTH captures (this table) AND attribution passes (on the
        discovery row) — attribution loads a model too (R2 Q1). Serialized by
        a transaction-scoped advisory lock so the check-then-insert can't race
        (R2 B2); the lock releases at commit/rollback."""
        from sqlalchemy import text

        from ..models.circuit_runs import CircuitDiscoveryRun

        db.execute(text("SELECT pg_advisory_xact_lock(:k)"),
                   {"k": CircuitCaptureService._GPU_LOCK_KEY})
        active = db.query(CircuitCaptureRun).filter(
            CircuitCaptureRun.status.in_(
                ("pending", "estimating", "running"))).first()
        if active is not None:
            raise CaptureConflictError(
                f"Capture {active.id} is already {active.status} — one GPU "
                f"circuit task runs at a time; wait or cancel it first")
        attr = db.query(CircuitDiscoveryRun).filter(
            CircuitDiscoveryRun.attribution_status.in_(
                ("pending", "running"))).first()
        if attr is not None:
            raise CaptureConflictError(
                f"Attribution pass on {attr.id} is {attr.attribution_status} — "
                f"one GPU circuit task runs at a time; wait or cancel it first")
        # Validation is a GPU task too (R1 #7/Q1 — the guard missed it, so a
        # capture/attribution could run concurrently with a validation pass).
        val = db.query(CircuitDiscoveryRun).filter(
            CircuitDiscoveryRun.validation_status.in_(
                ("pending", "running"))).first()
        if val is not None:
            raise CaptureConflictError(
                f"Validation pass on {val.id} is {val.validation_status} — "
                f"one GPU circuit task runs at a time; wait or cancel it first")
        # Faithfulness runs on a circuit and loads a model too (R2 B-5).
        from ..models.circuit import Circuit
        faith = db.query(Circuit).filter(
            Circuit.faithfulness_status.in_(("pending", "running"))).first()
        if faith is not None:
            raise CaptureConflictError(
                f"Faithfulness pass on {faith.id} is {faith.faithfulness_status} "
                f"— one GPU circuit task runs at a time; wait or cancel it first")
        # Calibration (Feature 20) also runs on a circuit and loads a model —
        # same single-GPU guard, or a calibration could race a capture/
        # faithfulness and OOM as an opaque task failure.
        calib = db.query(Circuit).filter(
            Circuit.calibration_status.in_(("pending", "running"))).first()
        if calib is not None:
            raise CaptureConflictError(
                f"Calibration pass on {calib.id} is {calib.calibration_status} "
                f"— one GPU circuit task runs at a time; wait or cancel it first")
        # Steered-transcript recording (circuit/cluster/feature) also loads a
        # model on the single GPU — its marker lives in steering_record_runs
        # (cluster/feature jobs have no circuit row).
        from ..models.steering_record_run import SteeringRecordRun
        rec = db.query(SteeringRecordRun).filter(
            SteeringRecordRun.status.in_(("pending", "running"))).first()
        if rec is not None:
            raise CaptureConflictError(
                f"A steering-record job ({rec.id}) is {rec.status} — one GPU "
                f"task runs at a time; wait or cancel it first")

    # ── run creation / validation (called from the endpoint) ─────────────

    @staticmethod
    def create_run(db, config: Dict[str, Any]) -> CircuitCaptureRun:
        """Validate config against the DB and create the run row (pending)."""
        dataset_id = config.get("dataset_id")
        layers = config.get("layers") or []
        if not dataset_id:
            raise CaptureConfigError("dataset_id is required")
        if not layers:
            raise CaptureConfigError("at least one {layer, sae_id} entry is required")
        seen_layers = set()
        for entry in layers:
            if "layer" not in entry or "sae_id" not in entry:
                raise CaptureConfigError("each layers[] entry needs layer and sae_id")
            if entry["layer"] in seen_layers:
                raise CaptureConfigError(f"duplicate layer {entry['layer']}")
            seen_layers.add(entry["layer"])

        dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
        if dataset is None:
            raise CaptureConfigError(f"Dataset {dataset_id} not found")

        model_id = config.get("model_id")
        saes: Dict[str, ExternalSAE] = {}
        for entry in layers:
            sae = db.query(ExternalSAE).filter(
                ExternalSAE.id == entry["sae_id"]).first()
            if sae is None:
                raise CaptureConfigError(f"SAE {entry['sae_id']} not found")
            if not sae.local_path:
                raise CaptureConfigError(f"SAE {sae.id} has no local path")
            if sae.layer is not None and sae.layer != entry["layer"]:
                raise CaptureConfigError(
                    f"SAE {sae.id} was trained on layer {sae.layer}, "
                    f"config asks for layer {entry['layer']} — own-layer rule")
            saes[entry["sae_id"]] = sae
            if model_id is None:
                model_id = sae.model_id
            # Wide-SAE bound: feature_idx is u32, but assert d_sae fits so a
            # broken SAE record surfaces at config time, not mid-capture.
            if sae.n_features is not None and sae.n_features > 2**32:
                raise CaptureConfigError(
                    f"SAE {sae.id} has {sae.n_features} features — exceeds u32")

        # model_id must resolve, else the tokenization filter matches NULL and
        # fails confusingly (R1 CR#4).
        if model_id is None:
            raise CaptureConfigError(
                "model_id could not be resolved — pass it explicitly or use "
                "SAEs whose model_id is set")

        _validate_attention_config(db, config, model_id, seen_layers)

        tokenization = db.query(DatasetTokenization).filter(
            DatasetTokenization.dataset_id == dataset_id,
            DatasetTokenization.model_id == model_id,
        ).first()
        if tokenization is None or tokenization.status != TokenizationStatus.READY:
            raise CaptureConfigError(
                f"Dataset {dataset_id} has no READY tokenization for model "
                f"{model_id} — tokenize it first")

        epsilon = float(config.get("epsilon", DEFAULT_EPSILON))
        if not (0.0 <= epsilon < 1.0):
            raise CaptureConfigError("epsilon must be in [0, 1)")
        sample_cap = int(config.get("sample_cap", DEFAULT_SAMPLE_CAP))
        if sample_cap < PROBE_SAMPLES:
            raise CaptureConfigError(f"sample_cap must be >= {PROBE_SAMPLES}")

        manifest = {
            "corpus": {
                "dataset_id": dataset_id,
                "tokenization_id": tokenization.id,
                "sample_cap": sample_cap,
            },
            "model_id": model_id,
            "layers": [
                {"layer": e["layer"], "sae_id": e["sae_id"],
                 "threshold_mode": "epsilon_max" if epsilon > 0 else "floor",
                 "epsilon": epsilon,
                 "theta_floor": float(config.get("theta_floor", DEFAULT_THETA_FLOOR))}
                for e in layers
            ],
            "split": {
                "method": "per_document",
                "ratio": DEFAULT_SPLIT_RATIO,
                "seed": int(config.get("split_seed", 42)),
                "heldout_docs": [],  # filled at capture completion
            },
            "attention_capture": config.get("attention_capture"),  # {layers, heads, top_k}|None
            "created_at": utc_now().isoformat(),
            "stale": False,
        }
        run = CircuitCaptureRun(status="pending", manifest=manifest)
        db.add(run)
        db.commit()
        db.refresh(run)
        return run

    # ── stale flagging ───────────────────────────────────────────────────

    @staticmethod
    def mark_stale_for_sae(db, sae_id: str) -> int:
        """Flag (never delete) every completed run referencing this SAE."""
        runs = db.query(CircuitCaptureRun).filter(
            CircuitCaptureRun.status == "completed",
            CircuitCaptureRun.stale == False,  # noqa: E712
        ).all()
        n = 0
        for run in runs:
            if any(l.get("sae_id") == sae_id
                   for l in (run.manifest or {}).get("layers", [])):
                run.stale = True
                manifest = dict(run.manifest)
                manifest["stale"] = True
                run.manifest = manifest
                n += 1
        if n:
            db.commit()
        return n

    # ── deletion ─────────────────────────────────────────────────────────

    @staticmethod
    def delete_run(db, run: CircuitCaptureRun) -> None:
        if run.status == "running":
            raise CaptureConfigError("Cannot delete a running capture — cancel first")
        if run.store_path:
            store = settings.resolve_data_path(run.store_path)
            # Containment: never rm outside the captures root.
            if store.is_dir() and captures_dir() in store.parents:
                shutil.rmtree(store, ignore_errors=True)
        db.delete(run)
        db.commit()

    # ── worker body ──────────────────────────────────────────────────────

    @staticmethod
    def run_capture(db, run_id: str, *, confirmed: bool,
                    cancel_check=None, progress_cb=None) -> Dict[str, Any]:
        """Execute (probe [+ capture]) for a run. Heavy imports stay inside
        so the module imports cleanly on GPU-less API processes."""
        import torch
        from datasets import load_from_disk

        from ..ml.forward_hooks import HookManager, HookType
        from ..ml.model_loader import load_model_from_hf
        from ..models.model import Model
        from .extraction_service import cleanup_gpu_memory

        run = db.query(CircuitCaptureRun).filter(
            CircuitCaptureRun.id == run_id).first()
        if run is None:
            raise ValueError(f"Capture run {run_id} not found")
        manifest = dict(run.manifest)
        # Hoisted above the model load: whether attention is being captured
        # decides the attention backend, which is a load-time argument.
        attn_cfg = manifest.get("attention_capture") or None
        wants_attention = bool(attn_cfg and attn_cfg.get("layers"))

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_record = db.query(Model).filter(
            Model.id == manifest["model_id"]).first()
        if model_record is None:
            raise CaptureConfigError(f"Model {manifest['model_id']} not found")

        tokenization = db.query(DatasetTokenization).filter(
            DatasetTokenization.id == manifest["corpus"]["tokenization_id"]).first()
        dataset = load_from_disk(
            str(settings.resolve_data_path(tokenization.tokenized_path)))
        sample_cap = min(int(manifest["corpus"]["sample_cap"]), len(dataset))
        dataset = dataset.select(range(sample_cap))

        model = tokenizer = None
        saes: Dict[int, Any] = {}
        try:
            from ..models.model import QuantizationFormat
            resolved_model_path = (settings.resolve_data_path(model_record.file_path)
                                   if model_record.file_path else None)
            model_is_downloaded = bool(resolved_model_path
                                       and resolved_model_path.exists())
            model, tokenizer, _config, _meta = load_model_from_hf(
                repo_id=model_record.repo_id,
                quant_format=QuantizationFormat(model_record.quantization),
                cache_dir=resolved_model_path,
                device_map=device,
                local_files_only=model_is_downloaded,
                # Attention PROBABILITIES require eager. SDPA and flash kernels
                # never materialise them — `sdpa_attention_forward` returns
                # `(attn_output, None)` — so a capture asking for attention on
                # an SDPA model collected nothing and then indexed an empty
                # tuple. Eager is slower, so it is requested only when needed.
                attn_implementation="eager" if wants_attention else None,
            )
            model.eval()

            # ASSERT THE REQUEST TOOK, before the probe rather than 90 seconds
            # into the capture. transformers can decline or an OOM fallback can
            # drop it, and the failure mode is silent: the hooks simply capture
            # nothing.
            if wants_attention:
                resolved_attn = _meta.get("attn_implementation")
                if resolved_attn != "eager":
                    raise CaptureConfigError(
                        "attention capture needs eager attention, but the model "
                        f"loaded with {resolved_attn!r}. Attention probabilities "
                        "cannot be recorded from this backend."
                    )

            fingerprints = {}
            d_sae_by_layer: Dict[int, Optional[int]] = {}
            for entry in manifest["layers"]:
                sae_record = db.query(ExternalSAE).filter(
                    ExternalSAE.id == entry["sae_id"]).first()
                sae = _load_sae_sync(sae_record, device)
                saes[entry["layer"]] = sae
                fingerprints[entry["sae_id"]] = _sae_fingerprint(sae)
                # From the RECORD, not the tensor. `W_enc` is [d_sae, d_model]
                # here and [d_model, d_sae] in other conventions, so reading a
                # shape would silently return d_model for some SAEs — and this
                # number is the denominator of the density the user is shown.
                d_sae_by_layer[entry["layer"]] = sae_record.n_features
            manifest["sae_fingerprints"] = fingerprints

            # WHICH POINT OF THE RESIDUAL STREAM THIS CAPTURE READ.
            #
            # "residual" is resolved by pattern-matching a layer's children, and
            # a Llama-style layer offers two candidates — `input_layernorm` and
            # `post_attention_layernorm`. Until the patterns were given a fixed
            # order that choice varied per worker process, so a capture could
            # read a different point than the extraction that trained its SAE,
            # with no disagreement recorded anywhere. Feeding an SAE
            # out-of-distribution activations makes its learned thresholds
            # meaningless and it fires densely — which is what an implausible
            # event estimate looks like from the outside.
            #
            # Recorded so that a future mismatch is visible in the artifact
            # rather than inferred from a suspicious number.
            from ..ml.layer_discovery import discover_transformer_structure
            try:
                _struct = discover_transformer_structure(
                    model, architecture_hint=getattr(
                        model.config, "model_type", "auto"))
                manifest["hook_points"] = {
                    "residual": _struct.residual_norm_module,
                    "attention": ("per-layer self-attention module"
                                  if wants_attention else None),
                }
            except Exception as exc:  # noqa: BLE001 - provenance, not control flow
                logger.warning("Could not record hook points: %s", exc)

            layer_indices = sorted(saes.keys())

            run.status = "estimating"
            run.progress = 0.0
            db.commit()

            # ── probe: per-feature max + event-rate estimate ────────────
            t0 = time.monotonic()
            probe_max, probe_events, probe_tokens = _probe(
                model, tokenizer, dataset, saes, layer_indices, device,
                epsilon_by_layer={e["layer"]: e["epsilon"]
                                  for e in manifest["layers"]},
                floor_by_layer={e["layer"]: e["theta_floor"]
                                for e in manifest["layers"]})
            probe_seconds = time.monotonic() - t0
            total_tokens_est = probe_tokens / PROBE_SAMPLES * sample_cap
            events_est = int(probe_events / max(probe_tokens, 1) * total_tokens_est)
            # MEASURED SPARSITY, REPORTED. A capture's size is set almost
            # entirely by how densely its SAEs fire on THIS corpus, and that
            # number was nowhere in the estimate — so a run needing 240 GB
            # looked like any other, distinguishable only by a big number with
            # no explanation attached. These SAEs fire ~25x denser on
            # `hard-negatives` than the 0.48% they recorded during training,
            # which is the whole reason the run was enormous.
            per_token = (probe_events / max(probe_tokens, 1)
                         / max(len(layer_indices), 1))
            widths = [w for w in d_sae_by_layer.values() if w]
            density = (per_token / (sum(widths) / len(widths))) if widths else None

            estimate = {
                "events": events_est,
                "bytes": events_est * EVENT_BYTES,
                "minutes": round(probe_seconds / PROBE_SAMPLES
                                 * sample_cap / 60.0, 1),
                "probe_samples": PROBE_SAMPLES,
                "probe_events": int(probe_events),
                "features_per_token": round(per_token, 1),
                "density": round(density, 5) if density is not None else None,
                "memory_required_bytes": (events_est * EVENT_BYTES
                                          * PEAK_MEMORY_MULTIPLIER),
                "memory_budget_bytes": _memory_budget_bytes(),
            }
            manifest["estimate"] = estimate
            run.manifest = manifest
            if not confirmed:
                run.status = "estimated"
                run.progress = None
                db.commit()
                return {"status": "estimated", "estimate": estimate}

            # REFUSE A CAPTURE THAT CANNOT FIT IN MEMORY, before it starts.
            #
            # Every event is buffered in host RAM until `finalize()`, so the
            # binding constraint is memory and nothing was checking it. This is
            # the guard `cap_cda1e1da6a0a` needed: it was confirmed against an
            # estimate of 20 billion events, ran to 45.6%, and the worker was
            # OOM-killed with ~110 GB of rows held. Failing here costs nothing;
            # failing there wedged the single-GPU guard for every later capture.
            over = _exceeds_memory_budget(events_est)
            if over is not None:
                run.status = "failed"
                suggested = _suggested_sample_cap(events_est, sample_cap)
                fix = (f"lower sample_cap to about {suggested:,} "
                       f"(from {sample_cap:,}), capture fewer layers, "
                       "or use an SAE that is sparser on this data"
                       if suggested is not None else
                       "lower sample_cap, capture fewer layers, or use an SAE "
                       "that is sparser on this data")
                run.error_message = (
                    f"Capture refused: {over}. This corpus activates "
                    f"{estimate.get('features_per_token')} features per token "
                    f"({(estimate.get('density') or 0) * 100:.1f}% of the "
                    f"dictionary, against the ~0.5% a well-trained SAE gives) — "
                    f"{fix}. Raising epsilon barely helps at this density: "
                    "these features have too little dynamic range for a "
                    "relative threshold to remove much.")
                db.commit()
                return {"status": "failed", "reason": "memory_ceiling",
                        "estimate": estimate}

            # ── full capture ─────────────────────────────────────────────
            store_dir = captures_dir() / run.id
            store_dir.mkdir(parents=True, exist_ok=True)
            # Persist store_path NOW, not at success (R3 B-R3-1): if the worker
            # is OOM-killed mid-capture, cleanup_stuck_circuit_runs can still
            # rmtree the orphaned partial store (its guard is `if store_path`).
            run.status = "running"
            run.store_path = str(store_dir)
            db.commit()
            # Store-size guardrail (R1 QA-P1): abort if the true event rate
            # blows past the probe estimate, or the volume runs low on space.
            if shutil.disk_usage(store_dir).free < MIN_FREE_DISK_BYTES:
                run.status = "failed"
                run.error_message = "Insufficient free disk on the data volume"
                db.commit()
                shutil.rmtree(store_dir, ignore_errors=True)
                return {"status": "failed", "reason": "disk"}
            writers = {L: open_writers(store_dir, L,
                                       attention=bool(attn_cfg and L in
                                                      (attn_cfg.get("layers") or [])))
                       for L in layer_indices}

            batch_size = 8
            n_docs = len(dataset)
            doc_lengths: Dict[int, int] = {}
            for batch_start in range(0, n_docs, batch_size):
                if cancel_check is not None and cancel_check():
                    run.status = "cancelled"
                    db.commit()
                    shutil.rmtree(store_dir, ignore_errors=True)
                    return {"status": "cancelled"}
                batch = dataset[batch_start:min(batch_start + batch_size, n_docs)]
                input_ids, attention_mask, lengths = _pad_batch(
                    batch, tokenizer)
                for i, L in enumerate(lengths):
                    doc_lengths[batch_start + i] = L
                _capture_batch(
                    model, saes, layer_indices, writers,
                    input_ids.to(model.device), attention_mask.to(model.device),
                    batch_start, lengths,
                    epsilon_by_layer={e["layer"]: e["epsilon"]
                                      for e in manifest["layers"]},
                    floor_by_layer={e["layer"]: e["theta_floor"]
                                    for e in manifest["layers"]},
                    probe_max=probe_max, attn_cfg=attn_cfg)
                # Running byte estimate from buffered events (u32 idx + u16
                # pos + u32 doc + f16 act ≈ EVENT_BYTES/event, plus errnorm).
                #
                # ATTENTION ROWS COUNT TOO. They were excluded, which was
                # harmless only for as long as the sidecar was always empty —
                # attention capture could never produce a row, because it
                # crashed first. Now that it works, a default request (2000 docs
                # x 32 heads x 512 queries x top_k 4) is ~131M rows, buffered in
                # host RAM until finalize(), and this guard was the only thing
                # standing between that and an OOM it could not see coming.
                buffered = _buffered_rows(writers)
                # ABSOLUTE, not relative. `exceeds_size_ceiling` is 5x this
                # run's own estimate, so an estimate that is itself enormous
                # authorises something five times more enormous — it can only
                # catch a run that under-estimated, never one whose estimate was
                # the problem. Memory is the hard bound and does not scale with
                # anyone's expectations.
                over_mem = _exceeds_memory_budget(buffered)
                if over_mem is not None:
                    run.status = "failed"
                    run.error_message = (
                        f"Capture aborted to avoid running the worker out of "
                        f"memory: {over_mem}")
                    db.commit()
                    shutil.rmtree(store_dir, ignore_errors=True)
                    return {"status": "failed", "reason": "memory_ceiling"}
                if exceeds_size_ceiling(buffered, events_est):
                    run.status = "failed"
                    run.error_message = (
                        f"Capture exceeded {STORE_SIZE_MULTIPLIER}× its size "
                        f"estimate ({buffered} events) — aborted to protect "
                        f"the data volume; lower sample_cap or raise epsilon")
                    db.commit()
                    shutil.rmtree(store_dir, ignore_errors=True)
                    return {"status": "failed", "reason": "size_ceiling"}
                pct = min(99.0, (batch_start + batch_size) / n_docs * 100.0)
                run.progress = pct
                db.commit()
                if progress_cb is not None:
                    progress_cb(pct)

            # finalize writers
            events_total = 0
            for L, (ev, en, at) in writers.items():
                events_total += ev.finalize()
                en.finalize()
                if at is not None:
                    at.finalize()

            # per-document split, seeded, recorded
            rng = np.random.default_rng(int(manifest["split"]["seed"]))
            all_docs = np.arange(n_docs)
            perm = rng.permutation(all_docs)
            cut = int(len(perm) * float(manifest["split"]["ratio"]))
            heldout = sorted(int(d) for d in perm[cut:])
            manifest["split"]["heldout_docs"] = heldout
            manifest["doc_lengths"] = {str(k): v for k, v in doc_lengths.items()}
            manifest["counts"] = {"documents": n_docs, "events": events_total,
                                  "tokens": int(sum(doc_lengths.values()))}
            bytes_total = sum(f.stat().st_size for f in store_dir.iterdir())
            manifest["bytes"] = bytes_total

            _write_manifest_atomic(store_dir / "manifest.json", manifest)

            # Last-writer race: a cancel between the final cancel-check and here
            # must NOT be clobbered by 'completed' (R1 CR#6). Re-read status.
            db.refresh(run)
            if run.status == "cancelled":
                shutil.rmtree(store_dir, ignore_errors=True)
                return {"status": "cancelled"}
            run.manifest = manifest
            run.store_path = str(store_dir)
            run.events_total = events_total
            run.bytes_total = bytes_total
            run.status = "completed"
            run.progress = 100.0
            db.commit()
            return {"status": "completed", "events": events_total,
                    "bytes": bytes_total, "heldout_docs": len(heldout)}
        finally:
            cleanup_gpu_memory(
                [m for m in [model, *saes.values()] if m is not None],
                context=f"circuit_capture:{run_id}")


# ── helpers ──────────────────────────────────────────────────────────────

def _validate_attention_config(db, config, model_id, capture_layers) -> None:
    """Refuse an unrunnable attention request AT SUBMIT, not 90 seconds in.

    Every one of these was previously discovered on the GPU, after the run had
    taken a slot on the single-GPU queue — possibly behind a 45-minute fit.

    Tri-state on "is this layer an attention layer", deliberately. The API
    process has no model in memory, so the honest answers are known-bad (refuse)
    and not-known (let the worker decide, which it now does before the probe).
    Guessing from an architecture name is the third option and it is the wrong
    one — the same reasoning as `jlens_acquire_service.model_dims`.
    """
    from ..models.model import Model

    attn = config.get("attention_capture") or None
    if not attn:
        return

    layers = attn.get("layers") or []
    if not layers:
        raise CaptureConfigError(
            "attention_capture was requested with no layers; either name the "
            "layers or turn it off (it forces eager attention, which is slower)")

    # An attention layer outside the capture's SAE layers silently produced no
    # file: `open_writers` only creates a sidecar writer for layers it knows.
    stray = sorted(set(layers) - set(capture_layers))
    if stray:
        raise CaptureConfigError(
            f"attention layers {stray} are not among the capture's layers "
            f"{sorted(capture_layers)} — no sidecar would be written for them")

    dupes = sorted({L for L in layers if layers.count(L) > 1})
    if dupes:
        raise CaptureConfigError(f"duplicate attention layers {dupes}")

    top_k = int(attn.get("top_k") or 0)
    if top_k < 1:
        raise CaptureConfigError("attention top_k must be at least 1")

    model = db.query(Model).filter(Model.id == model_id).first()
    if model is None:
        raise CaptureConfigError(f"Model {model_id} not found")

    arch = model.architecture_config or {}
    n_layers = arch.get("num_hidden_layers")
    if isinstance(n_layers, int):
        oob = sorted(L for L in layers if not 0 <= L < n_layers)
        if oob:
            raise CaptureConfigError(
                f"attention layers {oob} are out of range for a model with "
                f"{n_layers} layers")

    n_heads = arch.get("num_attention_heads")
    heads = attn.get("heads")
    if heads and isinstance(n_heads, int):
        oob = sorted(h for h in heads if not 0 <= h < n_heads)
        if oob:
            raise CaptureConfigError(
                f"attention heads {oob} are out of range for a model with "
                f"{n_heads} heads")

    # KNOWN-BAD only: a config that declares per-layer block types tells us
    # exactly which layers have attention. Absent that key, say nothing.
    layer_types = arch.get("layer_types")
    if isinstance(layer_types, list) and layer_types:
        not_attention = sorted(
            L for L in layers
            if L < len(layer_types) and layer_types[L] != "full_attention")
        if not_attention:
            raise CaptureConfigError(
                f"layers {not_attention} are not attention layers on this model "
                f"({', '.join(sorted(set(layer_types)))}) — they have no "
                "attention probabilities to capture")


def _load_sae_sync(sae_record: "ExternalSAE", device: str):
    """Sync SAE load (worker context) — the same path SteeringService.load_sae
    uses internally: auto-detect format → create → load weights → eval."""
    import torch

    from ..ml.community_format import load_sae_auto_detect
    from ..ml.sparse_autoencoder import create_sae

    sae_path = settings.resolve_data_path(sae_record.local_path)
    state_dict, config, _fmt = load_sae_auto_detect(sae_path, device="cpu")
    d_in = sae_record.d_model or (
        state_dict["encoder.weight"].shape[1] if "encoder.weight" in state_dict
        else state_dict["W_enc"].shape[0])
    d_sae = sae_record.n_features or (
        state_dict["encoder.weight"].shape[0] if "encoder.weight" in state_dict
        else state_dict["W_enc"].shape[1])
    architecture = (sae_record.architecture or "standard").lower()
    # MIS-E2E-083: carry the TRAINED normalization convention. `config` was
    # loaded and discarded, so every SAE built here silently took the
    # constructor default instead of the mode it was trained with — and no
    # consumer could recover it afterwards.
    normalize_activations = getattr(config, "normalize_activations", None) or "none"
    sae = create_sae(
        architecture,
        hidden_dim=d_in,
        latent_dim=d_sae,
        normalize_activations=normalize_activations,
    )
    cleaned = {k.removeprefix("model."): v for k, v in state_dict.items()}
    sae.load_state_dict(cleaned, strict=False)
    sae.to(device).eval()
    return sae


def _sae_fingerprint(sae) -> str:
    """Cheap decoder identity: shape + parameter checksum."""
    import torch

    from .steering_service import resolve_decoder_weight
    w = resolve_decoder_weight(sae)
    if w is None:
        return "unknown"
    with torch.no_grad():
        return f"{tuple(w.shape)}:{float(w.float().abs().sum()):.3f}"


def _pad_batch(batch: Dict[str, Any], tokenizer):
    """HF-dict batch → right-padded (input_ids, attention_mask, lengths)."""
    import torch

    rows = batch["input_ids"]
    seqs: List[List[int]] = []
    for row in rows:
        ids = row.tolist() if hasattr(row, "tolist") else list(row)
        seqs.append(ids[:MAX_SEQ_LENGTH])
    lengths = [len(s) for s in seqs]
    max_len = max(lengths) if lengths else 0
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id or 0
    input_ids = torch.full((len(seqs), max_len), pad_id, dtype=torch.long)
    mask = torch.zeros((len(seqs), max_len), dtype=torch.long)
    for i, s in enumerate(seqs):
        input_ids[i, :len(s)] = torch.tensor(s, dtype=torch.long)
        mask[i, :len(s)] = 1
    return input_ids, mask, lengths


def _encode_layer(sae, acts):
    """acts [n_tokens, d_model] fp32 → z [n_tokens, d_sae] (no grad).

    MIS-E2E-083: this called `sae.encode` bare, feeding raw activations to a
    dictionary trained on normalized ones. Every circuit mined from a capture
    inherited the wrong basis.
    """
    from ..ml.sparse_autoencoder import encode_with_training_normalization

    return encode_with_training_normalization(sae, acts)


#: What `finalize()` costs on top of the buffer itself. It concatenates every
#: chunk (a second full copy), argsorts that (an int64 index, 8 bytes a row),
#: and materialises the sorted result (a third copy) — roughly 3.7x the
#: buffered bytes at peak, rounded up for margin. A guard that compared only
#: the buffer would pass and then die inside finalize.
PEAK_MEMORY_MULTIPLIER = 4


def _memory_budget_bytes() -> Optional[int]:
    """Bytes this process may safely hold, or None if it cannot be determined.

    The capture buffers every event in HOST RAM until `finalize()`, so the
    binding constraint is memory, not disk — and nothing was checking it.
    `cap_cda1e1da6a0a` was OOM-killed at 45.6% of a 20-billion-event estimate,
    about 110 GB of rows on a 131 GB node with no cgroup limit set.

    The existing ceiling could not have caught it: it is 5x the run's OWN
    estimate, so an estimate of 20 billion events authorises 100 billion. A
    ceiling expressed as a multiple of a number that is itself the problem
    scales with the problem.

    Prefers the cgroup limit, because that is what the kernel kills on. Falls
    back to MemAvailable. Returns None rather than a guess when neither can be
    read — the caller then declines to refuse, since inventing a budget could
    block a capture that would have run.
    """
    for path in ("/sys/fs/cgroup/memory.max",
                 "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        try:
            raw = Path(path).read_text().strip()
        except OSError:
            continue
        if raw == "max":
            # Unlimited — the node's own memory is the real bound. READABILITY,
            # NOT CONTROL FLOW: `int("max")` raises below and lands in the same
            # place, so no mutation of this line can change an answer. Said out
            # loud so a later reader does not mistake it for the guard.
            break
        try:
            value = int(raw)
        except ValueError:
            continue
        # cgroup v1 reports a sentinel near 2**63 for "unlimited".
        if 0 < value < 2**62:
            return value
        break
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    return None


def _max_rows_that_fit() -> Optional[int]:
    """How many events the budget allows, or None when it cannot be read."""
    budget = _memory_budget_bytes()
    if budget is None:
        return None
    return budget // (EVENT_BYTES * PEAK_MEMORY_MULTIPLIER)


def _suggested_sample_cap(events_est: int, sample_cap: int) -> Optional[int]:
    """The sample_cap that would fit, since events scale linearly with it.

    A refusal that lists knobs makes the reader do arithmetic the service has
    already done. This is the one lever that is reliably linear — raising
    epsilon is not: measured on this corpus, epsilon 0.1 -> 0.9 removed only 80%
    of events (20.1B -> 4.1B, still 183 GB against a 97 GB budget), because
    these features have so little dynamic range that even a 90%-of-max threshold
    admits most of them.
    """
    fits = _max_rows_that_fit()
    if fits is None or events_est <= 0:
        return None
    return max(1, int(sample_cap * fits / events_est))


def _exceeds_memory_budget(rows: int) -> Optional[str]:
    """A message when buffering `rows` would not survive finalize, else None."""
    budget = _memory_budget_bytes()
    if budget is None:
        return None
    needed = rows * EVENT_BYTES * PEAK_MEMORY_MULTIPLIER
    if needed <= budget:
        return None
    return (f"{rows:,} events need ~{needed / 2**30:.0f} GB of RAM to buffer "
            f"and sort, against ~{budget / 2**30:.0f} GB available")


def _buffered_rows(writers) -> int:
    """Rows held in host RAM across every writer, including the sidecar.

    EXTRACTED SO IT CAN BE TESTED. Inline, the only available guard was a test
    grepping `run_capture`'s source for the expression — which kept passing
    against a mutation that left the text intact and multiplied the result by
    zero. A source scrape fails open; this does not.

    Attention rows are the reason this exists. `_BufferedWriter` keeps every
    chunk in memory until `finalize()`, and a default attention request is
    ~131M rows (~1.5 GiB) per layer. The ceiling counted only SAE events, which
    was harmless for exactly as long as the sidecar could never hold anything.
    """
    return sum(
        ev.count + (at.count if at is not None else 0)
        for ev, _en, at in writers.values()
    )


def _event_threshold(z, probe_max, eps: float, floor: float):
    """Per-feature activation threshold — the ONE definition.

    Used by both the probe's estimate and the capture's writer. They had
    separate rules (`z > 0` versus this), which is why the estimate could not
    predict the size of the thing it was estimating.
    """
    import torch

    if eps > 0 and probe_max is not None:
        return torch.clamp(eps * probe_max, min=floor)
    return torch.full((z.shape[-1],), floor, device=z.device)


def _probe(model, tokenizer, dataset, saes, layer_indices, device,
           *, epsilon_by_layer=None, floor_by_layer=None):
    """~PROBE_SAMPLES docs → per-layer per-feature max + event-rate sample.

    TWO PASSES, AND THE SECOND ONE IS THE POINT. The count returned here is what
    the size and duration estimate is built from, and the user confirms a
    multi-hundred-gigabyte run against it — so it has to count the same thing
    the capture will write.

    It did not. The estimate counted `z > 0`, every strictly positive encoder
    output, while the capture writes only `z > clamp(eps * probe_max, floor)`.
    The two are not close: on granite-4.1-8b the unthresholded count came to
    19,719 features per token of 32,768 — 60% density, against the 0.48% the
    SAE recorded during its own training — and the estimate read 20.2 billion
    events / 242 GB.

    The threshold needs the FINAL per-feature max, which is only known once the
    whole probe has been seen, so the count cannot be accumulated in the same
    sweep that builds it. Hence a second sweep. It is PROBE_SAMPLES documents;
    the cost is trivial next to being wrong.
    """
    import torch

    from ..ml.forward_hooks import HookManager, HookType

    probe_max: Dict[int, "torch.Tensor"] = {}
    events = 0
    tokens = 0
    n = min(PROBE_SAMPLES, len(dataset))
    model_type = getattr(model.config, "model_type", "auto")

    def _sweep(count_with_threshold: bool) -> int:
        nonlocal events, tokens
        seen = 0
        with HookManager(model) as hm:
            hm.register_hooks(layer_indices, [HookType.RESIDUAL], model_type)
            for start in range(0, n, 8):
                batch = dataset[start:min(start + 8, n)]
                input_ids, mask, lengths = _pad_batch(batch, tokenizer)
                with torch.no_grad():
                    _ = model(input_ids=input_ids.to(model.device),
                              attention_mask=mask.to(model.device))
                for L in layer_indices:
                    acts = hm.activations[f"layer_{L}_residual"][-1]  # [b, s, h]
                    flat = acts.to(device).float().reshape(-1, acts.shape[-1])
                    with torch.no_grad():
                        z = _encode_layer(saes[L], flat)
                    if count_with_threshold:
                        events += int((z > _event_threshold(
                            z, probe_max.get(L),
                            (epsilon_by_layer or {}).get(L, 0.0),
                            (floor_by_layer or {}).get(L, 0.0),
                        ).unsqueeze(0)).sum())
                    else:
                        fmax = z.max(dim=0).values
                        probe_max[L] = (torch.maximum(probe_max[L], fmax)
                                        if L in probe_max else fmax)
                if count_with_threshold:
                    seen += int(sum(lengths))
                hm.clear_activations()
        return seen

    _sweep(count_with_threshold=False)      # builds probe_max
    tokens = _sweep(count_with_threshold=True)   # counts what will be written
    return probe_max, events, tokens


def _capture_batch(model, saes, layer_indices, writers, input_ids, mask,
                   doc_base, lengths, *, epsilon_by_layer, floor_by_layer,
                   probe_max, attn_cfg):
    import torch

    from ..ml.forward_hooks import HookManager, HookType

    with HookManager(model) as hm:
        model_type = getattr(model.config, "model_type", "auto")
        hm.register_hooks(layer_indices, [HookType.RESIDUAL], model_type)

        # ATTENTION WEIGHTS COME FROM OUR OWN HOOKS, keyed by absolute layer.
        #
        # This used to read `out.attentions[L]` after passing
        # `output_attentions=True`. Two things were wrong with that. In
        # transformers 5 the flag is gone — the decoder no longer builds
        # `all_self_attns`, and the replacement collects module outputs only
        # when they are not None, which under SDPA they always are — so
        # `out.attentions` was `()`, passed the `is not None` guard, and
        # `()[34]` raised `IndexError: tuple index out of range`. And the index
        # itself was wrong in principle: that tuple holds one entry per
        # attention module that RAN, so on a hybrid like LFM2 (attention on 6
        # of 16 layers) `[L]` reads a different layer's weights, or overruns.
        #
        # A hook keyed `layer_{L}_attention_weights` cannot have either bug.
        attn_layers = sorted(attn_cfg.get("layers") or []) if attn_cfg else []
        if attn_layers:
            before = len(hm.hook_names)
            hm.register_hooks(attn_layers, [HookType.ATTENTION_WEIGHTS], model_type)
            registered = len(hm.hook_names) - before
            # `register_hooks` only raises when NOTHING is registered, and the
            # residual hooks above already made the list non-empty — so without
            # this check a total failure to attach is silent.
            if registered != len(attn_layers):
                raise CaptureConfigError(
                    f"attention hooks attached to {registered} of "
                    f"{len(attn_layers)} requested layers {attn_layers}; the "
                    "others have no attention module"
                )

        with torch.no_grad():
            # The return value is not read: residual activations and attention
            # weights both arrive through hooks. The forward is run for its
            # side effects.
            model(input_ids=input_ids, attention_mask=mask)
        for L in layer_indices:
            acts = hm.activations[f"layer_{L}_residual"][-1]  # [b, s, h] cpu
            ev_w, en_w, at_w = writers[L]
            b, s, h = acts.shape
            flat = acts.to(input_ids.device).float().reshape(-1, h)
            with torch.no_grad():
                z = _encode_layer(saes[L], flat)          # [b*s, d_sae]
                recon = saes[L].decode(z)
                err = (flat - recon).norm(dim=-1)          # [b*s]
            # The SAME rule the probe's estimate uses — see `_event_threshold`.
            thresh = _event_threshold(z, probe_max.get(L),
                                      epsilon_by_layer[L], floor_by_layer[L])
            hits = (z > thresh.unsqueeze(0)).nonzero(as_tuple=False)  # [n, 2]
            if len(hits):
                vals = z[hits[:, 0], hits[:, 1]].cpu().numpy()
                tok_flat = hits[:, 0].cpu().numpy()
                feats = hits[:, 1].cpu().numpy()
                docs_rel = tok_flat // s
                poss = tok_flat % s
                # drop padding positions
                keep = poss < np.array(lengths)[docs_rel]
                ev_w.append((docs_rel[keep] + doc_base).astype(np.uint32),
                            poss[keep], feats[keep], vals[keep])
            # errnorm: every REAL token
            for i, L_i in enumerate(lengths):
                row = err[i * s:i * s + L_i].cpu().numpy()
                en_w.append(np.full(L_i, doc_base + i, dtype=np.uint32),
                            np.arange(L_i), row)
            # attention sidecar
            if at_w is not None and attn_cfg:
                captured = hm.activations.get(f"layer_{L}_attention_weights")
                if captured:
                    _append_attention(at_w, captured[-1], attn_cfg,
                                      doc_base, lengths)
                    # Released immediately: [b, heads, q, k] is ~128 MiB per
                    # layer at seq 512, and holding every requested layer's
                    # weights to the end of the loop multiplies that.
                    captured.clear()
        hm.clear_activations()


def _append_attention(at_w, attn, cfg, doc_base, lengths):
    """attn [b, heads, q, k] → top-k keys per (head, query)."""
    import torch

    top_k = int(cfg.get("top_k", 4))
    heads = cfg.get("heads")  # list | None = all
    b, n_heads, q_len, _ = attn.shape
    head_ids = list(heads) if heads else list(range(n_heads))

    # BOUNDS-CHECKED. `heads` arrives from a user-supplied config and was
    # indexed straight into the tensor; an out-of-range head gave an opaque
    # torch error mid-capture.
    bad = [h for h in head_ids if not 0 <= int(h) < n_heads]
    if bad:
        raise CaptureConfigError(
            f"attention heads {bad} are out of range for a model with "
            f"{n_heads} heads")

    for i, L_i in enumerate(lengths):
        for hd in head_ids:
            probs = attn[i, hd, :L_i, :L_i]
            k = min(top_k, probs.shape[-1])
            mass, keys = torch.topk(probs, k, dim=-1)
            q_idx = torch.arange(L_i, device=keys.device).unsqueeze(-1).expand_as(keys)

            # DROP FUTURE KEYS AND EMPTY MASS. topk runs over the full L_i x
            # L_i square, so for a query at position q < k it necessarily
            # returns keys the mask forbids — positions carrying exactly 0.0 —
            # and those were written as though they were attention edges. A
            # zero-mass edge is not a weak edge; it is a position the model
            # could not see.
            keep = (keys <= q_idx) & (mass > 0)
            if not bool(keep.any()):
                continue
            q_sel = q_idx[keep]
            at_w.append(
                np.full(q_sel.numel(), doc_base + i, dtype=np.uint32),
                q_sel.cpu().numpy(),
                np.full(q_sel.numel(), hd, dtype=np.uint16),
                keys[keep].cpu().numpy(),
                mass[keep].float().cpu().numpy())


def _write_manifest_atomic(path: Path, manifest: Dict[str, Any]) -> None:
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False,
                                     suffix=".tmp") as f:
        json.dump(manifest, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
        tmp = f.name
    os.replace(tmp, path)
