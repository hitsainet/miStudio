"""
Circuit faithfulness service (Feature 017, IDL-34 — A.7 normative).

necessity   = (B_clean − B_ablate_M)          / (B_clean − B_ablate_all)
sufficiency = (B_ablate_topk_nonmembers − B_ablate_all) / (B_clean − B_ablate_all)

where M = the circuit's members (cluster_ref members expand to features),
B = a behavior metric (v1 = the compare-workflow output-shift, imported — the
metric identity is ALWAYS recorded in the manifest so the open question stays
visible). "Ablate all at layers" is intractable literally → per-layer top-N
proxy (N recorded). Suppression of a member LIST is one hook per layer
subtracting the SUM (the FTID pitfall: N separate hooks reorder float ops).

The math lives in circuit_validation_math (necessity/sufficiency); this
service supplies the four behavior measurements from real forward passes and
persists scores on the circuit record (018) + a faithfulness manifest.
"""

import logging
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from ..core.config import settings
from . import circuit_validation_math as vmath

logger = logging.getLogger(__name__)

DEFAULT_TOPK_NONMEMBERS = 256   # per layer, disclosed
DEFAULT_ABLATE_ALL_N = 1024     # per-layer top-N proxy for "ablate all", recorded


class FaithfulnessConfigError(ValueError):
    """Invalid faithfulness config — surfaces as a 422."""


class FaithfulnessRunError(RuntimeError):
    """A run couldn't be set up (no members, no capture store, …) — 409."""


def expand_members(circuit_members: List[Dict[str, Any]],
                   resolve_cluster) -> Dict[int, List[int]]:
    """circuit members → {layer: [feature_idx]}. cluster_ref members expand to
    their profile's features via `resolve_cluster(profile_id) -> [idx]`."""
    by_layer: Dict[int, List[int]] = {}
    for m in circuit_members:
        layer = m["layer"]
        if m.get("member_kind") == "cluster_ref" or m.get("cluster_profile_id"):
            for idx in resolve_cluster(m.get("cluster_profile_id")):
                by_layer.setdefault(layer, []).append(int(idx))
        else:
            feat = m.get("feature") or {}
            if feat.get("feature_idx") is not None:
                by_layer.setdefault(layer, []).append(int(feat["feature_idx"]))
    return by_layer


def scores_from_behaviors(b_clean: float, b_ablate_m: float,
                          b_ablate_all: float,
                          b_ablate_nonmembers: Optional[float],
                          *, mode: str) -> Dict[str, Any]:
    """Assemble necessity (+ sufficiency when mode='both') from the four
    behavior measurements. sufficiency is marked untested in necessity-only
    mode (never silently omitted)."""
    nec = vmath.necessity(b_clean, b_ablate_m, b_ablate_all)
    out = {"necessity": nec}
    if mode == "both" and b_ablate_nonmembers is not None:
        out["sufficiency"] = vmath.sufficiency(
            b_ablate_nonmembers, b_ablate_all, b_clean)
    else:
        out["sufficiency"] = None
        out["sufficiency_status"] = "untested (necessity-only mode)"
    return out


class CircuitFaithfulnessService:
    @staticmethod
    def create_config(config: Dict[str, Any]) -> Dict[str, Any]:
        mode = config.get("mode", "both")
        if mode not in ("necessity", "both"):
            raise FaithfulnessConfigError("mode must be necessity|both")
        return {
            "mode": mode,
            "k_nonmembers": int(config.get("k_nonmembers", DEFAULT_TOPK_NONMEMBERS)),
            "ablate_all_n": int(config.get("ablate_all_n", DEFAULT_ABLATE_ALL_N)),
            "metric_id": config.get("metric_id", "compare_output_shift/v1"),
            "n_prompts": int(config.get("n_prompts", 16)),
            "seed": int(config.get("seed", 0)),
        }

    @staticmethod
    def build_manifest_payload(circuit_id: str, config: Dict[str, Any],
                               scores: Dict[str, Any],
                               behaviors: Dict[str, float],
                               *, metric_notes: Optional[str] = None,
                               provenance: Optional[Dict[str, Any]] = None
                               ) -> Dict[str, Any]:
        """Self-contained faithfulness manifest (metric identity ALWAYS
        recorded — the open question stays visible)."""
        payload = {
            "intervention": {"kind": "member_sum_suppression"},
            "config": config,
            "seeds": [config["seed"]],
            "circuit_id": circuit_id,
            "metric_id": config["metric_id"],
            # The EXACT behavior metric definition, recorded so a reviewer never
            # has to trust a bare number: B = mean over prompts of the summed
            # activation of the circuit's downstream-most members.
            "metric_definition": (
                "B = mean over prompts of the summed activation of the "
                "circuit's downstream-most members (members at the circuit's "
                "highest layer), averaged over valid (non-pad) token positions. "
                "Suppression subtracts each member's decoder direction from the "
                "residual (never re-decoded). necessity/sufficiency are ratios "
                "of these four behavior measurements (A.7)."),
            "behaviors": behaviors,
            "scores": scores,
            "ablate_all_proxy": {"per_layer_top_n": config["ablate_all_n"],
                                 "note": "literal ablate-all intractable; "
                                         "per-layer top-N proxy (N recorded)"},
        }
        if metric_notes:
            payload["metric_notes"] = metric_notes
        if provenance:
            payload["provenance"] = provenance
        return payload

    # ── worker body (GPU) ────────────────────────────────────────────────

    @staticmethod
    def run(db, circuit_id: str, config: Dict[str, Any], *,
            cancel_check: Optional[Callable] = None,
            progress_cb: Optional[Callable] = None) -> Dict[str, Any]:
        """GPU orchestrator (mirrors CircuitInterventionService.run):

        Loads the circuit, expands its members to {layer: [feature_idx]}, runs
        FOUR behavior passes over prompts drawn from the discovery run's capture
        store (SAME tokenization the circuit was mined on), computes
        necessity/sufficiency via the math module, persists a `faithfulness`
        manifest, and writes {necessity, sufficiency, metric_id, manifest_ref,
        k, n} onto the circuit through the CircuitDefinitionV1 contract (so the
        validators run + the version bumps) — never a raw JSONB mutation.

        Behavior metric (v1, recorded in the manifest): B = mean over prompts of
        the SUMMED activation of the circuit's downstream-most members (the
        "output" the circuit drives). Suppression lowers it; the ratios are
        necessity/sufficiency.
        """
        import torch
        from datasets import load_from_disk

        from ..ml.layer_discovery import (discover_transformer_structure,
                                          get_hookable_module)
        from ..ml.model_loader import load_model_from_hf
        from ..models.circuit import Circuit
        from ..models.circuit_runs import CircuitCaptureRun, CircuitDiscoveryRun
        from ..models.cluster_profile import ClusterProfile
        from ..models.dataset_tokenization import DatasetTokenization
        from ..models.external_sae import ExternalSAE
        from ..models.model import Model, QuantizationFormat
        from .circuit_capture_service import _load_sae_sync, _pad_batch
        from .circuit_capture_store import EventReader
        from .extraction_service import cleanup_gpu_memory
        from .steering_service import resolve_decoder_weight

        cfg = CircuitFaithfulnessService.create_config(config)
        circuit = db.query(Circuit).filter(Circuit.id == circuit_id).first()
        if circuit is None:
            raise FaithfulnessRunError(f"Circuit {circuit_id} not found")
        if not circuit.members:
            raise FaithfulnessRunError("Circuit has no members")
        circuit.faithfulness_status = "running"  # in-flight marker (R2 B-5)
        db.commit()

        # Resolve cluster_ref members via ClusterProfile.members (feature_idx).
        def _resolve_cluster(profile_id):
            prof = db.query(ClusterProfile).filter(
                ClusterProfile.id == profile_id).first()
            if prof is None:
                return []
            return [int(m["feature_idx"]) for m in (prof.members or [])
                    if m.get("feature_idx") is not None]

        by_layer = expand_members(list(circuit.members), _resolve_cluster)
        if not by_layer:
            raise FaithfulnessRunError("Circuit members expand to no features")

        # The capture store the circuit was mined on is the honest prompt source
        # (SAME tokenization; PREFERRED over fresh generations). Reached via the
        # soft discovery_run_id → capture_run_id chain.
        if not circuit.discovery_run_id:
            raise FaithfulnessRunError(
                "v1 faithfulness needs the circuit's discovery capture store "
                "for prompts (circuit has no discovery_run_id)")
        run = db.query(CircuitDiscoveryRun).filter(
            CircuitDiscoveryRun.id == circuit.discovery_run_id).first()
        if run is None:
            raise FaithfulnessRunError(
                "Circuit's discovery run is gone (prunable) — cannot source "
                "prompts for v1 faithfulness")
        capture = db.query(CircuitCaptureRun).filter(
            CircuitCaptureRun.id == run.capture_run_id).first()
        if capture is None or not capture.store_path:
            raise FaithfulnessRunError("Capture store missing")
        manifest = capture.manifest or {}

        store_dir = settings.resolve_data_path(capture.store_path)
        sae_id_by_layer = {e["layer"]: e["sae_id"]
                           for e in manifest.get("layers", [])}
        readers = {e["layer"]: EventReader(store_dir, e["layer"])
                   for e in manifest.get("layers", [])}
        # Every circuit layer needs a capture SAE + reader to measure/suppress.
        circuit_layers = sorted(by_layer.keys())
        missing = [L for L in circuit_layers if L not in sae_id_by_layer]
        if missing:
            raise FaithfulnessRunError(
                f"Circuit layers {missing} were not in the capture store — "
                "cannot measure faithfulness against this store")
        down_layer = circuit_layers[-1]  # downstream-most = highest layer
        down_features = list(dict.fromkeys(by_layer[down_layer]))

        tokenization = db.query(DatasetTokenization).filter(
            DatasetTokenization.id ==
            manifest["corpus"]["tokenization_id"]).first()
        dataset = load_from_disk(
            str(settings.resolve_data_path(tokenization.tokenized_path)))

        # Deterministic prompt selection: the docs where the downstream members
        # fire most (the circuit's "output" is actually exercised), seeded.
        doc_ids = CircuitFaithfulnessService._select_prompts(
            readers[down_layer], down_features, cfg["n_prompts"], cfg["seed"])
        if not doc_ids:
            raise FaithfulnessRunError(
                "No prompts where the circuit's downstream members fire — "
                "nothing to measure faithfulness on")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_record = db.query(Model).filter(
            Model.id == manifest.get("model_id")).first()

        model = None
        saes: Dict[int, Any] = {}
        t0 = time.monotonic()
        try:
            resolved = (settings.resolve_data_path(model_record.file_path)
                        if model_record.file_path else None)
            model, tokenizer, _c, _m = load_model_from_hf(
                repo_id=model_record.repo_id,
                quant_format=QuantizationFormat(model_record.quantization),
                cache_dir=resolved, device_map=device,
                local_files_only=bool(resolved and resolved.exists()))
            model.eval()
            structure = discover_transformer_structure(model)
            for L in circuit_layers:
                sae_rec = db.query(ExternalSAE).filter(
                    ExternalSAE.id == sae_id_by_layer[L]).first()
                saes[L] = _load_sae_sync(sae_rec, device)

            def _prog(frac):
                if progress_cb is not None:
                    progress_cb(round(frac * 100.0, 1))

            # Build the four suppression member-lists per layer.
            members_by_layer = {L: list(dict.fromkeys(by_layer[L]))
                                for L in circuit_layers}
            # top-N per layer (ablate-all proxy) + top-k non-members.
            topn_by_layer = {
                L: CircuitFaithfulnessService._top_features(
                    readers[L], cfg["ablate_all_n"])
                for L in circuit_layers}
            nonmember_by_layer = {
                L: CircuitFaithfulnessService._top_nonmembers(
                    readers[L], set(members_by_layer[L]), cfg["k_nonmembers"])
                for L in circuit_layers} if cfg["mode"] == "both" else {}

            b_kwargs = dict(
                model=model, structure=structure,
                get_hookable_module=get_hookable_module, saes=saes,
                readers=readers, dataset=dataset, tokenizer=tokenizer,
                doc_ids=doc_ids, down_layer=down_layer,
                down_features=down_features, device=device,
                resolve_decoder_weight=resolve_decoder_weight,
                cancel_check=cancel_check, _pad_batch=_pad_batch)

            _prog(0.05)
            b_clean = CircuitFaithfulnessService._behavior(suppress={}, **b_kwargs)
            _prog(0.30)
            b_ablate_m = CircuitFaithfulnessService._behavior(
                suppress=members_by_layer, **b_kwargs)
            _prog(0.55)
            b_ablate_all = CircuitFaithfulnessService._behavior(
                suppress=topn_by_layer, **b_kwargs)
            _prog(0.80)
            b_ablate_nonmembers = None
            if cfg["mode"] == "both":
                b_ablate_nonmembers = CircuitFaithfulnessService._behavior(
                    suppress=nonmember_by_layer, **b_kwargs)
            _prog(0.95)

            scores = scores_from_behaviors(
                b_clean, b_ablate_m, b_ablate_all, b_ablate_nonmembers,
                mode=cfg["mode"])
            behaviors = {
                "b_clean": round(b_clean, 6),
                "b_ablate_m": round(b_ablate_m, 6),
                "b_ablate_all": round(b_ablate_all, 6),
                "b_ablate_nonmembers": (round(b_ablate_nonmembers, 6)
                                        if b_ablate_nonmembers is not None
                                        else None),
            }
            provenance = {
                "capture_run_id": capture.id,
                "discovery_run_id": run.id,
                "down_layer": down_layer,
                "n_down_members": len(down_features),
                "n_prompts_used": len(doc_ids),
                "circuit_version_at_run": circuit.version,
            }
            payload = CircuitFaithfulnessService.build_manifest_payload(
                circuit_id, cfg, scores, behaviors, provenance=provenance)

            manifest_id = CircuitFaithfulnessService._persist_manifest(
                db, circuit_id, payload)
            CircuitFaithfulnessService._write_circuit_faithfulness(
                db, circuit_id, scores, cfg, manifest_id)
            _prog(1.0)

            result = {
                "status": "completed", "circuit_id": circuit_id,
                "necessity": scores.get("necessity"),
                "sufficiency": scores.get("sufficiency"),
                "metric_id": cfg["metric_id"], "manifest_id": manifest_id,
                "behaviors": behaviors,
                "wall_clock_seconds": round(time.monotonic() - t0, 1),
            }
            return result
        finally:
            cleanup_gpu_memory(
                [m for m in [model, *saes.values()] if m is not None],
                context=f"circuit_faithfulness:{circuit_id}")

    # ── prompt / feature selection (from the SAME capture store) ─────────

    @staticmethod
    def _select_prompts(down_reader, down_features: List[int], n_prompts: int,
                        seed: int) -> List[int]:
        """Docs where the circuit's downstream members fire STRONGEST — the
        prompts that actually exercise the circuit's output. Deterministic:
        aggregate by doc, sort by summed activation, take the top n (seed only
        breaks exact ties)."""
        by_doc: Dict[int, float] = {}
        for f in down_features:
            ev = down_reader.feature_events(int(f))
            if len(ev) == 0:
                continue
            # Vectorized per-doc sum (R2 B-7) — no per-event Python loop.
            docs = np.asarray(ev["doc_id"], dtype=np.int64)
            acts = np.asarray(ev["act"], dtype=np.float64)
            sums = np.bincount(docs, weights=acts)
            for d in np.nonzero(sums)[0]:
                by_doc[int(d)] = by_doc.get(int(d), 0.0) + float(sums[d])
        if not by_doc:
            return []
        rng = np.random.default_rng(seed)
        # stable ordering: (−summed_act, tiny jitter for tie-break) then doc_id
        items = list(by_doc.items())
        jitter = rng.random(len(items)) * 1e-9
        order = sorted(range(len(items)),
                       key=lambda i: (-items[i][1] - jitter[i], items[i][0]))
        return [items[i][0] for i in order[:n_prompts]]

    @staticmethod
    def _top_features(reader, n: int) -> List[int]:
        """Top-N features at a layer by total captured activation mass — the
        per-layer 'ablate all' proxy (N recorded in the manifest). ONE
        vectorized pass over the events array (R2 B-6)."""
        mass = reader.feature_activation_mass()
        return [f for f, _s in sorted(mass.items(), key=lambda kv: kv[1],
                                      reverse=True)[:n]]

    @staticmethod
    def _top_nonmembers(reader, members: set, k: int) -> List[int]:
        """Top-k NON-member features (sufficiency probe). Vectorized (R2 B-7)."""
        mass = reader.feature_activation_mass()
        ranked = sorted(((f, s) for f, s in mass.items() if f not in members),
                        key=lambda kv: kv[1], reverse=True)
        return [f for f, _s in ranked[:k]]

    # ── behavior measurement (one forward per prompt, optional suppression) ─

    @staticmethod
    def _behavior(*, suppress: Dict[int, List[int]], model, structure,
                  get_hookable_module, saes, readers, dataset, tokenizer,
                  doc_ids, down_layer, down_features, device,
                  resolve_decoder_weight, cancel_check, _pad_batch) -> float:
        """B = mean over prompts of the summed downstream-member activation
        (averaged over valid tokens). `suppress` = {layer: [feature_idx]} to
        SUM-suppress at each layer via ONE hook per layer (per-layer SUM keeps
        float ordering stable — the FTID pitfall). Never re-decodes."""
        import torch

        from .circuit_intervention_hooks import suppress_feature_list

        # Read the downstream activation at the SAME residual submodule the SAE
        # was trained on (R2 B-2) — not the whole-layer output.
        down_module = get_hookable_module(
            structure.layers_module[down_layer], "residual", structure)
        # pre-resolve decoder weights + per-layer suppression hook modules
        hook_layers = {L: get_hookable_module(structure.layers_module[L],
                                              "residual", structure)
                       for L in suppress if suppress.get(L)}
        w_dec = {L: resolve_decoder_weight(saes[L]) for L in hook_layers}

        per_prompt = []
        for doc_id in doc_ids:
            if cancel_check is not None and cancel_check():
                # bubble a cancel signal up as an exception the task catches
                raise _FaithfulnessCancelled()
            if doc_id >= len(dataset):
                continue
            batch = dataset[doc_id:doc_id + 1]
            input_ids, mask, lengths = _pad_batch(batch, tokenizer)
            input_ids = input_ids.to(model.device)
            mask = mask.to(model.device)

            handles = []
            for L, feats in suppress.items():
                if not feats:
                    continue
                enc = CircuitFaithfulnessService._multi_encoder(saes[L], feats)
                handles.append(hook_layers[L].register_forward_hook(
                    CircuitFaithfulnessService._make_sum_hook(
                        w_dec[L], feats, enc, suppress_feature_list)))
            try:
                a_down = CircuitFaithfulnessService._downstream_sum(
                    model, down_module, saes[down_layer], down_features,
                    input_ids, mask)
            finally:
                for h in handles:
                    h.remove()
            L_i = lengths[0]
            if L_i <= 0:
                continue
            per_prompt.append(float(a_down[0, :L_i].mean().item()))
        if not per_prompt:
            return 0.0
        return float(np.mean(per_prompt))

    @staticmethod
    def _downstream_sum(model, down_module, sae_d, feature_idxs, input_ids, mask):
        """One forward, capture down_module residual, encode, return the SUMMED
        activation of the downstream members [batch, seq]."""
        import torch

        captured = {}

        def cap(mod, inp, out):
            captured["h"] = (out[0] if isinstance(out, tuple) else out).detach()

        h = down_module.register_forward_hook(cap)
        try:
            with torch.no_grad():
                model(input_ids=input_ids, attention_mask=mask)
        finally:
            h.remove()
        # MIS-E2E-083 sibling sweep.
        from ..ml.sparse_autoencoder import encode_with_training_normalization

        z = encode_with_training_normalization(sae_d, captured["h"].float())
        if isinstance(z, tuple):
            z = z[0]
        idx = torch.as_tensor(list(feature_idxs), device=z.device, dtype=torch.long)
        return z.index_select(-1, idx).sum(dim=-1)  # [batch, seq]

    @staticmethod
    def _multi_encoder(sae, feature_idxs):
        """encode_fn(hidden) -> a_us [len(feats)] list of [batch,seq] tensors,
        for the per-layer SUM suppression hook."""
        def enc(hidden):
            import torch
            from ..ml.sparse_autoencoder import encode_with_training_normalization

            z = encode_with_training_normalization(sae, hidden.float())
            if isinstance(z, tuple):
                z = z[0]
            return [z[..., int(fi)] for fi in feature_idxs]
        return enc

    @staticmethod
    def _make_sum_hook(decoder_weight, feature_idxs, encode_fn,
                       suppress_feature_list):
        """One forward hook that SUM-suppresses a member list at a layer (never
        re-decodes — subtracts from the residual we were handed)."""
        def hook(module, inp, output):
            import torch
            is_tuple = isinstance(output, tuple)
            hidden = output[0] if is_tuple else output
            if hidden.dim() != 3:
                return output
            with torch.no_grad():
                a_us = encode_fn(hidden)
                suppress_feature_list(hidden, decoder_weight,
                                      list(feature_idxs), a_us)
            return output
        return hook

    # ── persistence ─────────────────────────────────────────────────────

    @staticmethod
    def _persist_manifest(db, circuit_id: str, payload: Dict[str, Any]) -> str:
        """Sync-context faithfulness manifest insert (worker uses a sync
        session), validated for self-containment like the intervention path."""
        from ..models.validation_manifest import ValidationManifest
        from .manifest_service import validate_payload
        validate_payload("faithfulness", payload)
        m = ValidationManifest(kind="faithfulness", payload=payload,
                               circuit_id=circuit_id)
        db.add(m)
        db.commit()
        db.refresh(m)
        return m.id

    @staticmethod
    def _write_circuit_faithfulness(db, circuit_id: str, scores: Dict[str, Any],
                                    cfg: Dict[str, Any], manifest_id: str) -> None:
        """Write {necessity, sufficiency, metric_id, manifest_ref, k} onto the
        circuit through the CircuitDefinitionV1 contract (validators run +
        version bump) — never a raw JSONB mutation (018 R2-A5), mirroring
        _write_promoted_circuit_edges in the intervention service."""
        from ..models.circuit import Circuit
        from ..schemas.circuit_definition import CircuitDefinitionV1

        circuit = db.query(Circuit).filter(Circuit.id == circuit_id).first()
        if circuit is None:
            return
        faithfulness = {
            "necessity": scores.get("necessity"),
            "sufficiency": scores.get("sufficiency"),
            "sufficiency_k": cfg["k_nonmembers"] if cfg["mode"] == "both" else None,
            "metric_id": cfg["metric_id"],
            "manifest_ref": manifest_id,
        }
        try:
            defn = CircuitDefinitionV1(
                name=circuit.name, narrative=circuit.narrative,
                saes=circuit.saes, members=circuit.members, edges=circuit.edges,
                budget=circuit.budget, faithfulness=faithfulness)
        except Exception:
            logger.exception(
                "Faithfulness write produced an invalid circuit %s — "
                "skipping faithfulness write", circuit_id)
            return
        circuit.faithfulness = defn.faithfulness.model_dump(mode="json")
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
            _select(Circuit.faithfulness_status).where(Circuit.id == circuit.id)
        ).scalar()
        if _fresh is None:
            # The circuit was deleted mid-run. Writing "completed"
            # here flushes an UPDATE matching zero rows, which is a
            # StaleDataError out of a clean finish — the same shape
            # as the ObjectDeletedError removed from the extraction
            # path. There is nothing left to record to.
            logger.info("Circuit %s was deleted mid-run", circuit.id)
            return
        if _is_cancelled("circuit_faithfulness", _fresh):
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
        circuit.faithfulness_status = "completed"  # clears the in-flight marker (R2 B-5)
        db.commit()


class _FaithfulnessCancelled(RuntimeError):
    """Internal: a cancel_check() fired mid-run — the task turns this into a
    cancelled status without a failure."""
