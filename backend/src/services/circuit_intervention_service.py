"""
Circuit intervention/validation service (Feature 017, IDL-34 — A.5 normative).

The GPU orchestrator for edge validation:
  for each edge (u→d) in top-K of the chosen ordering:
    prompts = windows around u's STRONGEST firings (016 store, SAME tokenization)
    clean pass → a_d(t);  intervened pass (suppress u) → a_d(t)
    Δ_p = mean_t[a_d clean − a_d intervened] over tokens where clean F_u(t)=1
    ES  = mean_p(Δ_p) / σ_d          # σ_d from the SAME capture store
  null = shuffled NON-edge pairs (random support-matched u' in u's layer, d fixed)
  verdict: rung-2 iff |ES| > null percentile AND sign-consistent ≥ frac (math module)
  batch: survival per ordering → uplift = survival(attr) − survival(coact)
  persist: an edge_batch manifest + write rung-2 results onto the discovery-run
           candidates AND, for a promoted circuit, its edges VIA CircuitService
           (never raw JSONB — 018 R2-A5) with the optimistic-concurrency version.

Never re-decodes (the suppression hook subtracts from the residual). σ_d comes
from the capture store the candidates came from — a different corpus's σ would
silently rescale ES.
"""

import logging
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from ..core.config import settings
from ..models.circuit_runs import CircuitCaptureRun, CircuitDiscoveryRun
from . import circuit_validation_math as vmath
from .circuit_capture_store import EventReader
from .circuit_intervention_hooks import make_suppression_hook

logger = logging.getLogger(__name__)

DEFAULT_K = 20
DEFAULT_PROMPTS_PER_EDGE = 8
DEFAULT_NULL_SAMPLES = 20
WINDOW = 48  # tokens around a firing


class InterventionConfigError(ValueError):
    """Invalid validation scope/config — surfaces as a 422."""


def sigma_d_from_store(reader: EventReader, feature_idx: int) -> float:
    """Downstream feature's activation SD from the SAME store (A.5: never a
    fresh corpus). Captured events are above-threshold, so this is the SD of
    the firing distribution — the scale ES is expressed in."""
    ev = reader.feature_events(feature_idx)
    acts = np.asarray(ev["act"], dtype=np.float64)
    if len(acts) < 2:
        return 1.0  # avoid div-by-zero; a lone firing can't scale ES
    sd = float(acts.std())
    return sd if sd > 0 else 1.0


class CircuitInterventionService:
    @staticmethod
    def create_scope(config: Dict[str, Any]) -> Dict[str, Any]:
        ordering = config.get("ordering", "coact")
        if ordering not in ("coact", "attr"):
            raise InterventionConfigError(
                f"ordering must be coact|attr, got {ordering!r}")
        k = int(config.get("k", DEFAULT_K))
        if k < 1:
            raise InterventionConfigError("k must be ≥ 1")
        return {
            "ordering": ordering, "k": k,
            "prompts_per_edge": int(config.get("prompts_per_edge",
                                               DEFAULT_PROMPTS_PER_EDGE)),
            "null_samples": int(config.get("null_samples", DEFAULT_NULL_SAMPLES)),
            "percentile": float(config.get("percentile",
                                           vmath.DEFAULT_NULL_PERCENTILE)),
            "sign_frac": float(config.get("sign_frac", vmath.DEFAULT_SIGN_FRAC)),
            "baseline": config.get("baseline", "zero"),  # zero | corpus_mean
            "seed": int(config.get("seed", 0)),
        }

    @staticmethod
    def top_k_edges(candidates: List[Dict[str, Any]], ordering: str,
                    k: int) -> List[Dict[str, Any]]:
        rank_key = "attr_rank" if ordering == "attr" else "coact_rank"

        def _rank(c):
            r = (c.get("orderings") or {}).get(rank_key)
            return r if r is not None else 10**9
        ordered = sorted(candidates, key=_rank)
        # attr ordering only meaningful once attribution ran
        return ordered[:k]

    # ── worker body (GPU) ────────────────────────────────────────────────

    @staticmethod
    def run(db, run_id: str, scope: Dict[str, Any], *,
            cancel_check: Optional[Callable] = None,
            progress_cb: Optional[Callable] = None) -> Dict[str, Any]:
        import torch
        from datasets import load_from_disk

        from ..ml.layer_discovery import (discover_transformer_structure,
                                          get_hookable_module)
        from ..ml.model_loader import load_model_from_hf
        from ..models.dataset_tokenization import DatasetTokenization
        from ..models.external_sae import ExternalSAE
        from ..models.model import Model, QuantizationFormat
        from .circuit_capture_service import _load_sae_sync, _pad_batch
        from .extraction_service import cleanup_gpu_memory
        from .steering_service import resolve_decoder_weight

        run = db.query(CircuitDiscoveryRun).filter(
            CircuitDiscoveryRun.id == run_id).first()
        if run is None:
            raise ValueError(f"Discovery run {run_id} not found")
        capture = db.query(CircuitCaptureRun).filter(
            CircuitCaptureRun.id == run.capture_run_id).first()
        if capture is None or not capture.store_path:
            raise InterventionConfigError("Capture store missing")
        manifest = capture.manifest or {}
        candidates = list(run.candidates or [])
        edges = CircuitInterventionService.top_k_edges(
            candidates, scope["ordering"], scope["k"])
        if not edges:
            raise InterventionConfigError("No candidates to validate")

        store_dir = settings.resolve_data_path(capture.store_path)
        readers = {e["layer"]: EventReader(store_dir, e["layer"])
                   for e in manifest.get("layers", [])}
        sae_by_layer = {e["layer"]: e["sae_id"]
                        for e in manifest.get("layers", [])}
        tokenization = db.query(DatasetTokenization).filter(
            DatasetTokenization.id == manifest["corpus"]["tokenization_id"]).first()
        dataset = load_from_disk(
            str(settings.resolve_data_path(tokenization.tokenized_path)))

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_record = db.query(Model).filter(
            Model.id == manifest.get("model_id")).first()
        run.validation_status = "running"
        run.validation_progress = 0.0
        db.commit()

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
            for entry in manifest.get("layers", []):
                L = entry["layer"]
                sae_rec = db.query(ExternalSAE).filter(
                    ExternalSAE.id == entry["sae_id"]).first()
                saes[L] = _load_sae_sync(sae_rec, device)

            edge_results = []
            for ei, cand in enumerate(edges):
                if cancel_check is not None and cancel_check():
                    run.validation_status = "cancelled"
                    db.commit()
                    return {"status": "cancelled"}
                up, down = cand["up"], cand["down"]
                if "feature_idx" not in up or "feature_idx" not in down:
                    continue  # cluster candidates: expanded elsewhere (v1 feature edges)
                result = CircuitInterventionService._validate_edge(
                    model, structure, get_hookable_module, saes, readers,
                    dataset, tokenizer, up, down, scope, sae_by_layer, device,
                    resolve_decoder_weight)
                edge_results.append(result)
                run.validation_progress = (ei + 1) / len(edges) * 100.0
                db.commit()
                if progress_cb is not None:
                    progress_cb(run.validation_progress)

            # survival + uplift (only meaningful with BOTH orderings — record
            # this ordering's survival; uplift filled when both tiers exist)
            survived = [r for r in edge_results if r["verdict"]["passed"]]
            survival = vmath.survival_rate(
                [r["verdict"]["passed"] for r in edge_results])

            payload = {
                "intervention": {"kind": "directional_suppression",
                                 "baseline": scope["baseline"]},
                "config": scope,
                "seeds": [scope["seed"]],
                "ordering": scope["ordering"], "k": scope["k"],
                "edges": edge_results,
                "survival": survival,
                "null_summary": {"samples": scope["null_samples"],
                                 "percentile": scope["percentile"],
                                 "kind": "shuffled_non_edge_support_matched"},
            }
            reproduce_of = scope.get("reproduce_of")
            if reproduce_of:
                # Reproduction path (Task 2.2): compare this re-execution against
                # the original manifest and persist a `reproduction` manifest
                # with per-edge deltas + a within-tolerance verdict — the test
                # that a rung-2 claim is reproducible, not a one-off.
                manifest_row = CircuitInterventionService._persist_reproduction(
                    db, run_id, reproduce_of, payload)
            else:
                manifest_row = CircuitInterventionService._persist_manifest(
                    db, run_id, payload)
                # write rung-2 verdicts back onto the discovery candidates
                # (a reproduction never re-writes the candidates — it only
                # confirms the original).
                CircuitInterventionService._write_back(
                    run, edge_results, scope["ordering"],
                    manifest_row_id=manifest_row)
                # Commit the run's own results (candidates + manifest) FIRST so
                # the promoted-circuit propagation below is a separate, best-
                # effort transaction — a hiccup there must NOT lose the
                # validation or mark the run failed (R2 pre-empt).
                db.commit()
                # Propagate rung-2 ES onto any PROMOTED circuit built from this
                # discovery run (R1 A1/#2) — what 015 reads for hazard
                # quantification. Best-effort: its own commit; failure logged.
                try:
                    CircuitInterventionService._write_promoted_circuit_edges(
                        db, run_id, edge_results, manifest_row)
                except Exception:
                    logger.exception(
                        "Validation %s: propagating rung-2 to promoted circuits "
                        "failed (validation itself succeeded + persisted)", run_id)
                    db.rollback()
                run = db.query(CircuitDiscoveryRun).filter(
                    CircuitDiscoveryRun.id == run_id).first()

            # Last-writer race: a cancel that landed during the pass must not be
            # clobbered by 'completed' (R1 #9).
            db.refresh(run)
            if run.validation_status == "cancelled":
                return {"status": "cancelled"}

            run.validation_status = "completed"
            run.validation_progress = 100.0
            if not reproduce_of:
                # Store survival PER ORDERING so a second-ordering run doesn't
                # overwrite the first, and compute UPLIFT when both exist (P2 —
                # 016's whole point: did attribution re-ranking earn its keep?).
                prev = dict((run.report or {}).get("validation") or {})
                by_ordering = dict(prev.get("by_ordering") or {})
                by_ordering[scope["ordering"]] = {
                    "survival": survival, "passed": len(survived),
                    "k": scope["k"], "manifest_id": manifest_row}
                uplift = vmath.uplift(
                    (by_ordering.get("attr") or {}).get("survival"),
                    (by_ordering.get("coact") or {}).get("survival"))
                run.report = {**(run.report or {}), "validation": {
                    "ordering": scope["ordering"], "k": scope["k"],
                    "survival": survival, "passed": len(survived),
                    "manifest_id": manifest_row,
                    "by_ordering": by_ordering, "uplift": uplift,
                    "wall_clock_seconds": round(time.monotonic() - t0, 1)}}
            db.commit()
            return {"status": "completed", "validated": len(edge_results),
                    "passed": len(survived), "survival": survival,
                    "manifest_id": manifest_row}
        finally:
            cleanup_gpu_memory(
                [m for m in [model, *saes.values()] if m is not None],
                context=f"circuit_validation:{run_id}")

    # ── per-edge (GPU) ───────────────────────────────────────────────────

    @staticmethod
    def _validate_edge(model, structure, get_hookable_module, saes, readers,
                       dataset, tokenizer, up, down, scope, sae_by_layer,
                       device, resolve_decoder_weight) -> Dict[str, Any]:
        import torch

        from .circuit_capture_service import _pad_batch

        up_L, up_i = up["layer"], up["feature_idx"]
        down_L, down_i = down["layer"], down["feature_idx"]
        up_reader = readers[up_L]
        down_reader = readers[down_L]
        sigma_d = sigma_d_from_store(down_reader, down_i)

        # prompt windows around u's strongest firings
        ev = up_reader.feature_events(up_i)
        if len(ev) == 0:
            return CircuitInterventionService._empty_result(up, down, "no firings")
        order = np.argsort(ev["act"])[::-1]
        doc_ids = list(dict.fromkeys(int(ev["doc_id"][j]) for j in order))
        doc_ids = doc_ids[:scope["prompts_per_edge"]]

        up_module = get_hookable_module(structure.layers_module[up_L],
                                        "residual", structure)
        # Capture the DOWNSTREAM activation at the SAME residual submodule the
        # SAE was trained on (R2 B-2) — the whole-layer output differs from the
        # residual-norm output by the MLP+residual add, so encoding it fed the
        # SAE an input it was never fit to and corrupted every ES.
        down_module = get_hookable_module(
            structure.layers_module[down_L], "residual", structure)
        W_dec_up = resolve_decoder_weight(saes[up_L])

        # a_base
        a_base = 0.0
        if scope["baseline"] == "corpus_mean":
            a_base = float(np.asarray(ev["act"], dtype=np.float64).mean())

        deltas = []
        for doc_id in doc_ids:
            if doc_id >= len(dataset):
                continue
            batch = dataset[doc_id:doc_id + 1]
            input_ids, mask, lengths = _pad_batch(batch, tokenizer)
            input_ids = input_ids.to(model.device)
            mask = mask.to(model.device)
            # clean-fire positions for u: where u fired in this doc
            fire_pos = set(int(ev["token_pos"][j]) for j in range(len(ev))
                           if int(ev["doc_id"][j]) == doc_id)
            if not fire_pos:
                continue
            a_d_clean = CircuitInterventionService._downstream_acts(
                model, down_module, saes[down_L], down_i, input_ids, mask)
            # intervened: suppress u at u's layer
            enc = CircuitInterventionService._encoder_for(saes[up_L], up_i)
            handle = up_module.register_forward_hook(
                make_suppression_hook(W_dec_up, up_i, a_base=a_base,
                                      encode_fn=enc))
            try:
                a_d_int = CircuitInterventionService._downstream_acts(
                    model, down_module, saes[down_L], down_i, input_ids, mask)
            finally:
                handle.remove()
            # Δ over clean-fire tokens
            L_i = lengths[0]
            pos = [p for p in fire_pos if p < L_i]
            if not pos:
                continue
            clean = a_d_clean[0, pos].mean().item()
            interv = a_d_int[0, pos].mean().item()
            deltas.append(clean - interv)

        # shuffled-non-edge null: random support-matched u' in up's layer
        null_es = CircuitInterventionService._null_effect_sizes(
            model, structure, get_hookable_module, saes, up_reader, down_reader,
            dataset, tokenizer, up_L, down_L, down_i, doc_ids, sigma_d, scope,
            resolve_decoder_weight, exclude=up_i)

        verdict = vmath.edge_verdict(
            deltas, sigma_d, null_es,
            percentile=scope["percentile"], sign_frac=scope["sign_frac"])
        return {
            "up": up, "down": down,
            "effect_size": round(verdict.effect_size, 5),
            "sign_consistency": round(verdict.sign_consistency, 3),
            "sigma_d": round(sigma_d, 5),
            # thin-support ES is less trustworthy (σ_d from few firings) — record
            # the count so a reviewer can discount it (R2 B-8).
            "n_down_firings": int(len(down_reader.feature_events(down_i))),
            "n_prompts": len(deltas),
            "null_percentile_value": round(verdict.null_percentile_value, 5),
            "verdict": {"passed": verdict.passed, "reason": verdict.reason},
            # rung 2 iff passed; else tested_and_failed history (018 ladder)
            "rung": 2 if verdict.passed else None,
            "tested_and_failed": (not verdict.passed),
        }

    @staticmethod
    def _downstream_acts(model, down_module, sae_d, feature_idx, input_ids, mask):
        """One forward, capture down_module residual, encode, return a_d [b,s]."""
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
        hidden = captured["h"].float()
        # MIS-E2E-083 sibling sweep.
        from ..ml.sparse_autoencoder import encode_with_training_normalization

        z = encode_with_training_normalization(sae_d, hidden)
        if isinstance(z, tuple):
            z = z[0]
        return z[..., feature_idx]

    @staticmethod
    def _encoder_for(sae, feature_idx):
        """encode_fn(hidden)->a_u for the suppression hook (same-pass encode)."""
        def enc(hidden):
            from ..ml.sparse_autoencoder import encode_with_training_normalization

            z = encode_with_training_normalization(sae, hidden.float())
            if isinstance(z, tuple):
                z = z[0]
            return z[..., feature_idx]
        return enc

    @staticmethod
    def _null_effect_sizes(model, structure, get_hookable_module, saes,
                           up_reader, down_reader, dataset, tokenizer, up_L,
                           down_L, down_i, doc_ids, sigma_d, scope,
                           resolve_decoder_weight, exclude) -> List[float]:
        """Support-matched random non-edges: pick u' in up's layer with similar
        support, suppress it, measure Δ on d — the null ES distribution."""
        import torch

        rng = np.random.default_rng(scope["seed"] + 7919)
        # ACTUAL support-matching (R1 #5): the null must be non-edges with
        # SIMILAR support to `exclude` — a random tiny-support u' yields ~0 Δ
        # and deflates the threshold, letting real edges pass too easily. Band
        # to within 2× of the excluded feature's support; widen if too few.
        target_support = len(up_reader.feature_events(exclude))
        all_others = [f for f in up_reader.feature_ids if f != exclude]
        if not all_others:
            return []

        def _supp(f):
            return len(up_reader.feature_events(int(f)))

        band = [f for f in all_others
                if 0.5 * target_support <= _supp(f) <= 2.0 * target_support]
        # widen to nearest-by-support if the band is too small for a real null
        pool = band if len(band) >= scope["null_samples"] else sorted(
            all_others, key=lambda f: abs(_supp(f) - target_support)
        )[:max(scope["null_samples"] * 3, len(band))]
        if not pool:
            return []
        picks = rng.choice(pool, size=min(scope["null_samples"], len(pool)),
                           replace=False)
        up_module = get_hookable_module(structure.layers_module[up_L],
                                        "residual", structure)
        # Capture the DOWNSTREAM activation at the SAME residual submodule the
        # SAE was trained on (R2 B-2) — the whole-layer output differs from the
        # residual-norm output by the MLP+residual add, so encoding it fed the
        # SAE an input it was never fit to and corrupted every ES.
        down_module = get_hookable_module(
            structure.layers_module[down_L], "residual", structure)
        W_dec_up = resolve_decoder_weight(saes[up_L])
        prompts_per_null = max(2, min(scope.get("prompts_per_edge", 8), 4))
        null = []
        for uprime in picks:
            ev = up_reader.feature_events(int(uprime))
            if len(ev) == 0:
                continue
            # Probe u'’s OWN strongest-firing docs (its natural support) — R2
            # B-3: reusing u's top docs almost never contained u'’s firings, so
            # the null systematically under-realized (<10 samples) and every
            # edge failed "underpowered". A null feature must be measured where
            # IT fires, exactly as the real edge is measured where u fires.
            order = np.argsort(ev["act"])[::-1]
            u_docs = list(dict.fromkeys(int(ev["doc_id"][j]) for j in order))
            u_docs = u_docs[:prompts_per_null]
            deltas = []
            for doc_id in u_docs:
                if doc_id >= len(dataset):
                    continue
                batch = dataset[doc_id:doc_id + 1]
                input_ids, mask, lengths = _pad_batch_local(batch, tokenizer)
                input_ids = input_ids.to(model.device)
                mask = mask.to(model.device)
                fire_pos = [int(ev["token_pos"][j]) for j in range(len(ev))
                            if int(ev["doc_id"][j]) == doc_id
                            and int(ev["token_pos"][j]) < lengths[0]]
                if not fire_pos:
                    continue
                clean = CircuitInterventionService._downstream_acts(
                    model, down_module, saes[down_L], down_i, input_ids, mask)
                enc = CircuitInterventionService._encoder_for(saes[up_L], int(uprime))
                handle = up_module.register_forward_hook(
                    make_suppression_hook(W_dec_up, int(uprime), encode_fn=enc))
                try:
                    interv = CircuitInterventionService._downstream_acts(
                        model, down_module, saes[down_L], down_i, input_ids, mask)
                finally:
                    handle.remove()
                deltas.append(clean[0, fire_pos].mean().item()
                              - interv[0, fire_pos].mean().item())
            if deltas:
                null.append(vmath.effect_size(deltas, sigma_d))
        return null

    @staticmethod
    def _empty_result(up, down, reason):
        # Include EVERY key a full result has (R1 #4): the manifest UI does
        # e.null_percentile_value.toFixed(...) and reads sigma_d, so a missing
        # key crashes the whole drawer.
        return {"up": up, "down": down, "effect_size": 0.0,
                "sign_consistency": 0.0, "sigma_d": 0.0,
                "null_percentile_value": 0.0, "n_prompts": 0,
                "verdict": {"passed": False, "reason": reason},
                "rung": None, "tested_and_failed": True}

    @staticmethod
    def _write_promoted_circuit_edges(db, run_id, edge_results, manifest_id):
        """Propagate rung-2 validation onto PROMOTED circuits built from this
        discovery run (R1 A1) — the durable, 015-readable record. Routes
        through the contract (validate + rung recompute + version bump) exactly
        like CircuitService.write_edge_validation, but sync (worker context).
        Never raw JSONB mutation (018 R2-A5)."""
        from ..models.circuit import Circuit
        from ..schemas.circuit_definition import CircuitDefinitionV1

        updates = {}
        for r in edge_results:
            up, down = r["up"], r["down"]
            key = (up.get("layer"), up.get("feature_idx"),
                   down.get("layer"), down.get("feature_idx"))
            if r["rung"] == 2:
                updates[key] = {"rung": 2, "effect_size": r["effect_size"],
                                "validation_manifest_ref": manifest_id}
            elif r["tested_and_failed"]:
                # history, never a demotion (018 ladder)
                updates[key] = {"validation_manifest_ref": manifest_id,
                                "_tested_and_failed": r["verdict"]["reason"]}
        if not updates:
            return
        circuits = db.query(Circuit).filter(
            Circuit.discovery_run_id == run_id,
            Circuit.promoted == True).all()  # noqa: E712
        for circuit in circuits:
            edges = [dict(e) for e in (circuit.edges or [])]
            changed = False
            for e in edges:
                up, down = e.get("up", {}), e.get("down", {})
                k = (up.get("layer"), up.get("feature_idx"),
                     down.get("layer"), down.get("feature_idx"))
                if k in updates:
                    upd = dict(updates[k])
                    failed = upd.pop("_tested_and_failed", None)
                    if failed is not None:
                        hist = list(e.get("tested_and_failed") or [])
                        # Set-like: record rung 2 ONCE (R2 B-4 — re-validating
                        # both orderings appended [2,2,2…] each pass).
                        if 2 not in hist:
                            hist.append(2)
                        e["tested_and_failed"] = hist
                    e.update(upd)
                    changed = True
            if not changed:
                continue
            # round-trip through the contract so validators + rung recompute run
            try:
                defn = CircuitDefinitionV1(
                    name=circuit.name, narrative=circuit.narrative,
                    saes=circuit.saes, members=circuit.members, edges=edges,
                    budget=circuit.budget, faithfulness=circuit.faithfulness)
            except Exception:
                logger.exception(
                    "Validation write-back produced an invalid circuit %s — "
                    "skipping edge propagation", circuit.id)
                continue
            circuit.edges = [e.model_dump(mode="json") for e in defn.edges]
            circuit.rung = int(defn.displayed_rung())
            circuit.version = (circuit.version or 1) + 1
        db.commit()

    @staticmethod
    def _persist_manifest(db, run_id, payload) -> str:
        """Sync-context manifest insert (worker uses a sync session)."""
        from ..models.validation_manifest import ValidationManifest
        from .manifest_service import validate_payload
        validate_payload("edge_batch", payload)
        m = ValidationManifest(kind="edge_batch", payload=payload,
                               discovery_run_id=run_id)
        db.add(m)
        db.commit()
        db.refresh(m)
        return m.id

    @staticmethod
    def _persist_reproduction(db, run_id, original_id, repro_payload) -> str:
        """Persist a `reproduction` manifest: the re-executed edges + a verdict
        comparing per-edge ES against the original (deterministic greedy passes
        ⇒ deltas ~0; tolerance guards fp noise). Task 2.2."""
        from ..models.validation_manifest import ValidationManifest
        from .manifest_service import ManifestService
        original = db.query(ValidationManifest).filter(
            ValidationManifest.id == original_id).first()
        verdict = (ManifestService.reproduction_verdict(
            original.payload, repro_payload) if original is not None
            else {"within_tolerance": None, "reason": "original manifest gone"})
        payload = {
            "reproduce_of": original_id,
            "reproduced_edges": repro_payload.get("edges", []),
            "verdict": verdict,
            # keep the config so the reproduction is itself self-contained
            "config": repro_payload.get("config"),
            "seeds": repro_payload.get("seeds"),
            "intervention": repro_payload.get("intervention"),
        }
        m = ValidationManifest(kind="reproduction", payload=payload,
                               discovery_run_id=run_id,
                               parent_manifest_id=original_id)
        db.add(m)
        db.commit()
        db.refresh(m)
        return m.id

    @staticmethod
    def _write_back(run, edge_results, ordering, manifest_row_id):
        """Write validation onto the discovery-run candidates (rung history +
        validation field). Promoted-circuit edge writes go through
        CircuitService.write_edge_validation at the endpoint/caller (async);
        here we annotate the discovery candidates (sync JSONB is the run's own
        working data, not a circuit's contract-governed edges)."""
        by_key = {}
        for r in edge_results:
            up, down = r["up"], r["down"]
            by_key[(up.get("layer"), up.get("feature_idx"),
                    down.get("layer"), down.get("feature_idx"))] = r
        cands = [dict(c) for c in (run.candidates or [])]
        for c in cands:
            k = (c["up"].get("layer"), c["up"].get("feature_idx"),
                 c["down"].get("layer"), c["down"].get("feature_idx"))
            if k in by_key:
                r = by_key[k]
                c["validation"] = {
                    "ordering": ordering,
                    "effect_size": r["effect_size"],
                    "passed": r["verdict"]["passed"],
                    "manifest_id": manifest_row_id,
                }
                if r["rung"] == 2:
                    c["validated_rung"] = 2
                if r["tested_and_failed"]:
                    hist = c.get("tested_and_failed_history", [])
                    hist.append({"ordering": ordering,
                                 "reason": r["verdict"]["reason"]})
                    c["tested_and_failed_history"] = hist
        run.candidates = cands


def _pad_batch_local(batch, tokenizer):
    from .circuit_capture_service import _pad_batch
    return _pad_batch(batch, tokenizer)
