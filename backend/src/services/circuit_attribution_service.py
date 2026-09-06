"""
Tier-2 gradient attribution service (Feature 016, BR-022 — IDL-36; A.6).

Pass-through construction (per prompt, captured layers L):
    f_L = SAE_L.encode(x_L)          # in-graph — the cross-layer gradient path
    x̂_L = SAE_L.decode(f_L)
    ε_L = (x_L − x̂_L).detach()       # STOP-GRADIENT through the SAE error
    layer output := x̂_L + ε_L        # numerically ≡ x_L (identity-pinned)
SAE parameters are frozen. Downstream unit d at layer Lj:
    m = mean_t a_d(t)  →  ONE backward per (prompt, d)
    attr(u→d) += Σ_t ∂m/∂a_u(t) · a_u(t)    for every upstream u in d's group
Batched by DOWNSTREAM unit — cost O(prompts × distinct downstreams), never
per-candidate (the naive trap that makes Tier-2 unaffordable).

Outputs per candidate: {score, sign_consistency, method}. Rung-1 gate
(attribution_supported): sign agrees with the mined association AND |attr|
≥ the run's percentile floor. BOTH orderings persist so 017 can measure
survival-rate uplift. Envelope (wall-clock, peak VRAM) recorded in report.
"""

import logging
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from ..core.config import settings
from ..models.circuit_runs import CircuitCaptureRun, CircuitDiscoveryRun

logger = logging.getLogger(__name__)

DEFAULT_PROMPT_LIMIT = 32
RUNG1_MAGNITUDE_PERCENTILE = 50.0  # |attr| floor within the run


class PassThroughState:
    """Holds per-layer SAE codes (in-graph) for one forward pass."""

    def __init__(self):
        self.codes: Dict[int, Any] = {}  # layer → f tensor [b, s, d_sae]


def make_passthrough_hook(layer: int, sae, state: PassThroughState):
    """Forward hook: x → x̂ + stopgrad(ε), recording the in-graph code."""
    import torch

    def hook(module, inputs, output):
        x = output[0] if isinstance(output, tuple) else output
        xf = x.float()
        # The code must be a graph node we can read gradients from. In the real
        # model, x carries grad from the frozen-param forward; in isolation
        # (toy tests) it may be a non-grad leaf, so make the code require grad.
        # MIS-E2E-083: normalize the way training did. `normalize` is a scalar
        # rescale, so the cross-layer gradient path is preserved — and now runs
        # through the same transform the dictionary was fitted under.
        from ..ml.sparse_autoencoder import encode_with_training_normalization

        f = encode_with_training_normalization(
            sae, xf if xf.requires_grad else xf.detach().requires_grad_(True)
        )
        if isinstance(f, tuple):
            f = f[0]
        f.retain_grad()
        state.codes[layer] = f
        x_hat = sae.decode(f)
        eps = (x.float() - x_hat).detach()
        patched = (x_hat + eps).to(x.dtype)
        if isinstance(output, tuple):
            return (patched,) + output[1:]
        return patched

    return hook


def attribute_prompt(state: PassThroughState,
                     groups: Dict[Tuple[int, int], List[Tuple[int, int]]],
                     ) -> Dict[Tuple, float]:
    """One prompt's contribution: for each downstream (layer, feature) group,
    ONE backward from m = mean_t a_d(t); accumulate Σ_t g_u(t)·a_u(t) per
    upstream. Returns {(up_layer, up_idx, down_layer, down_idx): attr}."""
    import torch

    scores: Dict[Tuple, float] = {}
    down_items = list(groups.items())
    for gi, ((down_layer, down_idx), ups) in enumerate(down_items):
        f_down = state.codes.get(down_layer)
        if f_down is None:
            continue
        m = f_down[..., down_idx].mean()
        # zero stale grads on upstream codes
        for up_layer, _ in ups:
            f_up = state.codes.get(up_layer)
            if f_up is not None and f_up.grad is not None:
                f_up.grad = None
        retain = gi < len(down_items) - 1
        m.backward(retain_graph=retain)
        for up_layer, up_idx in ups:
            f_up = state.codes.get(up_layer)
            if f_up is None or f_up.grad is None:
                continue
            g = f_up.grad[..., up_idx]
            a = f_up[..., up_idx].detach()
            scores[(up_layer, up_idx, down_layer, down_idx)] = \
                float((g * a).sum().item())
    return scores


class CircuitAttributionService:
    @staticmethod
    def run(db, run_id: str, *, prompt_limit: Optional[int] = None,
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
        from .circuit_capture_service import (MAX_SEQ_LENGTH, _load_sae_sync,
                                              _pad_batch)
        from .extraction_service import cleanup_gpu_memory

        run = db.query(CircuitDiscoveryRun).filter(
            CircuitDiscoveryRun.id == run_id).first()
        if run is None:
            raise ValueError(f"Discovery run {run_id} not found")
        capture = db.query(CircuitCaptureRun).filter(
            CircuitCaptureRun.id == run.capture_run_id).first()
        if capture is None:
            raise ValueError("Capture run missing")
        manifest = capture.manifest or {}
        candidates = list(run.candidates or [])
        if not candidates:
            raise ValueError("No candidates to attribute")
        granularity = (run.params or {}).get("granularity", "feature")

        # candidate pairs → feature-level (up, down) sets; clusters expand
        # to members with subgradient-to-argmax semantics handled by summing
        # member scores (documented: max's subgradient hits the argmax member;
        # summing member Σg·a aggregates exactly those nonzero terms).
        expanded = _expand_candidates(db, candidates, granularity)

        groups: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}
        for ci, pairs in expanded.items():
            for up_layer, up_idx, down_layer, down_idx in pairs:
                groups.setdefault((down_layer, down_idx), []).append(
                    (up_layer, up_idx))

        limit = int(prompt_limit or DEFAULT_PROMPT_LIMIT)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        t0 = time.monotonic()
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()

        model_record = db.query(Model).filter(
            Model.id == manifest.get("model_id")).first()
        tokenization = db.query(DatasetTokenization).filter(
            DatasetTokenization.id == manifest["corpus"]["tokenization_id"]).first()
        dataset = load_from_disk(
            str(settings.resolve_data_path(tokenization.tokenized_path)))
        dataset = dataset.select(range(min(limit, len(dataset))))

        model = None
        saes: Dict[int, Any] = {}
        totals: Dict[Tuple, float] = {}
        sign_counts: Dict[Tuple, List[int]] = {}
        try:
            resolved = (settings.resolve_data_path(model_record.file_path)
                        if model_record.file_path else None)
            model, tokenizer, _c, _m = load_model_from_hf(
                repo_id=model_record.repo_id,
                quant_format=QuantizationFormat(model_record.quantization),
                cache_dir=resolved, device_map=device,
                local_files_only=bool(resolved and resolved.exists()))
            model.eval()
            for p in model.parameters():
                p.requires_grad_(False)
            structure = discover_transformer_structure(model)
            layer_modules = {}
            for entry in manifest.get("layers", []):
                L = entry["layer"]
                sae_record = db.query(ExternalSAE).filter(
                    ExternalSAE.id == entry["sae_id"]).first()
                sae = _load_sae_sync(sae_record, device)
                for p in sae.parameters():
                    p.requires_grad_(False)
                saes[L] = sae
                # get_hookable_module expects the layer MODULE, not the int
                # index (HookManager passes structure.layers_module[idx] —
                # R1 code-review #1: passing L crashed every attribution pass).
                if L >= len(structure.layers_module):
                    raise ValueError(
                        f"Layer {L} exceeds model depth "
                        f"{len(structure.layers_module)}")
                layer_modules[L] = get_hookable_module(
                    structure.layers_module[L], "residual", structure)

            # Attribution's OWN lifecycle fields — the completed discovery's
            # status/report/candidates stay intact (R1 QA-P2).
            run.attribution_status = "running"
            run.attribution_progress = 0.0
            db.commit()

            for di in range(len(dataset)):
                if cancel_check is not None and cancel_check():
                    run.attribution_status = "cancelled"
                    db.commit()
                    return {"status": "cancelled"}
                batch = dataset[di:di + 1]
                input_ids, mask, _lens = _pad_batch(batch, tokenizer)
                state = PassThroughState()
                handles = [mod.register_forward_hook(
                    make_passthrough_hook(L, saes[L], state))
                    for L, mod in layer_modules.items()]
                try:
                    _ = model(input_ids=input_ids.to(model.device),
                              attention_mask=mask.to(model.device))
                    prompt_scores = attribute_prompt(state, groups)
                finally:
                    for h in handles:
                        h.remove()
                for k, v in prompt_scores.items():
                    totals[k] = totals.get(k, 0.0) + v
                    sign_counts.setdefault(k, []).append(
                        1 if v > 0 else (-1 if v < 0 else 0))
                run.attribution_progress = (di + 1) / len(dataset) * 100.0
                if progress_cb is not None:
                    progress_cb(run.attribution_progress)

            # ── aggregate per candidate ─────────────────────────────────
            cand_scores: List[float] = []
            for ci, pairs in expanded.items():
                score = sum(totals.get(p, 0.0) for p in pairs) / max(len(dataset), 1)
                signs = [s for p in pairs for s in sign_counts.get(p, [])]
                consistency = (sum(1 for s in signs if s == np.sign(score))
                               / len(signs)) if signs and score != 0 else 0.0
                candidates[ci]["attribution"] = {
                    "score": round(score, 6),
                    "sign_consistency": round(consistency, 3),
                    "method": "raw",
                }
                cand_scores.append(abs(score))
            floor = float(np.percentile(cand_scores,
                                        RUNG1_MAGNITUDE_PERCENTILE)) \
                if cand_scores else 0.0
            for ci in expanded:
                attr = candidates[ci]["attribution"]
                pmi = candidates[ci]["stats"]["pmi"]
                sign_agrees = (attr["score"] > 0) == (pmi > 0)
                attr["rung1_gate"] = bool(
                    sign_agrees and abs(attr["score"]) >= floor
                    and attr["score"] != 0)
            # attribution-re-ranked ordering (BOTH kept for 017's uplift)
            order = sorted(expanded.keys(),
                           key=lambda ci: abs(candidates[ci]["attribution"]["score"]),
                           reverse=True)
            for rank, ci in enumerate(order):
                candidates[ci]["orderings"]["attr_rank"] = rank

            wall = time.monotonic() - t0
            peak_vram = (torch.cuda.max_memory_allocated() / 2**20
                         if device == "cuda" else 0)
            report = dict(run.report or {})
            capture_minutes = ((manifest.get("estimate") or {})
                               .get("minutes"))
            report["attribution"] = {
                "method": "raw",
                "prompts": len(dataset),
                "downstream_groups": len(groups),
                "wall_clock_seconds": round(wall, 1),
                "peak_vram_mb": round(peak_vram, 1),
                "capture_minutes_reference": capture_minutes,
                "magnitude_floor_percentile": RUNG1_MAGNITUDE_PERCENTILE,
                "lag0_disclosure": (
                    "Attribution shares the capture's lag-0 frame; "
                    "attention-mediated paths need Tier-2.5."),
            }
            run.candidates = candidates
            run.report = report
            # Discovery stays 'completed'; attribution has its own terminal.
            run.attribution_status = "completed"
            run.attribution_progress = 100.0
            db.commit()
            return {"status": "completed",
                    "attributed": len(expanded),
                    "wall_clock_seconds": round(wall, 1)}
        finally:
            cleanup_gpu_memory(
                [m for m in [model, *saes.values()] if m is not None],
                context=f"circuit_attribution:{run_id}")


def _expand_candidates(db, candidates: List[Dict[str, Any]],
                       granularity: str) -> Dict[int, List[Tuple[int, int, int, int]]]:
    """candidate index → [(up_layer, up_idx, down_layer, down_idx)] feature
    pairs. Cluster candidates expand to member×member pairs."""
    from ..models.cluster_profile import ClusterProfile

    def _features(ref) -> List[Tuple[int, int]]:
        layer = ref["layer"]
        # Value-check, not key presence (R2 B1 class): candidate refs from
        # _unit_ref carry one key, but stay robust to both-keys-present dicts.
        if ref.get("feature_idx") is not None:
            return [(layer, int(ref["feature_idx"]))]
        prof = db.query(ClusterProfile).filter(
            ClusterProfile.id == ref.get("cluster_profile_id")).first()
        if prof is None:
            return []
        return [(layer, int(m["feature_idx"])) for m in (prof.members or [])]

    out: Dict[int, List[Tuple[int, int, int, int]]] = {}
    for ci, cand in enumerate(candidates):
        ups = _features(cand["up"])
        downs = _features(cand["down"])
        pairs = [(ul, ui, dl, di) for ul, ui in ups for dl, di in downs]
        if pairs:
            out[ci] = pairs
    return out
