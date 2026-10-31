"""Check a circuit capture store against an independent forward pass (016 FTASKS 6.1).

The open half of 016 6.1: "the store's positions and error-norm values checked
against an independent forward pass". A capture's event and error-norm files were
counted on hardware (events = the probe's count exactly, one error norm per token);
their VALUES were never re-derived. This script re-derives them.

What is independent and what is shared:

* INDEPENDENT: the residual read. A plain `register_forward_hook` on the decoder
  layer module (`discover_transformer_structure(model).layers_module[L]`), the
  whole-layer output (resid_post), not the capture's `HookManager`. A capture that
  hooked the wrong module (memory: residual-hook-was-a-norm-output) fails here.
  Also independent: thresholds, the above-threshold set, positions and the error
  norm, recomputed here from that read.
* SHARED, deliberately: the model and SAE loaders and the SAE encode with its
  training normalisation (`encode_with_training_normalization`). Those are what the
  capture is supposed to use; re-implementing them would check a different SAE.

Per document it reports:
  events          stored events for the doc
  act_mismatch    stored events whose act differs from the recomputed z at the same
                  (token, feature) by more than the tolerance
  missing         recomputed above-threshold (token, feature) pairs absent from the
                  store, excluding borderline ones
  extra           stored pairs the recomputation puts below threshold, excluding
                  borderline ones
  borderline      pairs within the tolerance band of the threshold (reported, never
                  failed: fp16 storage and batch numerics decide them)
  norm_tokens / norm_missing / norm_mismatch  the same for the per-token error norm

PASS = every doc has act_mismatch = missing = extra = norm_missing = norm_mismatch = 0.

Run inside the backend pod (it reads the capture's own manifest, dataset and SAEs):

    python -m scripts.verify_capture_store cap_7b1ba1cfcf1c --docs 32 --device cuda
    python -m scripts.verify_capture_store cap_7b1ba1cfcf1c --docs 4 --device cpu
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict

import numpy as np

#: fp16 storage keeps ~3 significant digits; batched vs single kernels differ a little more.
DEFAULT_RTOL = 2e-2
DEFAULT_ATOL = 2e-3


def compare_doc(events: np.ndarray, norms: np.ndarray, z: np.ndarray, err: np.ndarray,
                thresh: np.ndarray, *, rtol: float = DEFAULT_RTOL,
                atol: float = DEFAULT_ATOL) -> Dict[str, int]:
    """Compare one document's stored rows with its recomputation.

    events: structured rows (token_pos, feature_idx, act) for this doc
    norms:  structured rows (token_pos, norm) for this doc
    z:      [n_tokens, d_sae] recomputed codes for the doc's REAL tokens
    err:    [n_tokens] recomputed reconstruction-error norms
    thresh: [d_sae] the per-feature threshold the capture writes above
    """
    n_tok, d_sae = z.shape
    out = {"events": int(len(events)), "act_mismatch": 0, "missing": 0, "extra": 0,
           "borderline": 0, "out_of_range": 0, "norm_tokens": int(len(norms)),
           "norm_missing": 0, "norm_mismatch": 0}

    pos = np.asarray(events["token_pos"], dtype=np.int64)
    feat = np.asarray(events["feature_idx"], dtype=np.int64)
    act = np.asarray(events["act"], dtype=np.float32)
    ok = (pos < n_tok) & (feat < d_sae)
    out["out_of_range"] = int((~ok).sum())
    pos, feat, act = pos[ok], feat[ok], act[ok]

    recomputed = z[pos, feat].astype(np.float32)
    out["act_mismatch"] = int((~np.isclose(act, recomputed, rtol=rtol, atol=atol)).sum())

    band = np.abs(z - thresh[None, :]) <= (atol + rtol * np.abs(thresh[None, :]))
    above = (z > thresh[None, :]) & ~band
    stored = np.zeros_like(above, dtype=bool)
    stored[pos, feat] = True
    out["missing"] = int((above & ~stored).sum())
    out["extra"] = int((stored & ~(z > thresh[None, :]) & ~band).sum())
    out["borderline"] = int((band & ((z > thresh[None, :]) | stored)).sum())

    have = np.zeros(n_tok, dtype=bool)
    npos = np.asarray(norms["token_pos"], dtype=np.int64)
    nval = np.asarray(norms["norm"], dtype=np.float32)
    keep = npos < n_tok
    out["out_of_range"] += int((~keep).sum())
    npos, nval = npos[keep], nval[keep]
    have[npos] = True
    out["norm_missing"] = int((~have).sum())
    out["norm_mismatch"] = int((~np.isclose(nval, err[npos].astype(np.float32),
                                            rtol=rtol, atol=atol)).sum())
    return out


def doc_passes(row: Dict[str, int]) -> bool:
    return not any(row[k] for k in ("act_mismatch", "missing", "extra", "out_of_range",
                                    "norm_missing", "norm_mismatch"))


def _run(capture_id: str, n_docs: int, device: str, rtol: float, atol: float) -> Dict[str, Any]:
    import torch
    from datasets import load_from_disk

    from src.core.config import settings
    from src.core.database import SyncSessionLocal
    from src.ml.layer_discovery import discover_transformer_structure
    from src.ml.model_loader import load_model_from_hf
    from src.ml.sparse_autoencoder import encode_with_training_normalization
    from src.models.circuit_runs import CircuitCaptureRun
    from src.models.dataset_tokenization import DatasetTokenization
    from src.models.external_sae import ExternalSAE
    from src.models.model import Model, QuantizationFormat
    from src.services.circuit_capture_service import (
        CAPTURE_BATCH_DOCS,
        MAX_SEQ_LENGTH,
        PROBE_SAMPLES,
        _load_sae_sync,
    )

    db = SyncSessionLocal()
    try:
        run = db.query(CircuitCaptureRun).filter(CircuitCaptureRun.id == capture_id).one()
        manifest = dict(run.manifest)
        store_dir = settings.resolve_data_path(run.store_path)
        model_row = db.query(Model).filter(Model.id == manifest["model_id"]).one()
        tok_row = db.query(DatasetTokenization).filter(
            DatasetTokenization.id == manifest["corpus"]["tokenization_id"]).one()
        layers = {e["layer"]: e for e in manifest["layers"]}
        sae_rows = {L: db.query(ExternalSAE).filter(ExternalSAE.id == e["sae_id"]).one()
                    for L, e in layers.items()}
    finally:
        db.close()

    dataset = load_from_disk(str(settings.resolve_data_path(tok_row.tokenized_path)))
    sample_cap = min(int(manifest["corpus"]["sample_cap"]), len(dataset))
    dataset = dataset.select(range(sample_cap))
    path = settings.resolve_data_path(model_row.file_path) if model_row.file_path else None
    model, tokenizer, _cfg, _meta = load_model_from_hf(
        repo_id=model_row.repo_id, quant_format=QuantizationFormat(model_row.quantization),
        cache_dir=path, local_files_only=bool(path and path.exists()), device_map=device)
    model.eval()
    saes = {L: _load_sae_sync(row, device) for L, row in sae_rows.items()}
    blocks = discover_transformer_structure(model).layers_module

    captured: Dict[int, Any] = {}

    def hook_for(L):
        def hook(_module, _inp, output):
            captured[L] = (output[0] if isinstance(output, tuple) else output).detach()
        return hook

    handles = [blocks[L].register_forward_hook(hook_for(L)) for L in layers]

    def forward(docs):
        """Right-padded batch like the capture's, so padding effects match."""
        seqs = [list(dataset[i]["input_ids"])[:MAX_SEQ_LENGTH] for i in docs]
        width = max(len(s) for s in seqs)
        pad = tokenizer.pad_token_id or tokenizer.eos_token_id or 0
        ids = torch.full((len(seqs), width), pad, dtype=torch.long)
        mask = torch.zeros_like(ids)
        for r, s in enumerate(seqs):
            ids[r, :len(s)] = torch.tensor(s)
            mask[r, :len(s)] = 1
        with torch.no_grad():
            model(input_ids=ids.to(device), attention_mask=mask.to(device))
        codes = {}
        for L in layers:
            acts = captured[L].to(device=device, dtype=torch.float32)
            b, s, h = acts.shape
            flat = acts.reshape(-1, h)
            with torch.no_grad():
                z = encode_with_training_normalization(saes[L], flat)
                err = (flat - saes[L].decode(z)).norm(dim=-1)
            codes[L] = (z.reshape(b, s, -1).cpu().numpy(), err.reshape(b, s).cpu().numpy())
        return [len(s) for s in seqs], codes

    try:
        # Thresholds: per-feature max over the probe documents (padding included,
        # as the capture's probe does), then clamp(eps * max, floor).
        probe_n = min(PROBE_SAMPLES, len(dataset))
        probe_max: Dict[int, np.ndarray] = {}
        for start in range(0, probe_n, CAPTURE_BATCH_DOCS):
            _lens, codes = forward(range(start, min(start + CAPTURE_BATCH_DOCS, probe_n)))
            for L, (z, _e) in codes.items():
                m = z.reshape(-1, z.shape[-1]).max(axis=0)
                probe_max[L] = np.maximum(probe_max[L], m) if L in probe_max else m
        thresh = {}
        for L, e in layers.items():
            eps, floor = float(e["epsilon"]), float(e["theta_floor"])
            thresh[L] = (np.maximum(eps * probe_max[L], floor) if eps > 0
                         else np.full_like(probe_max[L], floor))

        docs = list(range(min(n_docs, len(dataset))))
        ev = {L: np.load(store_dir / f"layer_{L}.events", mmap_mode="r") for L in layers}
        en = {L: np.load(store_dir / f"layer_{L}.errnorm", mmap_mode="r") for L in layers}
        ev_docs = {L: ev[L][np.isin(ev[L]["doc_id"], docs)] for L in layers}
        report: Dict[str, Any] = {"capture_id": capture_id, "device": device,
                                  "rtol": rtol, "atol": atol, "layers": {}}
        for start in range(0, len(docs), CAPTURE_BATCH_DOCS):
            batch = docs[start:start + CAPTURE_BATCH_DOCS]
            lens, codes = forward(batch)
            for L in layers:
                z, err = codes[L]
                d = en[L]["doc_id"]
                for r, doc in enumerate(batch):
                    n = lens[r]
                    lo, hi = np.searchsorted(d, doc, "left"), np.searchsorted(d, doc, "right")
                    row = compare_doc(ev_docs[L][ev_docs[L]["doc_id"] == doc], en[L][lo:hi],
                                      z[r, :n], err[r, :n], thresh[L], rtol=rtol, atol=atol)
                    report["layers"].setdefault(str(L), {})[str(doc)] = row
        report["pass"] = all(doc_passes(r) for per in report["layers"].values()
                             for r in per.values())
        return report
    finally:
        for h in handles:
            h.remove()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("capture_id")
    ap.add_argument("--docs", type=int, default=4)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--rtol", type=float, default=DEFAULT_RTOL)
    ap.add_argument("--atol", type=float, default=DEFAULT_ATOL)
    args = ap.parse_args(argv)
    report = _run(args.capture_id, args.docs, args.device, args.rtol, args.atol)
    print(json.dumps(report, indent=1))
    print("PASS" if report["pass"] else "FAIL", file=sys.stderr)
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
