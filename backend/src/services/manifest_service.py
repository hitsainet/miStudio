"""
Manifest service (Feature 017, FTDD §4): persist / retrieve / reproduce
validation manifests. Reproduction re-executes from the self-contained
payload and stores a `reproduction` manifest with per-value deltas + a
tolerance verdict — the correctness test that a manifest really is
self-contained.
"""

import logging
from typing import Any, Dict, List, Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..models.validation_manifest import ValidationManifest

logger = logging.getLogger(__name__)

# Fields a manifest payload MUST carry to be self-contained (no live refs).
_REQUIRED_PAYLOAD_KEYS = {"intervention", "config", "seeds"}
# A calibration manifest reproduces a band from its probes + config + the judged
# trace, and carries first-class `transcripts` (the generated text per dial/prompt)
# so the manifest is analysis-ready for an LLM meaning pass. `transcripts` is
# REQUIRED on NEW manifests; legacy calibration manifests written before this key
# existed are never re-validated (validate_payload runs only on WRITE), so they
# stay readable + reproducible — pinned by test_calibration_transcript back-compat.
_REQUIRED_CALIBRATION_KEYS = {"probes", "config", "band", "trace", "transcripts"}
# A steering-samples manifest is the recorder's self-contained transcript record:
# which artifact + dials + prompts were run, and the generated text per
# (prompt, dial) — the raw material for an LLM meaning-analysis pass.
_REQUIRED_STEERING_SAMPLES_KEYS = {"artifact", "dials", "prompts", "transcripts", "config"}


class ManifestError(ValueError):
    """Malformed manifest — surfaces as a 422."""


def validate_payload(kind: str, payload: Dict[str, Any]) -> None:
    """A manifest with live refs that can drift is not reproducible."""
    if kind not in ("edge_batch", "faithfulness", "reproduction", "calibration",
                    "steering_samples"):
        raise ManifestError(f"Unknown manifest kind {kind!r}")
    if kind == "calibration":
        missing = _REQUIRED_CALIBRATION_KEYS - set(payload or {})
        if missing:
            raise ManifestError(
                "calibration manifest payload missing self-contained keys: "
                f"{sorted(missing)}")
    if kind == "steering_samples":
        missing = _REQUIRED_STEERING_SAMPLES_KEYS - set(payload or {})
        if missing:
            raise ManifestError(
                "steering_samples manifest payload missing self-contained keys: "
                f"{sorted(missing)}")
    if kind in ("edge_batch", "faithfulness"):
        missing = _REQUIRED_PAYLOAD_KEYS - set(payload or {})
        if missing:
            raise ManifestError(
                f"{kind} manifest payload missing self-contained keys: "
                f"{sorted(missing)}")
    # No secrets/filesystem paths leak into a portable manifest — but the guard
    # is about internal REFS, not free text. A user prompt or a model generation
    # is legitimately arbitrary text that may begin with "/data/" or "/home/";
    # scanning it would false-positive and discard a completed GPU run (R1). So
    # keys holding user/model TEXT are exempt from the path scan.
    _assert_no_paths(payload, text_keys=_TEXT_KEYS)


# Payload keys whose VALUES are free user/model text (not internal refs) — exempt
# from the filesystem-path scan. Applied at any depth.
_TEXT_KEYS = frozenset({
    "prompt", "prompts", "generation", "unsteered_output", "steered_output",
    "expected", "narrative",
})


def _is_free_text(v: Any) -> bool:
    """A string, or a list of strings — free user/model text with no nested
    structure that could hide a ref."""
    if isinstance(v, str):
        return True
    if isinstance(v, (list, tuple)):
        return all(isinstance(x, str) for x in v)
    return False


def _assert_no_paths(obj: Any, _depth: int = 0, text_keys: frozenset = frozenset()) -> None:
    if _depth > 12:
        return
    if isinstance(obj, str):
        if obj.startswith("/data/") or obj.startswith("/home/"):
            raise ManifestError("manifest payload must not embed filesystem paths")
    elif isinstance(obj, dict):
        for k, v in obj.items():
            # Exempt a text key's value when it is free TEXT — a string, or a
            # list of strings (e.g. `prompts`). A DICT (or a list containing
            # dicts) under a text-named key is still walked, so a ref can't hide
            # nested inside it (R2 hardening).
            if k in text_keys and _is_free_text(v):
                continue
            _assert_no_paths(v, _depth + 1, text_keys)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _assert_no_paths(v, _depth + 1, text_keys)


class ManifestService:
    @staticmethod
    async def create(db: AsyncSession, *, kind: str, payload: Dict[str, Any],
                     discovery_run_id: Optional[str] = None,
                     circuit_id: Optional[str] = None,
                     parent_manifest_id: Optional[str] = None) -> ValidationManifest:
        validate_payload(kind, payload)
        m = ValidationManifest(
            kind=kind, payload=payload, discovery_run_id=discovery_run_id,
            circuit_id=circuit_id, parent_manifest_id=parent_manifest_id)
        db.add(m)
        await db.commit()
        await db.refresh(m)
        return m

    @staticmethod
    async def get(db: AsyncSession, manifest_id: str) -> Optional[ValidationManifest]:
        return (await db.execute(
            select(ValidationManifest).where(
                ValidationManifest.id == manifest_id))).scalar_one_or_none()

    @staticmethod
    async def list_by_parent(db: AsyncSession, *,
                             discovery_run_id: Optional[str] = None,
                             circuit_id: Optional[str] = None
                             ) -> List[ValidationManifest]:
        q = select(ValidationManifest).order_by(
            ValidationManifest.created_at.desc())
        if discovery_run_id:
            q = q.where(ValidationManifest.discovery_run_id == discovery_run_id)
        if circuit_id:
            q = q.where(ValidationManifest.circuit_id == circuit_id)
        return list((await db.execute(q)).scalars().all())

    @staticmethod
    def reproduction_verdict(original: Dict[str, Any], repro: Dict[str, Any],
                             tolerance: float = 0.1) -> Dict[str, Any]:
        """Compare per-edge ES between an original edge_batch payload and a
        re-executed one. Deterministic greedy passes ⇒ deltas should be ~0;
        tolerance guards fp noise. Returns {deltas, max_delta, within_tolerance}."""
        orig_edges = {_edge_key(e): e for e in original.get("edges", [])}
        deltas = []
        max_delta = 0.0
        for e in repro.get("edges", []):
            k = _edge_key(e)
            o = orig_edges.get(k)
            if o is None or o.get("effect_size") is None or e.get("effect_size") is None:
                continue
            d = abs(float(e["effect_size"]) - float(o["effect_size"]))
            deltas.append({"edge": k, "delta": round(d, 5)})
            max_delta = max(max_delta, d)
        if not deltas:
            # Nothing was compared ⇒ NOT a pass (R1 #14): an empty overlap used
            # to report within_tolerance=True ("reproduced") having compared 0
            # edges. Reproducing nothing is not reproducing.
            return {"deltas": [], "max_delta": None, "within_tolerance": None,
                    "tolerance": tolerance,
                    "reason": "no overlapping edges to compare"}
        return {"deltas": deltas, "max_delta": round(max_delta, 5),
                "within_tolerance": max_delta <= tolerance,
                "tolerance": tolerance}

    @staticmethod
    def calibration_reproduction_verdict(
            original: Dict[str, Any], repro: Dict[str, Any],
            tolerance: float = 0.1) -> Dict[str, Any]:
        """Compare the band (onset/sweet_spot/cliff) between an original
        calibration manifest payload and a re-executed one. A seeded search over
        the same probes should reproduce the band within tolerance; a large delta
        means the measurement is not reproducible (model nondeterminism, or a
        judge that answers inconsistently)."""
        o = original.get("band") or {}
        r = repro.get("band") or {}
        keys = ("onset", "sweet_spot", "cliff")
        deltas = {}
        max_delta = 0.0
        compared = 0
        for k in keys:
            if o.get(k) is None or r.get(k) is None:
                continue
            d = abs(float(r[k]) - float(o[k]))
            deltas[k] = round(d, 5)
            max_delta = max(max_delta, d)
            compared += 1
        if compared == 0:
            return {"deltas": {}, "max_delta": None, "within_tolerance": None,
                    "tolerance": tolerance,
                    "reason": "no band values to compare"}
        return {"deltas": deltas, "max_delta": round(max_delta, 5),
                "within_tolerance": max_delta <= tolerance,
                "tolerance": tolerance}


def _edge_key(e: Dict[str, Any]):
    up, down = e.get("up", {}), e.get("down", {})
    return (up.get("layer"), up.get("feature_idx"),
            down.get("layer"), down.get("feature_idx"))
