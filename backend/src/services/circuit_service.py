"""
Circuit service (Feature 018, IDL-33): CRUD, assembly, promotion, rung
recomputation, and contract projection for circuits.

Validation strategy: every write round-trips through CircuitDefinitionV1 so
the CONTRACT validators (per-layer caps, edge-endpoint integrity, layer
ascension, SAE-ref completeness) are the single source of structural truth —
the service never re-implements them.
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..models.circuit import Circuit
from ..schemas.circuit_definition import (
    CircuitDefinitionV1,
    to_layer_slice,
)
from ..schemas.cluster_profile import (
    ClusterDefinitionV1,
    DefinitionModelRef,
    DefinitionProvenance,
)
from ..schemas.evidence_ladder import EvidenceRung, circuit_rung

logger = logging.getLogger(__name__)

# Single source: reuse the profile service constant (review R1 — version drift).
from .cluster_profile_service import APP_VERSION  # noqa: E402
from ..core.clock import utc_now


class CircuitValidationError(ValueError):
    """Structural validation failure — surfaces as a 422."""


class CircuitConcurrencyError(RuntimeError):
    """A stale optimistic-concurrency version — surfaces as a 409 (017 Task 3.0)."""


class CircuitService:
    # ── validation ──────────────────────────────────────────────────────

    @staticmethod
    def _validate(name: str, narrative: Optional[str], saes: list, members: list,
                  edges: list, budget: Optional[dict],
                  faithfulness: Optional[dict] = None,
                  discovery: Optional[dict] = None,
                  calibration: Optional[dict] = None) -> CircuitDefinitionV1:
        """Round-trip through the contract model — its validators are the law."""
        # An SAE entry that names no id becomes `mistudio_sae_id: null`, which
        # exports clean and only fails at miLLM as an unbound SAE. The contract
        # model cannot catch this: it accepts `mistudio_sae_id` and `sae_id`
        # via validation_alias, and cannot use extra="forbid" without
        # publishing a JSON Schema that rejects the alias it accepts (see
        # DefinitionSAERef). So the check belongs HERE, where the offending key
        # and its layer can both be named.
        _SAE_ID_KEYS = {"mistudio_sae_id", "sae_id"}
        for entry in saes or []:
            if not isinstance(entry, dict):
                continue
            if any(entry.get(k) for k in _SAE_ID_KEYS):
                continue
            layer = entry.get("layer")
            if _SAE_ID_KEYS & set(entry):
                # The key is there and NULL. The usual source is
                # `from_candidates`, where `sae_by_layer.get(L, {}).get(...)`
                # yields None for a layer the capture manifest never covered —
                # so point at that rather than at a spelling mistake.
                detail = (
                    f"SAE entry for layer {layer} has a null SAE id. If this "
                    "came from a discovery run, the capture manifest has no "
                    f"SAE for layer {layer}, so that layer cannot be served."
                )
            else:
                stray = sorted(set(entry) - {"layer", "hook_type", "n_features",
                                             "d_model", "source_hint"})
                detail = (
                    f"SAE entry for layer {layer} names no SAE id. Use "
                    "`sae_id` or `mistudio_sae_id`"
                    + (f" — got {stray}" if stray else "")
                    + "."
                )
            raise CircuitValidationError(
                detail + " A definition that cannot name its SAEs cannot be "
                "served: it would export clean and fail later at miLLM as an "
                "unbound SAE."
            )
        try:
            return CircuitDefinitionV1(
                name=name,
                narrative=narrative,
                saes=saes,
                members=members,
                edges=edges,
                budget=budget,
                faithfulness=faithfulness,
                calibration=calibration,
                discovery=discovery,
            )
        except Exception as e:  # pydantic ValidationError → domain error
            raise CircuitValidationError(str(e)) from e

    # ── discovery → circuit producer (016/017 seam) ─────────────────────

    @staticmethod
    async def from_candidates(db: AsyncSession, *, discovery_run_id: str,
                              name: str, candidate_keys: List[tuple],
                              narrative: Optional[str] = None) -> Circuit:
        """Build a circuit from selected candidates of a discovery run — the
        MISSING producer (R2): without this, no circuit carries a
        `discovery_run_id`, so a validation pass's rung-2 ES never propagates
        onto a promoted circuit (nothing to write to) and 015 reads no ES.

        `candidate_keys` = list of (up_layer, up_idx, down_layer, down_idx)
        selecting which of the run's candidates become edges. The circuit's
        members are the union of the selected candidates' endpoints; SAE refs
        come from the capture manifest. The circuit is created UNPROMOTED at
        the min rung its candidates carry (mined/attribution-supported) —
        promotion is a separate act, and validation later lifts it to rung 2.
        """
        from ..models.circuit_runs import CircuitCaptureRun, CircuitDiscoveryRun

        run = (await db.execute(select(CircuitDiscoveryRun).where(
            CircuitDiscoveryRun.id == discovery_run_id))).scalar_one_or_none()
        if run is None:
            raise CircuitValidationError(
                f"Discovery run {discovery_run_id} not found")
        capture = (await db.execute(select(CircuitCaptureRun).where(
            CircuitCaptureRun.id == run.capture_run_id))).scalar_one_or_none()
        manifest = (capture.manifest or {}) if capture else {}
        wanted = set(candidate_keys)
        cands = run.candidates or []

        def _ckey(c):
            u, d = c["up"], c["down"]
            return (u.get("layer"), u.get("feature_idx"),
                    d.get("layer"), d.get("feature_idx"))

        selected = [c for c in cands if _ckey(c) in wanted] if wanted else cands
        if not selected:
            raise CircuitValidationError(
                "No matching candidates selected for the circuit")

        # members = union of endpoint features; edges = the selected candidates
        members_by_key = {}
        edges = []
        for c in selected:
            for ref in (c["up"], c["down"]):
                if ref.get("feature_idx") is None:
                    continue
                k = (ref["layer"], ref["feature_idx"])
                members_by_key.setdefault(k, {
                    "layer": ref["layer"], "member_kind": "feature_ref",
                    "feature": {"feature_idx": ref["feature_idx"],
                                "strength": 0.0}})
            stats = c.get("stats") or {}
            edge = {
                "up": {"layer": c["up"]["layer"],
                       "feature_idx": c["up"].get("feature_idx")},
                "down": {"layer": c["down"]["layer"],
                         "feature_idx": c["down"].get("feature_idx")},
                "rung": 2 if c.get("validated_rung") == 2
                        else (1 if (c.get("attribution") or {}).get("rung1_gate")
                              else 0),
                "coactivation": {"pmi": stats.get("pmi"),
                                 "support": stats.get("support"),
                                 "null_percentile": stats.get("null_pct"),
                                 "replicated_heldout": c.get("replicated_heldout")},
            }
            # Carry the VALIDATED effect size onto the edge (R3 arc-closure): a
            # circuit built AFTER validation (validate→promote ordering) must
            # ship the rung-2 ES so 015 quantifies the hazard, not just the
            # heuristic. Written by 017's validation write-back onto the
            # candidate's `validation` field.
            val = c.get("validation") or {}
            if val.get("effect_size") is not None:
                edge["effect_size"] = val["effect_size"]
            if val.get("manifest_id"):
                edge["validation_manifest_ref"] = val["manifest_id"]
            edges.append(edge)
        layers = sorted({m["layer"] for m in members_by_key.values()})
        sae_by_layer = {e["layer"]: e for e in manifest.get("layers", [])}
        saes = [{"mistudio_sae_id": sae_by_layer.get(L, {}).get("sae_id"),
                 "layer": L} for L in layers]

        return await CircuitService.create(db, {
            "name": name, "narrative": narrative,
            "granularity": run.params.get("granularity", "feature")
            if run.params else "feature",
            "saes": saes,
            "members": list(members_by_key.values()),
            "edges": edges,
            "discovery_run_id": discovery_run_id,
            "model_id": manifest.get("model_id"),
            "discovery": {"mode": (run.params or {}).get("mode"),
                          "granularity": (run.params or {}).get("granularity"),
                          "discovery_run_id": discovery_run_id},
        })

    # ── CRUD ────────────────────────────────────────────────────────────

    @staticmethod
    async def create(db: AsyncSession, data: Dict[str, Any]) -> Circuit:
        defn = CircuitService._validate(
            data["name"], data.get("narrative"), data.get("saes", []),
            data.get("members", []), data.get("edges", []), data.get("budget"),
            data.get("faithfulness"), data.get("discovery"),
            data.get("calibration"),
        )
        # hf_id is the cross-instance-stable model identifier (R3-B2). When
        # the caller didn't supply one but referenced a local model, fill it
        # from the models table so our own exports carry it too.
        model_hf_id = data.get("model_hf_id")
        if model_hf_id is None and data.get("model_id"):
            from ..models.model import Model
            model_hf_id = (
                await db.execute(
                    select(Model.repo_id).where(Model.id == data["model_id"]))
            ).scalar_one_or_none()
        circuit = Circuit(
            name=defn.name,
            narrative=defn.narrative,
            granularity=data.get("granularity", "feature"),
            saes=[s.model_dump(mode="json") for s in defn.saes],
            members=[m.model_dump(mode="json") for m in defn.members],
            edges=[e.model_dump(mode="json") for e in defn.edges],
            budget=defn.budget.model_dump(mode="json") if defn.budget else None,
            faithfulness=(defn.faithfulness.model_dump(mode="json")
                          if defn.faithfulness else None),
            calibration=(defn.calibration.model_dump(mode="json")
                         if defn.calibration else None),
            discovery=(defn.discovery.model_dump(mode="json")
                       if defn.discovery else None),
            rung=int(defn.displayed_rung()),
            discovery_run_id=data.get("discovery_run_id"),
            model_id=data.get("model_id"),
            model_hf_id=model_hf_id,
        )
        # Import faithfulness (R2 B1): a definition's authored created_at is
        # evidence provenance — keep it instead of stamping DB-insert time.
        # Normalize tz-aware values ("…Z" — the normal foreign-export form) to
        # naive UTC: the column is timestamp-without-tz and asyncpg raises a
        # DataError on aware datetimes (R3-B1, live-reproduced 500).
        if data.get("created_at") is not None:
            authored = data["created_at"]
            if authored.tzinfo is not None:
                authored = authored.astimezone(timezone.utc).replace(tzinfo=None)
            circuit.created_at = authored
        db.add(circuit)
        await db.commit()
        await db.refresh(circuit)
        return circuit

    @staticmethod
    async def get(db: AsyncSession, circuit_id: str) -> Optional[Circuit]:
        return (
            await db.execute(select(Circuit).where(Circuit.id == circuit_id))
        ).scalar_one_or_none()

    @staticmethod
    async def list(db: AsyncSession, *, promoted: Optional[bool] = None,
                   min_rung: Optional[int] = None,
                   granularity: Optional[str] = None) -> List[Circuit]:
        q = select(Circuit).order_by(Circuit.updated_at.desc())
        if promoted is not None:
            q = q.where(Circuit.promoted == promoted)
        if min_rung is not None:
            q = q.where(Circuit.rung >= min_rung)
        if granularity:
            q = q.where(Circuit.granularity == granularity)
        return list((await db.execute(q)).scalars().all())

    @staticmethod
    async def update(db: AsyncSession, circuit: Circuit,
                     data: Dict[str, Any],
                     expected_version: Optional[int] = None) -> Circuit:
        """The ONE write path for a circuit's structure (017 Task 3.0 rule):
        every edge/member write goes through here so the contract validators
        and rung recompute always run — callers MUST NOT mutate the JSONB
        columns directly. `expected_version`, when given, enforces optimistic
        concurrency: a stale value 409s instead of clobbering a concurrent
        edit (017's validation writer vs a user editing in the panel)."""
        if expected_version is not None and circuit.version != expected_version:
            raise CircuitConcurrencyError(
                f"Circuit {circuit.id} was modified concurrently "
                f"(expected version {expected_version}, current {circuit.version})")
        merged = {
            "name": data.get("name", circuit.name),
            "narrative": data.get("narrative", circuit.narrative),
            "saes": data.get("saes", circuit.saes),
            "members": data.get("members", circuit.members),
            "edges": data.get("edges", circuit.edges),
            "budget": data.get("budget", circuit.budget),
            "faithfulness": data.get("faithfulness", circuit.faithfulness),
            "calibration": data.get("calibration", circuit.calibration),
        }
        defn = CircuitService._validate(
            merged["name"], merged["narrative"], merged["saes"],
            merged["members"], merged["edges"], merged["budget"],
            merged["faithfulness"], calibration=merged["calibration"],
        )
        if "granularity" in data and data["granularity"] is not None:
            circuit.granularity = data["granularity"]
        circuit.name = defn.name
        circuit.narrative = defn.narrative
        circuit.saes = [s.model_dump(mode="json") for s in defn.saes]
        circuit.members = [m.model_dump(mode="json") for m in defn.members]
        circuit.edges = [e.model_dump(mode="json") for e in defn.edges]
        circuit.budget = defn.budget.model_dump(mode="json") if defn.budget else None
        circuit.faithfulness = (defn.faithfulness.model_dump(mode="json")
                                if defn.faithfulness else None)
        circuit.calibration = (defn.calibration.model_dump(mode="json")
                               if defn.calibration else None)
        circuit.rung = int(defn.displayed_rung())
        circuit.version = (circuit.version or 1) + 1  # optimistic-lock bump
        await db.commit()
        await db.refresh(circuit)
        return circuit

    @staticmethod
    async def write_edge_validation(
            db: AsyncSession, circuit: Circuit,
            edge_updates: Dict[tuple, Dict[str, Any]],
            expected_version: Optional[int] = None) -> Circuit:
        """017's edge-validation writer — the ONLY way validation results
        reach a circuit's edges (018 R2-A5 `update()`-only rule). Merges
        per-edge validation fields (rung, effect_size, validation_manifest_ref,
        tested_and_failed history) into the matching edges by (up, down)
        endpoint key, then routes through update() so contract validators +
        rung recompute run and the version bumps. `edge_updates` is keyed by
        (up_layer, up_idx, down_layer, down_idx)."""
        edges = [dict(e) for e in (circuit.edges or [])]
        for e in edges:
            up, down = e.get("up", {}), e.get("down", {})
            key = (up.get("layer"), up.get("feature_idx"),
                   down.get("layer"), down.get("feature_idx"))
            if key in edge_updates:
                e.update(edge_updates[key])
        return await CircuitService.update(
            db, circuit, {"edges": edges}, expected_version=expected_version)

    @staticmethod
    async def apply_calibration(
            db: AsyncSession, circuit: Circuit, band: Dict[str, Any],
            expected_version: Optional[int] = None) -> Circuit:
        """Write a completed calibration band onto the circuit AND clamp the
        serving dial to it (IDL-37) — the ASYNC "ship the band" path, through
        update() (validators + version bump + optimistic-concurrency 409).

        Calibration itself runs in a SYNC Celery task, which uses the sync
        sibling `CircuitCalibrationService._write_calibration`. This async
        method is the entry a future async caller (e.g. an inline API apply, or
        the eventual serve-time re-verify) would use; both share the invariant
        checks. Kept — and tested — as the async contract-routed writer so the
        two never silently diverge; `test_calibration_service` pins that
        `_write_calibration` produces the same clamp this does.

        Badge, not gate: only a COMPLETED, usable band reaches here — a
        no-usable-band run never clamps (CircuitCalibrationService.run).
        """
        onset, sweet, cliff = band["onset"], band["sweet_spot"], band["cliff"]
        if not (onset <= sweet <= cliff):
            raise CircuitValidationError(
                f"calibration band is not ordered (onset={onset}, "
                f"sweet_spot={sweet}, cliff={cliff}); refusing to clamp the dial "
                "to an inverted range")
        # Merge the clamp into the EXISTING budget (keep formula_id, per-layer
        # budgets); only the dial envelope changes. The intensity∈range invariant
        # is now enforced by CircuitBudget itself, so update()'s contract
        # validation catches any bad clamp for BOTH this and the sync path.
        budget = dict(circuit.budget or {})
        budget["intensity_range"] = [onset, cliff]
        budget["intensity"] = sweet
        return await CircuitService.update(
            db, circuit,
            {"budget": budget, "calibration": band},
            expected_version=expected_version,
        )

    @staticmethod
    async def delete(db: AsyncSession, circuit: Circuit) -> None:
        await db.delete(circuit)
        await db.commit()

    @staticmethod
    async def set_promoted(db: AsyncSession, circuit: Circuit,
                           promoted: bool = True) -> Circuit:
        """Badge, not gate (BR-012) — and reversible: a badge you can pin
        is a badge you can unpin (review R1 finding #7)."""
        circuit.promoted = promoted
        await db.commit()
        await db.refresh(circuit)
        return circuit


    # ── rung maintenance (017 writes validation results, then calls this) ──

    @staticmethod
    async def recompute_rung(db: AsyncSession, circuit: Circuit) -> Circuit:
        # Tolerate malformed stored rungs (017 writes here): clamp into 0..3
        # instead of crashing the write path (review R1 QA-4).
        def _safe(e):
            try:
                return EvidenceRung(max(0, min(3, int(e.get("rung", 0)))))
            except (TypeError, ValueError, AttributeError):
                # AttributeError: a non-dict edge entry — exactly the malformed
                # stored data this clamp exists to survive (R2 B9).
                return EvidenceRung.MINED
        rungs = [_safe(e) for e in (circuit.edges or [])]
        circuit.rung = int(circuit_rung(rungs))
        await db.commit()
        await db.refresh(circuit)
        return circuit

    # ── contract projection ─────────────────────────────────────────────

    @staticmethod
    def to_definition(circuit: Circuit) -> CircuitDefinitionV1:
        defn = CircuitService._validate(
            circuit.name, circuit.narrative, circuit.saes, circuit.members,
            circuit.edges, circuit.budget, circuit.faithfulness,
            getattr(circuit, "discovery", None),
            getattr(circuit, "calibration", None),
        )
        defn.model = DefinitionModelRef(
            mistudio_model_id=circuit.model_id,
            hf_id=getattr(circuit, "model_hf_id", None),
        )
        defn.provenance = DefinitionProvenance(
            created_at=circuit.created_at,
            exported_at=utc_now(),
            mistudio_version=APP_VERSION,
        )
        return defn

    @staticmethod
    def slices_of(defn: CircuitDefinitionV1) -> List[ClusterDefinitionV1]:
        """BR-014: one valid v1 slice per member-bearing layer (definition-first
        so callers compute the parent rung ONCE from the same object)."""
        layers = sorted({m.layer for m in defn.members})
        return [to_layer_slice(defn, layer) for layer in layers]

    @staticmethod
    def to_slices(circuit: Circuit) -> List[ClusterDefinitionV1]:
        return CircuitService.slices_of(CircuitService.to_definition(circuit))
