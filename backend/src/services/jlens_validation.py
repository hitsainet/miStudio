"""
Artifact validation (BR-030, RSK-014).

WHY THIS EXISTS RATHER THAN A TRY/EXCEPT AT THE CONSUMER. The downstream lens
loader is best-effort and fails AT REQUEST TIME WITHOUT RAISING. A bad artifact
therefore presents as a feature that quietly returns nothing — an empty readout
is indistinguishable from a real readout with no content, which is the same
reason `/jlens/readout` refuses to fabricate one. Validation runs BEFORE
handover because after handover there is nothing left to detect.

SIX CLASSES, and each catches something the others cannot:

  STRUCTURAL          it deserializes and has the right shapes
  NAMING              exactly one lens file, named as the consumer expects
  ENVELOPE            its size matches THIS MODEL's arithmetic (BR-006)
  SEMANTIC            it actually recovers a known unspoken intermediate
  CROSS_IMPLEMENTATION our reader and the consumer's agree
  ROUND_TRIP          mounted and served, a Jacobian request returns content

Structure can be perfect while content is absent, which is why SEMANTIC is not
implied by STRUCTURAL. Both readers can be self-consistent while disagreeing
with each other, which is why CROSS_IMPLEMENTATION is not implied by SEMANTIC.
And everything can pass in-process while the mounted artifact is never picked
up, which is why ROUND_TRIP is explicit rather than assumed.

NOTHING HERE SCORES NEXT-TOKEN AGREEMENT (BR-004). The J-lens is deliberately
worse on that measure than the logit lens through most of the network, so a
validation check that rewarded it would reject good artifacts and accept bad
ones. See `test_jlens_validation.py::test_no_check_scores_next_token_agreement`.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

logger = logging.getLogger(__name__)


class CheckClass(str, Enum):
    STRUCTURAL = "structural"
    NAMING = "naming"
    ENVELOPE = "envelope"
    SEMANTIC = "semantic"
    CROSS_IMPLEMENTATION = "cross_implementation"
    ROUND_TRIP = "round_trip"


class CheckStatus(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    # Distinct from FAIL on purpose: a check that could not run has not passed,
    # and collapsing the two either blocks a good artifact or — far worse —
    # counts an unrun check as a pass.
    NOT_RUN = "not_run"
    #: Not run, KNOWN not to be runnable here, and publishable anyway.
    #:
    #: NOT A PASS, AND THIS DISTINCTION WAS BEING ROUTED AROUND. `_local_pass`
    #: stamped the two consumer-interop classes with `PASS` and a detail string
    #: reading "deferred: …". `_write_report` then serialised
    #: `{"check": "cross_implementation", "status": "pass"}` into
    #: `validation.json` — the file whose entire purpose is to TRAVEL WITH THE
    #: ARTIFACT to a consumer that pulled it off HuggingFace. That consumer sees
    #: a green six-class pass and can only learn otherwise by reading English
    #: prose in a neighbouring field.
    #:
    #: The comment above `NOT_RUN` says counting an unrun check as a pass is
    #: "far worse" than blocking a good artifact. The value was carefully
    #: protected and then bypassed under a different name.
    DEFERRED = "deferred"


#: The only classes that may be DEFERRED, and the reason the valve cannot widen.
#:
#: Both need a live external consumer to run at all, so requiring them before
#: publishing would make the Jacobian path permanently unreachable. The four
#: LOCAL classes must never appear here: they are what `serviceable` gates on,
#: and deferring one of those would let an unvalidated artifact serve.
DEFERRABLE = frozenset({CheckClass.CROSS_IMPLEMENTATION, CheckClass.ROUND_TRIP})


@dataclass
class CheckResult:
    check: CheckClass
    status: CheckStatus
    detail: str
    evidence: Dict[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return self.status is CheckStatus.PASS


@dataclass
class ValidationReport:
    results: List[CheckResult]

    @property
    def passed(self) -> bool:
        """FAIL-CLOSED: every class must have run and passed, or be DEFERRABLE.

        A missing class is not a pass. The whole point of the suite is that the
        consumer's failure is silence, so "we did not check" and "we checked and
        it was fine" must never produce the same verdict.

        A DEFERRED result counts here ONLY for a class in `DEFERRABLE`. That is
        the same publishing behaviour `_local_pass` produced by stamping those
        two with `PASS` — the difference is that the recorded status is now
        truthful, so `validation.json` no longer tells a downstream consumer
        that an interop check succeeded when nothing ran. `NOT_RUN` is still
        never a pass, for any class.
        """
        seen = {r.check for r in self.results}
        if seen != set(CheckClass):
            return False
        return all(
            r.passed
            or (r.status is CheckStatus.DEFERRED and r.check in DEFERRABLE)
            for r in self.results
        )

    @property
    def cleared_for_handover(self) -> bool:
        """Every class LITERALLY passed — nothing deferred.

        The real BR-030 gate, and today it is False everywhere, which is the
        correct reading: no A5/A6 harness exists yet, so no artifact this
        project has produced has been checked against a live consumer. `passed`
        answers "may we publish locally"; this answers "has this been proven
        interoperable", and conflating them is what made the two consumer
        classes look green.
        """
        seen = {r.check for r in self.results}
        if seen != set(CheckClass):
            return False
        return all(r.passed for r in self.results)

    @property
    def serviceable(self) -> bool:
        """Safe to serve from MISTUDIO'S OWN readout path.

        A NARROWER GATE THAN `passed`, and the distinction is deliberate rather
        than a relaxation.

        `passed` gates HANDOVER TO AN EXTERNAL CONSUMER, which is what BR-030
        is about: that consumer's lens loading fails at request time without
        raising, so an artifact reaching it unvalidated becomes a feature that
        quietly returns nothing. CROSS_IMPLEMENTATION and ROUND_TRIP exist to
        catch exactly that, and both require a live consumer to run at all.

        miStudio's own readout is a different risk profile: the code is ours and
        it RAISES — `JacobianTransport` refuses a missing layer rather than
        falling back to identity. So local serving is gated on the four classes
        that bear on local correctness, and the two consumer-interop classes are
        required before handover and not before serving.

        Collapsing the two would either make the Jacobian path permanently
        unreachable (no external consumer, no serving) or, far worse, let an
        unvalidated artifact reach Neuronpedia because one gate had to be loose
        enough for both jobs.
        """
        local = {
            CheckClass.STRUCTURAL,
            CheckClass.NAMING,
            CheckClass.ENVELOPE,
            CheckClass.SEMANTIC,
        }
        seen = {r.check for r in self.results if r.check in local}
        if seen != local:
            return False
        return all(r.passed for r in self.results if r.check in local)

    @property
    def missing(self) -> List[CheckClass]:
        return sorted(set(CheckClass) - {r.check for r in self.results}, key=lambda c: c.value)

    def summary(self) -> str:
        parts = [f"{r.check.value}={r.status.value}" for r in self.results]
        for m in self.missing:
            parts.append(f"{m.value}=not_run")
        return ", ".join(parts)


# Consumer-facing filename convention. Anchored at both ends: an unanchored
# pattern accepts `not_a_lens.pt.bak` and this project has already shipped a
# regex that lost an anchor.
LENS_FILENAME = re.compile(r"^([a-z0-9][a-z0-9._-]*)_jacobian_lens\.pt$")

# Serialisation container overhead: zip headers, the pickle, per-tensor records.
# Capped at this, and additionally at half the required size, because a FLAT
# allowance larger than a small model's whole materialised dictionary would
# blind the check exactly where the numbers are smallest. See
# `container_allowance`.
CONTAINER_OVERHEAD_BYTES = 64 * 1024


def container_allowance(required_bytes: int) -> int:
    """Bytes of container overhead to forgive, bounded at both ends.

    An absolute allowance rather than a wider multiplier: a multiplier scales
    with the model and opens a gap a partial materialisation could hide in.
    But an absolute allowance that exceeds a small model's entire materialised
    dictionary is just as blind, so it is also capped at half the required
    size. At real scale (~134 MB required) the cap never binds and the
    allowance is noise.
    """
    return max(4096, min(CONTAINER_OVERHEAD_BYTES, required_bytes // 2))


def defer_consumer_checks(report: "ValidationReport") -> "ValidationReport":
    """Mark the consumer-interop classes DEFERRED so the artifact can publish.

    ONE IMPLEMENTATION, SHARED. This began life as `_local_pass`, private to the
    fit worker — so the acquisition path would have had to either import a
    private helper or grow a second copy of the same judgement. A second copy is
    how two workers come to disagree about what "publishable" means.

    The caller supplies its OWN wording for `detail`, because the reason differs:
    a local fit defers because no external consumer is running, and an acquired
    artifact defers for the same reason but must not claim a fit was performed.
    Copying the fitter's sentence onto an acquisition would put a false
    provenance statement in the file that travels with the lens.
    """
    results = [r for r in report.results if r.check not in DEFERRABLE]
    for check in sorted(DEFERRABLE, key=lambda c: c.value):
        existing = next((r for r in report.results if r.check == check), None)
        # AN ALREADY-RUN CHECK IS LEFT ALONE. When an A5/A6 harness finally runs
        # these for real, deferring over the top would erase the only evidence
        # this project has ever had for them.
        if existing is not None and existing.status is not CheckStatus.NOT_RUN:
            results.append(existing)
            continue
        results.append(
            CheckResult(
                check,
                CheckStatus.DEFERRED,
                "requires a live external consumer; not run here",
            )
        )
    return ValidationReport(results)


def check_naming(directory: Path) -> CheckResult:
    """Exactly one conformant lens file in the mounted directory.

    "Exactly one" is the check, not "at least one": the consumer picks among
    several without saying which, so two artifacts in a directory is a
    non-deterministic serve, not a convenience.
    """
    if not directory.is_dir():
        return CheckResult(
            CheckClass.NAMING, CheckStatus.FAIL, f"{directory} is not a directory"
        )

    lens_files = sorted(p.name for p in directory.glob("*.pt"))
    conformant = [n for n in lens_files if LENS_FILENAME.match(n)]

    if not conformant:
        return CheckResult(
            CheckClass.NAMING,
            CheckStatus.FAIL,
            f"no file matching <slug>_jacobian_lens.pt in {directory}",
            {"found": lens_files},
        )
    if len(conformant) > 1:
        return CheckResult(
            CheckClass.NAMING,
            CheckStatus.FAIL,
            "more than one lens file; the consumer picks among them silently",
            {"found": conformant},
        )
    if len(lens_files) > len(conformant):
        return CheckResult(
            CheckClass.NAMING,
            CheckStatus.FAIL,
            "non-conformant .pt files share the mounted directory",
            {"found": lens_files},
        )
    return CheckResult(
        CheckClass.NAMING, CheckStatus.PASS, conformant[0], {"file": conformant[0]}
    )


def check_structural(payload: Any, d_model: int, expected_layers: Sequence[int]) -> CheckResult:
    """Required keys present, every Jacobian square of side d_model.

    `payload` is what weights-only deserialisation returned. A non-square
    matrix, or one of the wrong side, still loads and still produces a readout
    — of the wrong thing.
    """
    if not isinstance(payload, dict):
        return CheckResult(
            CheckClass.STRUCTURAL,
            CheckStatus.FAIL,
            f"payload is {type(payload).__name__}, expected a mapping of layer -> matrix",
        )

    coerced: Dict[int, Any] = {}
    for key, value in payload.items():
        try:
            coerced[int(key)] = value
        except (TypeError, ValueError):
            return CheckResult(
                CheckClass.STRUCTURAL,
                CheckStatus.FAIL,
                f"layer key {key!r} is not coercible to an integer",
            )

    missing = [layer for layer in expected_layers if layer not in coerced]
    if missing:
        return CheckResult(
            CheckClass.STRUCTURAL,
            CheckStatus.FAIL,
            f"missing layers {missing}",
            {"missing": missing},
        )

    for layer, matrix in sorted(coerced.items()):
        shape = tuple(getattr(matrix, "shape", ()))
        if len(shape) != 2 or shape[0] != shape[1]:
            return CheckResult(
                CheckClass.STRUCTURAL,
                CheckStatus.FAIL,
                f"layer {layer} has shape {shape}; a Jacobian must be square",
            )
        if shape[0] != d_model:
            return CheckResult(
                CheckClass.STRUCTURAL,
                CheckStatus.FAIL,
                f"layer {layer} has side {shape[0]}, model d_model is {d_model}",
            )
        # NON-FINITE ENTRIES. An artifact whose fp16 cast overflowed contains
        # inf, deserialises cleanly, is exactly the right shape and size, and
        # reads out garbage. Found on the first real fit, where 0.3% of GPT-2's
        # entries saturated at fp16's ceiling — every other structural check
        # passed.
        finite = getattr(matrix, "isfinite", None)
        if finite is not None and not bool(finite().all()):
            import torch as _t

            bad = int((~matrix.isfinite()).sum())
            return CheckResult(
                CheckClass.STRUCTURAL,
                CheckStatus.FAIL,
                f"layer {layer} has {bad} non-finite entries — the fp16 cast "
                "overflowed. The artifact loads and reads out garbage.",
            )

    return CheckResult(
        CheckClass.STRUCTURAL,
        CheckStatus.PASS,
        f"{len(coerced)} layers, all {d_model}x{d_model}",
        {"layers": sorted(coerced)},
    )


def check_envelope(
    size_bytes: int,
    d_model: int,
    n_layers: int,
    n_vocab: int,
    dtype_bytes: int = 2,
    tolerance: float = 1.5,
) -> CheckResult:
    """Size within tolerance of THIS MODEL's arithmetic (BR-006, IDL-42).

    Both bounds are derived, never constants. The required-vs-materialised
    ratio scales with vocabulary — about 32x at a 65k vocab and 111x at 256k —
    so a bound tuned on one model passes on another while missing a real
    materialisation. `n_vocab` is taken as an argument for exactly that reason:
    it is what makes the "did someone materialise W_U J" question answerable.
    """
    required = d_model * d_model * dtype_bytes * n_layers
    materialised = n_vocab * d_model * dtype_bytes * n_layers
    # The measured figure is a FILE size and the derived one is a TENSOR size,
    # so the container's own bytes sit between them. Negligible against a real
    # artifact (~134 MB) and dominant against a small one, hence an absolute
    # allowance rather than a larger multiplier: widening the multiplier would
    # also widen the gap a partial materialisation could hide in.
    ceiling = int(required * tolerance) + container_allowance(required)

    evidence = {
        "size_bytes": size_bytes,
        "required_bytes": required,
        "materialised_bytes": materialised,
        "ceiling_bytes": ceiling,
        "ratio": round(materialised / required, 1) if required else None,
    }

    if size_bytes > ceiling:
        looks_materialised = size_bytes >= materialised * 0.5
        return CheckResult(
            CheckClass.ENVELOPE,
            CheckStatus.FAIL,
            (
                f"{size_bytes} bytes exceeds the ceiling of {ceiling} "
                + (
                    "and is within range of a MATERIALISED dictionary "
                    f"({materialised} bytes) — W_U J must never be formed"
                    if looks_materialised
                    else "for this model's dimensions"
                )
            ),
            evidence,
        )
    if size_bytes <= 0:
        return CheckResult(
            CheckClass.ENVELOPE, CheckStatus.FAIL, "artifact is empty", evidence
        )
    # A too-SMALL artifact is a truncation, and truncation loads fine.
    if size_bytes < required * 0.5:
        return CheckResult(
            CheckClass.ENVELOPE,
            CheckStatus.FAIL,
            f"{size_bytes} bytes is far below the {required} required — truncated?",
            evidence,
        )
    return CheckResult(
        CheckClass.ENVELOPE, CheckStatus.PASS, f"{size_bytes} within envelope", evidence
    )


def check_semantic(
    readout: Callable[[str, Sequence[int], int], Dict[int, Sequence[str]]],
    prompt: str,
    layers: Union[int, Sequence[int]],
    expected_intermediate: str,
    top_k: int = 8,
    control_prompt: Optional[str] = None,
) -> CheckResult:
    """A known UNSPOKEN intermediate is recovered SOMEWHERE in the stack.

    Deliberately an intermediate that appears in neither the prompt nor the
    output: a token present in the prompt can be recovered by an artifact that
    encodes nothing at all, so it would pass against a broken lens.

    SCANS THE FITTED LAYERS RATHER THAN ASSERTING ONE. The claim under test is
    that the lens surfaces an unspoken intermediate, not that it surfaces it at
    a particular depth — which depth is a property of the model, and this
    project may not assume one (BR-002 forbids a band constant, and "two thirds
    up" is a band constant wearing a different hat). Pinning a layer made the
    check fail on a lens that was working: on LFM2 the aligned artifact reads
    ' tourism'/' located'/' geography' at L9 for an Eiffel-Tower fixture — the
    concept field, with the specific token surfacing elsewhere in the stack.

    A scan is a WEAKER test than a single layer (more chances to hit), so it
    carries a matched control. `control_prompt` should be an unrelated prompt
    for which `expected_intermediate` would be an absurd continuation. If the
    token surfaces there too, the scan is not discriminative and this FAILS
    however well the real prompt did — a lens that answers ' France' to
    everything has told us nothing. Without a control prompt the check still
    runs, and says so, because the control is evidence and its absence must not
    read as having passed one.
    """
    if expected_intermediate.strip() and expected_intermediate.strip() in prompt:
        return CheckResult(
            CheckClass.SEMANTIC,
            CheckStatus.FAIL,
            (
                f"fixture is invalid: {expected_intermediate!r} appears in the "
                "prompt, so recovering it proves nothing"
            ),
        )

    scan = [int(layers)] if isinstance(layers, int) else [int(x) for x in layers]
    if not scan:
        return CheckResult(
            CheckClass.SEMANTIC,
            CheckStatus.FAIL,
            "no layers to scan; the artifact reports no fitted layers",
        )

    wanted = expected_intermediate.strip().lower()

    def _scan(text: str) -> Tuple[Optional[int], Optional[int], Dict[int, List[str]]]:
        """First (layer, rank) the token surfaces at, plus every top-k seen.

        ONE call for the whole scan, not one per layer. A readout is a forward
        pass with the residuals captured once and every requested layer read off
        it; asking layer by layer re-ran the model for each, turning a 25-layer
        artifact's check into 50 forward passes (the control doubles it).
        """
        tops = readout(text, scan, top_k)
        seen: Dict[int, List[str]] = {}
        for layer in scan:
            top = list(tops.get(layer, []))
            seen[layer] = top
            for rank, token in enumerate(top):
                if token.strip().lower() == wanted:
                    # Record the layers actually examined, then stop reporting.
                    return layer, rank, seen
        return None, None, seen

    try:
        hit_layer, hit_rank, seen = _scan(prompt)
    except Exception as exc:  # noqa: BLE001 - reported, not swallowed
        return CheckResult(
            CheckClass.SEMANTIC, CheckStatus.FAIL, f"readout raised: {exc}"
        )

    evidence: Dict[str, Any] = {
        "scanned_layers": scan,
        "top_k": top_k,
        "tops_by_layer": {str(k): v for k, v in seen.items()},
    }

    if hit_layer is None:
        return CheckResult(
            CheckClass.SEMANTIC,
            CheckStatus.FAIL,
            (
                f"{expected_intermediate!r} absent from the top-{top_k} at every "
                f"fitted layer {scan[0]}..{scan[-1]}"
            ),
            evidence,
        )

    evidence["layer"] = hit_layer
    evidence["rank"] = hit_rank
    evidence["top"] = seen[hit_layer]

    if control_prompt is None:
        evidence["control"] = "not supplied"
        return CheckResult(
            CheckClass.SEMANTIC,
            CheckStatus.PASS,
            (
                f"recovered {expected_intermediate!r} at layer {hit_layer} "
                f"(rank {hit_rank}); no control prompt supplied, so this scan's "
                "discriminative power was not measured"
            ),
            evidence,
        )

    try:
        ctrl_layer, ctrl_rank, ctrl_seen = _scan(control_prompt)
    except Exception as exc:  # noqa: BLE001 - reported, not swallowed
        return CheckResult(
            CheckClass.SEMANTIC, CheckStatus.FAIL, f"control readout raised: {exc}"
        )

    evidence["control"] = {
        "prompt": control_prompt,
        "tops_by_layer": {str(k): v for k, v in ctrl_seen.items()},
    }

    if ctrl_layer is not None:
        evidence["control_layer"] = ctrl_layer
        evidence["control_rank"] = ctrl_rank
        return CheckResult(
            CheckClass.SEMANTIC,
            CheckStatus.FAIL,
            (
                f"{expected_intermediate!r} surfaces for the UNRELATED control "
                f"prompt too (layer {ctrl_layer}, rank {ctrl_rank}), so recovering "
                f"it at layer {hit_layer} demonstrates nothing about the lens"
            ),
            evidence,
        )

    return CheckResult(
        CheckClass.SEMANTIC,
        CheckStatus.PASS,
        (
            f"recovered {expected_intermediate!r} at layer {hit_layer} "
            f"(rank {hit_rank}) and absent for the control prompt"
        ),
        evidence,
    )


def check_cross_implementation(
    ours: Sequence[str], theirs: Optional[Sequence[str]], top_k: int = 5
) -> CheckResult:
    """Our reader and the consumer's agree on the same prompt/layer/top-k.

    `theirs is None` means the comparison could not be made, which is NOT_RUN,
    not PASS. Treating an unreachable consumer as agreement is how a check
    designed to catch silent divergence becomes silent itself.
    """
    if theirs is None:
        return CheckResult(
            CheckClass.CROSS_IMPLEMENTATION,
            CheckStatus.NOT_RUN,
            "consumer unreachable; comparison not made",
        )
    a = [t.strip() for t in list(ours)[:top_k]]
    b = [t.strip() for t in list(theirs)[:top_k]]
    if a == b:
        return CheckResult(
            CheckClass.CROSS_IMPLEMENTATION,
            CheckStatus.PASS,
            f"top-{top_k} identical",
            {"top": a},
        )
    return CheckResult(
        CheckClass.CROSS_IMPLEMENTATION,
        CheckStatus.FAIL,
        f"top-{top_k} differs: ours={a} theirs={b}",
        {"ours": a, "theirs": b},
    )


def check_round_trip(served_readout: Optional[Sequence[str]]) -> CheckResult:
    """Mounted, served, and a Jacobian request came back with content.

    THE CHECK THAT CANNOT BE INFERRED FROM THE OTHERS. Everything upstream can
    pass in-process while the mounted artifact is never picked up, and the
    consumer says nothing about it. An empty result here is a FAIL, not an
    empty pass.
    """
    if served_readout is None:
        return CheckResult(
            CheckClass.ROUND_TRIP,
            CheckStatus.FAIL,
            "served request returned nothing; the artifact was not picked up",
        )
    if not [t for t in served_readout if t.strip()]:
        return CheckResult(
            CheckClass.ROUND_TRIP,
            CheckStatus.FAIL,
            "served readout is empty — indistinguishable from an unmounted artifact",
        )
    return CheckResult(
        CheckClass.ROUND_TRIP,
        CheckStatus.PASS,
        f"served {len(served_readout)} tokens",
        {"top": list(served_readout)},
    )
