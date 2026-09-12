"""
J-lens artifact lifecycle: discover, load, validate, publish.

THE FILESYSTEM IS THE REGISTRY, not a database table (PADR IDL-46). A J-lens
artifact is consumed by MOUNTING a conformant directory — there is no upload
path, and Neuronpedia's entire J-lens database footprint is two tables
persisting shared analysis sessions. Making a DB row the source of truth would
invent a second registry that can disagree with the one the consumer actually
reads, and the consumer's disagreement is silent.

    <root>/<slug>/<slug>_jacobian_lens.pt
    <root>/<slug>/config.yaml

PUBLISH ONLY AFTER VALIDATION (BR-030). `load_for_readout` refuses an artifact
whose validation did not pass every class, because the failure downstream is an
empty readout rather than an error — indistinguishable from a real readout with
no content, which is the same reason `/jlens/readout` refuses to fabricate one.

STAGE, THEN COMMIT. A fit writes to a staging directory and is moved into place
only once it validates. A half-written artifact in the mounted directory is
served: the loader is best-effort and says nothing about what it found.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from .jlens_validation import (
    CheckClass,
    CheckResult,
    CheckStatus,
    ValidationReport,
    check_envelope,
    check_naming,
    check_structural,
)

logger = logging.getLogger(__name__)

STAGING_SUFFIX = ".staging"

#: Where the artifact a commit REPLACED is kept. One slot, overwritten each
#: time: enough to undo the last mistake without letting 276 MB artifacts
#: accumulate silently. Excluded from discovery like staging is.
SUPERSEDED_SUFFIX = ".superseded"

#: The scratch name `restore_superseded` parks a displaced artifact under for
#: the duration of its three-way rename.
#:
#: EXCLUDED FROM DISCOVERY, like the other two. It briefly holds the only copy
#: of a live lens under a name the listing would serve — the method's own
#: docstring says parking a displaced artifact under any other name "would
#: publish it as a second, differently-slugged lens for the same model", and
#: then it did exactly that.
SWAP_SUFFIX = ".swap"

#: Verdict recorded beside the artifact at publish time. Named so it cannot be
#: mistaken for part of the conformance layout — a consumer reading the upstream
#: format ignores it, and `_ref_for` does not require it.
VALIDATION_FILE = "validation.json"

#: Results of interventions that were RUN against this lens, recorded beside it.
#:
#: THE POINT IS PORTABILITY. A lens is consumed by mounting its directory, so a
#: lens published to HuggingFace and pulled down by a serving runtime arrives as
#: files and nothing else. A result that lived only in a task record would not
#: make that journey, and the consumer would be left with a dictionary it can
#: read and no measurements of what happened when it was applied.
#:
#: THIS MODULE STORES; IT DOES NOT INTERPRET. Each record carries its own
#: `evidence_rung` and its own control, and the strength of a claim is a
#: property of the record, never of the fact that a record exists. Wording that
#: asserted otherwise here would put a verdict in a storage layer — which is why
#: the claims audit covers this file and does not exempt it.
#:
#: Named like `validation.json` so a reader of the upstream conformance format
#: ignores it — `_ref_for` does not require either, and neither is part of the
#: layout a third-party consumer parses.
INTERVENTION_FILE = "interventions.json"


def slug_for(repo_id: str) -> str:
    """The consumer's slug for a HuggingFace id.

    Mirrors the conformance spec's slug function. Kept here rather than inlined
    so the one place it is defined is the one place it can drift.
    """
    slug = repo_id.split("/")[-1].lower()
    slug = re.sub(r"[^a-z0-9._-]+", "-", slug).strip("-")
    if not slug:
        raise ValueError(f"{repo_id!r} produces an empty slug")
    return slug


@dataclass
class ArtifactRef:
    """A located artifact. Existence is not validity — see `validate`."""

    slug: str
    directory: Path
    lens_path: Path
    config_path: Optional[Path]

    @property
    def size_bytes(self) -> int:
        return self.lens_path.stat().st_size if self.lens_path.exists() else 0


class ArtifactNotValidated(RuntimeError):
    """Raised rather than serving an artifact that has not passed the suite."""


class ArtifactCoverageLoss(RuntimeError):
    """Raised rather than destroying layers the replacement does not cover."""


class ArtifactConflict(RuntimeError):
    """Raised rather than clearing a path that may hold the only copy of a lens.

    The registry is the filesystem (PADR IDL-46), so debris from an interrupted
    rename is not a stale row to be tidied — it is data. Refusing costs a manual
    step; clearing cost the artifact.
    """


class ArtifactQualityRegression(RuntimeError):
    """Raised rather than replacing stronger evidence with weaker.

    Publishing is otherwise LAST WRITER WINS, and "last" means "finished
    last", not "best" or even "newest". Observed on 2026-08-04: a 400-prompt
    fit that never converged published over a 1097-prompt fit that did. The
    weaker job had been queued HOURS EARLIER, sat unclaimed in Redis through a
    series of pod rolls, and ran when the queue finally drained — so it was
    older by intent and newer by completion, and nothing compared the two.
    """


def _dtype_bytes(payload: Optional[Dict[Any, Any]]) -> int:
    """Bytes per element of the matrices actually on disk.

    Falls back to 2 when the payload could not be read — the envelope check is
    then bounded on the assumption every local fit satisfies, and STRUCTURAL has
    already failed for the unreadable case anyway.
    """
    if not payload:
        return 2
    try:
        total = sum(t.numel() * t.element_size() for t in payload.values())
        elements = sum(t.numel() for t in payload.values())
    except AttributeError:
        return 2
    if not elements:
        return 2
    # THE EXACT FIGURE, not the widest dtype present. `validate` is also called
    # on arbitrary mounted artifacts with no mixed-dtype gate, and taking the
    # max there doubles both the publication ceiling and the materialisation
    # threshold for a single stray fp32 tensor — weakening, in the permissive
    # direction, the one check that gates publication, for exactly a
    # nonconformant file. Both figures are already in hand.
    return max(1, round(total / elements))


class PayloadShapeError(ValueError):
    """The checkpoint is not a shape this project knows how to read."""


def normalise_payload(obj: Any) -> Dict[int, torch.Tensor]:
    """Both on-disk shapes to `{layer: matrix}`. Refuses anything else BY NAME.

    TWO SHAPES EXIST AND BOTH ARE REAL.

    The conformance spec (`0xcc/brds/neuronpedia-jlens-conformance.md` §2.2) —
    which the reference implementation and every lens published to HuggingFace
    follow — is a WRAPPER::

        {"J": {layer: Tensor}, "source_layers": [...], "n_prompts": int, "d_model": int}

    while every artifact this project has written so far is the bare map
    `{layer: Tensor}` (`write_staged`). Reading only the wrapper would strand the
    existing registry; reading only the bare map is what made a published lens
    fail with `ValueError: invalid literal for int() with base 10: 'J'`.

    IT REFUSES RATHER THAN GUESSES. A wrong unwrap yields a dict of square
    matrices that passes STRUCTURAL, ENVELOPE and NAMING and reads out plausible
    nonsense — the failure the whole validation suite exists to catch, arriving
    through the one door that suite does not watch. Every refusal names the keys
    it actually found, because "unrecognised checkpoint" sends a reader to the
    wrong file.
    """
    if not isinstance(obj, dict):
        raise PayloadShapeError(
            f"checkpoint is a {type(obj).__name__}, not a dict; expected either "
            f'{{"J": {{layer: matrix}}}} or {{layer: matrix}}'
        )
    if not obj:
        raise PayloadShapeError("checkpoint is an empty dict; there are no layers in it")

    def _int_like(k: Any) -> bool:
        try:
            int(k)
        except (TypeError, ValueError):
            return False
        return True

    has_wrapper = "J" in obj
    int_keys = [k for k in obj if _int_like(k)]

    # AMBIGUOUS IS A REFUSAL. A file carrying both a "J" block and loose layer
    # keys has two candidate lenses in it, and picking either one is a guess
    # about which the publisher meant.
    if has_wrapper and int_keys:
        raise PayloadShapeError(
            f'checkpoint has BOTH a "J" block and loose layer keys {sorted(int_keys)[:8]}; '
            "which of the two is the lens is not knowable from the file"
        )

    if has_wrapper:
        inner = obj["J"]
        if not isinstance(inner, dict) or not inner:
            raise PayloadShapeError(
                f'"J" is a {type(inner).__name__}, not a non-empty dict of layer matrices'
            )
        try:
            layers = {int(k): v for k, v in inner.items()}
        except (TypeError, ValueError) as exc:
            raise PayloadShapeError(
                f'"J" has keys that are not layer indices: {sorted(map(str, inner))[:8]}'
            ) from exc

        # A1: `source_layers`, when present, must EQUAL the key set. A
        # disagreement means the file was assembled from parts and one of the
        # two is stale — and the stale one might be the matrices.
        declared = obj.get("source_layers")
        if declared is not None:
            try:
                declared_set = {int(x) for x in declared}
            except (TypeError, ValueError) as exc:
                raise PayloadShapeError(
                    f"source_layers is not a list of layer indices: {declared!r}"
                ) from exc
            if declared_set != set(layers):
                raise PayloadShapeError(
                    f"source_layers {sorted(declared_set)} does not match the layers "
                    f"actually present {sorted(layers)}"
                )
        return layers

    if len(int_keys) != len(obj):
        stray = sorted(str(k) for k in obj if not _int_like(k))
        raise PayloadShapeError(
            f"checkpoint keys are not layer indices: {stray[:8]}"
            + (f" (+{len(stray) - 8} more)" if len(stray) > 8 else "")
        )
    return {int(k): obj[k] for k in obj}


class JLensArtifactService:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    # ---------------------------------------------------------- discovery

    def list_artifacts(self) -> List[ArtifactRef]:
        """Every conformant artifact under the root.

        Staging directories are excluded: an artifact still being written is
        not an artifact, and the whole point of staging is that it is invisible
        until it commits.
        """
        if not self.root.is_dir():
            return []
        found: List[ArtifactRef] = []
        for directory in sorted(p for p in self.root.iterdir() if p.is_dir()):
            if directory.name.endswith(
                (STAGING_SUFFIX, SUPERSEDED_SUFFIX, SWAP_SUFFIX)
            ):
                continue
            ref = self._ref_for(directory)
            if ref is not None:
                found.append(ref)
        return found

    def find(self, repo_id: str) -> Optional[ArtifactRef]:
        directory = self.root / slug_for(repo_id)
        return self._ref_for(directory) if directory.is_dir() else None

    def _ref_for(self, directory: Path) -> Optional[ArtifactRef]:
        lens_files = [p for p in directory.glob("*_jacobian_lens.pt")]
        if len(lens_files) != 1:
            # Zero is "not an artifact"; more than one is ambiguous, and the
            # consumer picks among them without saying which.
            return None
        config = directory / "config.yaml"
        return ArtifactRef(
            slug=directory.name,
            directory=directory,
            lens_path=lens_files[0],
            config_path=config if config.exists() else None,
        )

    # --------------------------------------------------------- validation

    def validate(
        self,
        ref: ArtifactRef,
        d_model: int,
        expected_layers: Sequence[int],
        n_vocab: int,
        semantic_result: Optional[CheckResult] = None,
        cross_impl_result: Optional[CheckResult] = None,
        round_trip_result: Optional[CheckResult] = None,
    ) -> ValidationReport:
        """Run every class. The three that need a live consumer are INJECTED.

        They are parameters rather than internal calls because they cannot be
        performed from here: SEMANTIC needs a loaded model, and
        CROSS_IMPLEMENTATION and ROUND_TRIP need a running consumer. Passing
        `None` records them as NOT_RUN, and `ValidationReport.passed` is
        fail-closed — so an artifact validated without them can never be
        published, rather than appearing to have passed a suite it never ran.
        """
        results: List[CheckResult] = [check_naming(ref.directory)]

        payload = self._load_payload(ref)
        if payload is None:
            results.append(
                CheckResult(
                    CheckClass.STRUCTURAL,
                    CheckStatus.FAIL,
                    f"{ref.lens_path} did not deserialize with weights-only loading",
                )
            )
        else:
            results.append(check_structural(payload, d_model, expected_layers))

        results.append(
            check_envelope(
                ref.size_bytes,
                d_model=d_model,
                n_layers=len(list(expected_layers)),
                n_vocab=n_vocab,
                # DERIVED FROM THE PAYLOAD THIS METHOD ALREADY LOADED, not
                # taken as an argument. A parameter defaulting to 2 is right for
                # every artifact this project FITS — the fitter writes fp16 —
                # and silently wrong for an acquired fp32 one, which is the
                # first path where a non-fp16 artifact can arrive. A caller that
                # forgets to pass it gets the wrong ceiling and no error, so
                # there is no parameter to forget.
                dtype_bytes=_dtype_bytes(payload),
            )
        )

        for supplied, check in (
            (semantic_result, CheckClass.SEMANTIC),
            (cross_impl_result, CheckClass.CROSS_IMPLEMENTATION),
            (round_trip_result, CheckClass.ROUND_TRIP),
        ):
            results.append(
                supplied
                if supplied is not None
                else CheckResult(
                    check,
                    CheckStatus.NOT_RUN,
                    "not supplied; this check requires a loaded model or a live consumer",
                )
            )

        return ValidationReport(results)

    def _load_payload(self, ref: ArtifactRef) -> Optional[Dict[Any, Any]]:
        """Weights-only deserialisation, NORMALISED to `{layer: matrix}`.

        `weights_only=True` is not a nicety: an artifact is an untrusted file on
        disk that this process is about to load, and the unrestricted loader
        executes pickled code.

        NORMALISED HERE BECAUSE THIS IS THE ONLY DESERIALISATION POINT. Six
        readers sit downstream of it — `validate`, `check_structural`,
        `_coverage_delta`, `load_for_readout`, both semantic-check sites and the
        endpoint's `sorted(int(k) for k in payload)`. Teaching the wrapper shape
        to one of them and not the others is how a file that is on disk and
        published still raises `ValueError('J')` somewhere. Same argument
        `slug_for` makes for itself: the one place it is defined is the one place
        it can drift.
        """
        try:
            raw = torch.load(ref.lens_path, map_location="cpu", weights_only=True)
        except Exception as exc:  # noqa: BLE001 - reported as a FAIL, not swallowed
            logger.warning("J-lens artifact %s failed to load: %s", ref.lens_path, exc)
            return None
        try:
            return normalise_payload(raw)
        except PayloadShapeError as exc:
            logger.warning("J-lens artifact %s has an unusable shape: %s", ref.lens_path, exc)
            return None

    # ------------------------------------------------------------ publish

    def staging_dir(self, repo_id: str) -> Path:
        return self.root / f"{slug_for(repo_id)}{STAGING_SUFFIX}"

    def _refuse_occupied_staging(self, repo_id: str, replace_staged: bool) -> None:
        """Staging is not scratch space when something conformant is already in it.

        A fit or an acquisition that validated but was refused by a gate is
        deliberately KEPT there, and the documented recovery is to re-run with a
        flag. This project once destroyed a converged 15-layer LFM2 artifact —
        754 seconds of GPU time — by treating staging as disposable, and the
        first version of this guard covered only the acquisition path, leaving
        the FIT path (the one that caused that incident) still clearing
        unconditionally.
        """
        existing = self.staging_dir(repo_id)
        if replace_staged or not existing.is_dir():
            return
        if self._ref_for(existing) is None:
            return
        raise ArtifactConflict(
            f"{existing.name} already holds a staged artifact for this model. "
            "It may be completed work that a gate refused; inspect it, or pass "
            "replace_staged to overwrite it deliberately."
        )

    def _fresh_staging(self, repo_id: str) -> Tuple[str, Path, Path]:
        """A cleared staging directory, its slug, and the lens path to write.

        THE LAYOUT INVARIANT, IN ONE PLACE. `check_naming` FAILs a directory
        that does not hold exactly one `<slug>_jacobian_lens.pt`, and the slug
        must be the one `slug_for` produces for the model — a second
        implementation of that rule, sitting next to the check whose job is to
        catch violations of it, is the shape this project keeps writing
        post-mortems about.

        THE NAME COMES FROM THE MODEL, NEVER FROM AN UPSTREAM FILE. `slug_for`
        lowercases and `LENS_FILENAME` is anchored lowercase, while a published
        lens preserves the HuggingFace case — so `Qwen/Qwen3-8B` is published as
        `Qwen3-8B_jacobian_lens.pt`, which fails NAMING. Keeping the upstream
        stem would break on exactly the models whose ids carry capitals, and
        pass on every lowercase one that gets tested first.
        """
        slug = slug_for(repo_id)
        staging = self.staging_dir(repo_id)
        if staging.exists():
            shutil.rmtree(staging)
        staging.mkdir(parents=True)
        return slug, staging, staging / f"{slug}_jacobian_lens.pt"

    def stage_from_file(
        self,
        repo_id: str,
        lens_source: Path,
        config_yaml: str,
        sidecars: Optional[Dict[str, Path]] = None,
        replace_staged: bool = False,
    ) -> ArtifactRef:
        """Stage bytes that ALREADY EXIST, without deserialising them.

        For an acquired artifact the file is the evidence: copying it verbatim
        means `local_sha256 == upstream_sha256` is a fact a third party can
        check against the source repo. Re-saving the tensors would destroy the
        only cryptographic identity an acquired lens has and replace it with
        "trust us, these are the same numbers" — a poor trade in a system whose
        existing weight-identity check is a string comparison that never opens
        the file.

        `sidecars` maps target filename to source path, for things like the
        convergence trace. A `.pt` sidecar is REFUSED: `check_naming` fails a
        directory with a second `.pt`, and a consumer globbing
        `*_jacobian_lens.pt` picks among multiple matches silently.
        """
        self._refuse_occupied_staging(repo_id, replace_staged)
        _slug, staging, lens_path = self._fresh_staging(repo_id)
        shutil.copyfile(lens_source, lens_path)
        (staging / "config.yaml").write_text(config_yaml)

        for name, source in (sidecars or {}).items():
            if name.endswith(".pt"):
                raise ValueError(
                    f"{name!r} is a second .pt in the artifact directory; NAMING "
                    "fails on that and a consumer would pick between them silently"
                )
            shutil.copyfile(source, staging / name)

        ref = self._ref_for(staging)
        if ref is None:
            raise RuntimeError(
                f"staged {staging} is not conformant after copying {lens_source}"
            )
        return ref

    def write_staged(
        self,
        repo_id: str,
        jacobians: Dict[int, torch.Tensor],
        config_yaml: str,
        replace_staged: bool = True,
        n_prompts: int = 0,
    ) -> ArtifactRef:
        """Write a fit into staging, where nothing will serve it.

        `replace_staged` DEFAULTS TRUE HERE, and that is the historical
        behaviour rather than a considered one: the fit worker banks on staging
        being cleared before the next fit. It is a parameter so a caller that
        cares can say otherwise — and so the two writers share one rule instead
        of the acquisition path being the only one that checks.

        The asymmetry is real and worth stating: a re-fit overwriting its own
        staged output is the normal loop, while an ACQUISITION landing on staged
        work belongs to a different operation entirely.
        """
        self._refuse_occupied_staging(repo_id, replace_staged)
        slug, staging, lens_path = self._fresh_staging(repo_id)
        # SAVE ON CPU, ALWAYS. `torch.save` records each tensor's device, and a
        # fit runs on the GPU — so an artifact written straight from a fit is
        # tagged cuda:0 and raises "Attempting to deserialize object on a CUDA
        # device" for any consumer without one. A J-lens artifact is a PORTABLE
        # DOCUMENT whose whole purpose is to be mounted and read elsewhere; a
        # file that only loads on the machine that produced it is not one.
        #
        # Our own loader passes map_location="cpu" and would never have noticed.
        # THE CONFORMANT WRAPPER, NOT A BARE MAP.
        #
        # Spec §2.2 is explicit that a consumer reads `payload["J"]` and that
        # "absence of `J` raises with the offending key list". This project wrote
        # the bare `{layer: matrix}` form, so EVERY artifact it has ever produced
        # would fail to load for anyone who downloaded it — which made publishing
        # them pointless and was invisible locally, because our own reader
        # accepted the bare form and nothing else ever read one.
        #
        # Safe to change only because `normalise_payload` now accepts both: every
        # artifact already on disk keeps working, and new ones are portable.
        #
        # `d_model` is the one other field a consumer reads without a fallback.
        # `source_layers` must EQUAL the key set or A1 fails, so it is derived
        # here rather than passed in.
        layers = {int(k): v.detach().to("cpu") for k, v in jacobians.items()}
        widths = {int(t.shape[0]) for t in layers.values()}
        if len(widths) != 1:
            raise ValueError(
                f"matrices have differing widths {sorted(widths)}; a lens has one "
                "d_model and a consumer reads it without a fallback"
            )
        torch.save(
            {
                "J": layers,
                "d_model": widths.pop(),
                "source_layers": sorted(layers),
                "n_prompts": int(n_prompts),
            },
            lens_path,
        )
        (staging / "config.yaml").write_text(config_yaml)

        return ArtifactRef(
            slug=slug,
            directory=staging,
            lens_path=lens_path,
            config_path=staging / "config.yaml",
        )

    def commit(
        self,
        repo_id: str,
        report: ValidationReport,
        allow_coverage_loss: bool = False,
        allow_quality_regression: bool = False,
    ) -> ArtifactRef:
        """Move a staged artifact into the mounted directory.

        REFUSES on anything short of a full pass. The mounted directory is read
        by a consumer that reports no errors, so this is the last point at which
        a bad artifact can be stopped by anything at all.
        """
        if not report.passed:
            raise ArtifactNotValidated(
                f"refusing to publish {repo_id}: {report.summary()}. "
                "The consumer fails silently, so an unvalidated artifact "
                "presents as a feature that quietly returns nothing."
            )

        staging = self.staging_dir(repo_id)
        if not staging.is_dir():
            raise FileNotFoundError(f"nothing staged for {repo_id} at {staging}")

        # REFUSE A SILENT LOSS OF COVERAGE. A refit is not automatically an
        # upgrade: the artifact this destroyed covered 16 layers on 120 prompts
        # and the replacement covered 9 on 400 — neither dominates, and nothing
        # told the user they were about to lose seven layers. Losing coverage
        # must be a DECISION, so it is refused unless asked for by name.
        lost, out_of_scope = self._coverage_delta(repo_id, staging)
        if lost and not allow_coverage_loss:
            raise ArtifactCoverageLoss(
                f"refusing to publish {repo_id}: the existing artifact covers "
                f"layers {lost} that this fit does not. Publishing would "
                "destroy them. Re-run with allow_coverage_loss=true if that is "
                "what you want, or fit the missing layers as well."
            )
        if out_of_scope:
            # NOT A LOSS — A RECIPE CHANGE. A layer above the new target has no
            # Jacobian to that target: the path is zero by causality, and the
            # fitter refuses to fit it at all. The old artifact holds those
            # layers only because it targeted a higher block.
            #
            # This used to raise. The refusal told the user to "fit the missing
            # layers as well", which is IMPOSSIBLE under the new target, so a
            # penultimate refit over a final-target artifact could not be
            # published by any action the message suggested. The previous
            # artifact is archived to `.superseded`, so nothing is unrecoverable.
            logger.info(
                "Publishing %s drops layers %s, which are above its %s target and "
                "cannot be fitted under this recipe. The previous artifact is "
                "archived, not deleted.",
                repo_id,
                out_of_scope,
                self.target_layer(self._ref_for(staging)) or "declared",
            )

        regression = self._quality_regression(repo_id, staging)
        if regression and not allow_quality_regression:
            raise ArtifactQualityRegression(
                f"refusing to publish {repo_id}: {regression}. The staged fit is "
                "kept; re-run with allow_quality_regression=true to publish it "
                "anyway."
            )

        final = self.root / slug_for(repo_id)
        if final.exists():
            # ARCHIVE, DO NOT DELETE. This used to be `shutil.rmtree(final)`,
            # and it destroyed a full 16-layer LFM2 lens when a later 9-layer
            # fit published over it — nine minutes of GPU and the reference
            # model's only full-stack artifact, gone with no warning and no way
            # back. One slot, overwritten each time: enough to undo the last
            # mistake without letting 276 MB artifacts pile up unnoticed.
            archive = self.root / f"{slug_for(repo_id)}{SUPERSEDED_SUFFIX}"
            if archive.exists():
                shutil.rmtree(archive)
            final.rename(archive)
            logger.info("Archived the previous %s artifact to %s", repo_id, archive)
        staging.rename(final)

        ref = self._ref_for(final)
        if ref is None:
            raise RuntimeError(f"published {final} is not a conformant artifact directory")

        # RECORD THE VERDICT WITH THE ARTIFACT. The filesystem is the registry
        # (PADR IDL-46), so the report belongs beside the file it describes and
        # not in a database that could disagree with it.
        #
        # Without this the fit's validation was DISCARDED and the readout
        # re-validated from scratch with its own hard-coded fixture — one that
        # assumes a mid-stack fit, so a legitimately validated PARTIAL artifact
        # was refused at read time. The caller chose a fixture appropriate to
        # the layers they fitted; substituting a different one and overruling
        # them is not a stricter check, it is a different question.
        self._write_report(ref, report)
        logger.info("Published J-lens artifact %s", final)
        return ref

    @staticmethod
    def _lens_digest(path: Path) -> str:
        """Content hash of the lens file.

        SIZE AND MTIME ARE NOT AN IDENTITY. A replacement with the same layer
        shapes has the same size, and mtime granularity is coarse enough that a
        file rewritten immediately keeps its timestamp — verified by the test
        below, which failed against the size+mtime version of this check. The
        thing being guarded is "are these the weights that were validated", so
        the guard has to look at the weights.
        """
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _coverage_delta(self, repo_id: str, staging: Path) -> Tuple[List[int], List[int]]:
        """Layers the CURRENT artifact has that the staged one does not, SPLIT.

        Returns `(lost, out_of_scope)`:

          * `lost` — layers the new fit could have covered and did not. Real
            loss, and the thing worth refusing over.
          * `out_of_scope` — layers ABOVE the staged fit's target. Their
            Jacobian to that target is zero by causality and the fitter refuses
            to fit them, so their absence is the recipe, not a gap.

        Both empty when there is no current artifact or either side is
        unreadable. Unreadable is treated as "nothing to lose" deliberately: a
        guard that blocks publishing because it could not parse the old file
        turns a corrupt artifact into a permanent obstruction.
        """
        current = self.find(repo_id)
        if current is None:
            return [], []
        staged = self._ref_for(staging)
        if staged is None:
            return [], []
        old = self._load_payload(current)
        new = self._load_payload(staged)
        if not isinstance(old, dict) or not isinstance(new, dict):
            return [], []
        missing = sorted({int(k) for k in old} - {int(k) for k in new})
        if not missing:
            return [], []

        ceiling = self._target_index(staged)
        if ceiling is None:
            # No recipe to appeal to, so nothing may be excused. Fail closed:
            # treating an unreadable recipe as permission to drop layers is how
            # a guard becomes decorative.
            return missing, []
        lost = [l for l in missing if l <= ceiling]
        out_of_scope = [l for l in missing if l > ceiling]
        return lost, out_of_scope

    def _quality_regression(self, repo_id: str, staging: Path) -> Optional[str]:
        """Why publishing `staging` would replace stronger evidence with weaker.

        Two comparisons, both read from the recipes the fits wrote themselves:

          * CONVERGENCE. A fit that reached its threshold is stronger evidence
            than one that ran out of corpus. This is the one that bit.
          * CORPUS SIZE, but only between fits of the SAME convergence status.
            A converged fit over 400 prompts is not worse than a converged fit
            over 1200; it got there sooner.

        Returns None when either recipe is unreadable. Unlike the coverage
        guard this fails OPEN, deliberately: coverage protects layers a user
        already paid GPU time for, where an unknown must not be discarded on a
        guess. Here an unreadable incumbent is not evidence worth defending,
        and failing closed would make an artifact with a corrupt config
        impossible to ever replace.
        """
        current = self.find(repo_id)
        if current is None:
            return None
        staged = self._ref_for(staging)
        if staged is None:
            return None

        cur_converged = self._config_bool(current, "converged")
        new_converged = self._config_bool(staged, "converged")
        cur_n = self._config_int(current, "n_prompts")
        new_n = self._config_int(staged, "n_prompts")

        if cur_converged is True and new_converged is False:
            return (
                f"the published artifact CONVERGED (over {cur_n} prompts) and this "
                f"fit did not (it stopped at {new_n}). A fit that ran out of corpus "
                "is weaker evidence than one that reached its threshold"
            )

        if (
            cur_converged == new_converged
            and cur_n is not None
            and new_n is not None
            and new_n < cur_n
        ):
            return (
                f"this fit saw {new_n} prompts and the published artifact saw "
                f"{cur_n}, with neither converging differently — publishing would "
                "replace a better-supported lens with a less-supported one"
            )

        return None

    def _config_bool(self, ref: ArtifactRef, key: str) -> Optional[bool]:
        """One boolean field from config.yaml, or None if absent/unparseable.

        None is NOT False. "we could not read whether it converged" and "it did
        not converge" are different facts, and collapsing them would let an
        unreadable recipe read as a failed one.
        """
        if ref.config_path is None or not ref.config_path.is_file():
            return None
        try:
            for raw in ref.config_path.read_text().splitlines():
                name, _, value = raw.partition(":")
                if name.strip() == key:
                    got = value.strip().lower()
                    if got in ("true", "yes"):
                        return True
                    if got in ("false", "no"):
                        return False
                    return None
        except OSError as exc:  # noqa: BLE001
            logger.warning("Could not read %s from %s: %s", key, ref.config_path, exc)
        return None

    def _target_index(self, ref: ArtifactRef) -> Optional[int]:
        """Highest layer this artifact's recipe COULD cover, from its own config.

        `None` when either the target or the layer count is unreadable — the
        caller must then excuse nothing.
        """
        target = self.target_layer(ref)
        n_layers = self._config_int(ref, "n_layers")
        if target is None or n_layers is None:
            return None
        return n_layers - 2 if target == "penultimate" else n_layers - 1

    def _config_int(self, ref: ArtifactRef, key: str) -> Optional[int]:
        """One integer field from config.yaml, or None if absent/unparseable."""
        if ref.config_path is None or not ref.config_path.is_file():
            return None
        try:
            for raw in ref.config_path.read_text().splitlines():
                name, _, value = raw.partition(":")
                if name.strip() == key:
                    return int(value.strip())
        except (OSError, ValueError) as exc:  # noqa: BLE001
            logger.warning("Could not read %s from %s: %s", key, ref.config_path, exc)
        return None

    def _write_report(self, ref: ArtifactRef, report: ValidationReport) -> None:
        stat = ref.lens_path.stat()
        payload = {
            "lens_file": ref.lens_path.name,
            "size_bytes": stat.st_size,
            "sha256": self._lens_digest(ref.lens_path),
            "summary": report.summary(),
            "passed": report.passed,
            "serviceable": report.serviceable,
            "results": [
                {"check": r.check.value, "status": r.status.value, "detail": r.detail}
                for r in report.results
            ],
        }
        (ref.directory / VALIDATION_FILE).write_text(json.dumps(payload, indent=2))

    def stored_report(self, ref: ArtifactRef) -> Optional[Dict[str, Any]]:
        """The verdict recorded when THIS EXACT FILE was published, if any.

        Returns None when the lens file has changed since — size and mtime are
        compared, so an artifact swapped on disk is revalidated rather than
        served on a verdict that described different weights. Serving a lens
        fitted for other weights is the failure this gate exists to prevent, so
        it must not be possible to smuggle one past by leaving a stale JSON
        file beside it.
        """
        path = ref.directory / VALIDATION_FILE
        if not path.is_file():
            return None
        try:
            stored = json.loads(path.read_text())
        except (ValueError, OSError) as exc:
            logger.warning("Unreadable validation report at %s: %s", path, exc)
            return None

        if stored.get("sha256") != self._lens_digest(ref.lens_path):
            logger.info(
                "Validation report for %s describes different weights; revalidating",
                ref.slug,
            )
            return None
        return stored

    def record_intervention_result(self, repo_id: str, record: Dict[str, Any]) -> None:
        """Append one intervention result to this artifact's records.

        BOUND TO THE WEIGHTS THAT WERE TESTED. The lens digest is stored with
        each record and checked on read, for the same reason `validation.json`
        carries one: a refit replaces the matrices, and evidence describing the
        previous lens travelling beside the new one is worse than no evidence —
        it is a claim about a file that no longer exists.

        REPLACED BY RECIPE, not appended blindly. Re-running the same
        experiment — same primitive, direction, layers and strength — supersedes
        its previous record rather than accumulating near-duplicates that a
        reader would have to date-sort to interpret.
        """
        ref = self.find(repo_id)
        if ref is None:
            raise FileNotFoundError(f"no published artifact for {repo_id!r}")

        record = dict(record)
        record["lens_sha256"] = self._lens_digest(ref.lens_path)

        existing = self._read_interventions(ref)
        key = self._recipe_key(record)
        kept = [r for r in existing if self._recipe_key(r) != key]
        kept.append(record)
        # ATOMIC. `write_text` truncates first, so an eviction mid-write leaves
        # invalid JSON — and `_read_interventions` fails to `[]`, so the next
        # successful record writes a one-element list over the wreckage. Total,
        # permanent, invisible loss.
        #
        # `validation.json` shares the non-atomic write and does not need this:
        # a verdict is idempotently regenerable, and a corrupt one triggers
        # revalidation by design. An intervention record is a GPU run. It is not
        # regenerable by anything.
        target = ref.directory / INTERVENTION_FILE
        tmp = target.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(kept, indent=2))
        tmp.replace(target)
        logger.info(
            "Recorded an intervention result for %s (%s), %d record(s) total",
            ref.slug,
            key,
            len(kept),
        )

    def intervention_results(self, ref: ArtifactRef) -> List[Dict[str, Any]]:
        """Records that describe THIS lens file. Others are dropped.

        A record whose digest does not match is not merely stale — it describes
        different weights, and reporting it would attribute one lens's measured
        behaviour to another.
        """
        current = self._lens_digest(ref.lens_path)
        kept = []
        for record in self._read_interventions(ref):
            if record.get("lens_sha256") == current:
                kept.append(record)
            else:
                logger.info(
                    "Dropping a record for %s: it describes different weights",
                    ref.slug,
                )
        return kept

    def _read_interventions(self, ref: ArtifactRef) -> List[Dict[str, Any]]:
        path = ref.directory / INTERVENTION_FILE
        if not path.is_file():
            return []
        try:
            loaded = json.loads(path.read_text())
        except (ValueError, OSError) as exc:
            logger.warning("Unreadable intervention records at %s: %s", path, exc)
            return []
        return loaded if isinstance(loaded, list) else []

    @staticmethod
    def _recipe_key(record: Dict[str, Any]) -> str:
        """What makes two intervention runs the SAME experiment.

        The direction, where it was applied, and how hard. Two runs differing in
        any of these are different experiments and both are worth keeping; two
        runs agreeing on all of them are the same experiment run twice, and only
        the later one is informative.
        """
        recipe = record.get("steering_recipe") or {}
        # TARGET AND POSITIONS ARE PART OF THE EXPERIMENT. A coordinate_swap IS
        # the pair — swapping dog with cat is not the run that swapped dog with
        # pet — and the panel picks the partner from whatever is pinned, so two
        # genuinely different swaps shared a key and the earlier one, a completed
        # GPU measurement, was deleted from the file whose whole purpose is
        # carrying evidence off this machine.
        return "|".join(
            str(recipe.get(field))
            for field in (
                "primitive",
                "direction_token",
                "target_token",
                "layers",
                "positions",
                "strength",
                # AND THE TRIAL SET. It is the most obvious variable in the
                # experiment and was the one missing: a 50-prompt run and a
                # one-prompt click agreed on every other field, so the click
                # evicted the 50-prompt record and left the artifact telling a
                # consumer the direction moves nothing.
                "prompts_sha256",
            )
        )

    def restore_superseded(self, slug: str) -> Dict[str, Any]:
        """Promote `<slug>.superseded` back to `<slug>`, archiving the incumbent.

        A SWAP, NOT A MOVE, so the operation is its own undo: call it twice and
        you are back where you started. Nothing is deleted at any point.

        A swap is also the only shape that keeps both directories WITHOUT
        inventing a third artifact. Discovery skips exactly `.staging` and
        `.superseded`, so parking the displaced one under any other name would
        publish it as a second, differently-slugged lens for the same model.

        WHY THIS EXISTS. Publishing is last-writer-wins, and on 2026-08-04 a
        stale 400-prompt fit that never converged published over a 1097-prompt
        fit that did. `_quality_regression` now refuses that, but an artifact
        already displaced needed a shell rename inside the pod to recover — an
        operation with no audit trail, no digest check, and nothing stopping a
        typo from destroying the archive.

        The restored artifact's recorded verdict is verified against the file
        it describes before promotion. A `.superseded` directory is not
        privileged: it is exactly as untrusted as any other artifact on disk,
        and serving a lens on a verdict that described different weights is the
        failure the whole publish gate exists to prevent.
        """
        archived_dir = self.root / f"{slug}{SUPERSEDED_SUFFIX}"
        archived = self._ref_for(archived_dir) if archived_dir.is_dir() else None
        if archived is None:
            raise FileNotFoundError(
                f"there is no archived artifact for {slug!r} at {archived_dir}"
            )

        if self.stored_report(archived) is None:
            raise ArtifactNotValidated(
                f"the archived artifact for {slug!r} carries no verdict matching "
                "its own weights, so promoting it would serve a lens on a "
                "validation that described a different file"
            )

        current_dir = self.root / slug
        restored_recipe = self._recipe_summary(archived)

        if current_dir.is_dir():
            displaced_ref = self._ref_for(current_dir)
            displaced_recipe = (
                self._recipe_summary(displaced_ref) if displaced_ref else {}
            )
            swap = self.root / f"{slug}{SWAP_SUFFIX}"
            # REFUSED, NOT CLEARED. A leftover swap directory is the debris of
            # an interrupted rename, which means it may be the ONLY copy of a
            # lens. `rmtree` here made the recovery operation the thing that
            # destroyed what was being recovered.
            if swap.exists():
                raise ArtifactConflict(
                    f"{swap.name} already exists. That is the debris of an "
                    "interrupted restore and may hold the only copy of a lens; "
                    "inspect it and move it aside by hand before retrying."
                )
            current_dir.rename(swap)
            archived_dir.rename(current_dir)
            swap.rename(archived_dir)
        else:
            # Nothing to archive; a plain promotion.
            displaced_recipe = {}
            archived_dir.rename(current_dir)

        logger.info(
            "Restored %s from its archive (%s), displacing (%s)",
            slug,
            restored_recipe,
            displaced_recipe,
        )
        return {
            "slug": slug,
            "restored": restored_recipe,
            "displaced": displaced_recipe,
        }

    def _recipe_summary(self, ref: ArtifactRef) -> Dict[str, Any]:
        """The few recipe fields that decide whether one lens beats another."""
        return {
            "n_prompts": self._config_int(ref, "n_prompts"),
            "converged": self._config_bool(ref, "converged"),
            "target_layer": self.target_layer(ref),
            "fitted_layers": len(self.fitted_layers(ref)),
        }

    # There is deliberately no `discard_staged`. It existed, and its only caller
    # deleted a converged artifact the moment a fixture failed — the expensive
    # half of the work thrown away to save a directory that `write_staged`
    # clears anyway on the next fit. A failed validation leaves the staged fit
    # in place so it can be re-validated for free.

    # ------------------------------------------------------------- serving

    def load_for_readout(
        self, repo_id: str, report: Optional[ValidationReport] = None
    ) -> Dict[int, torch.Tensor]:
        """Tensors for `JacobianTransport`, only if validation is SERVICEABLE.

        `report=None` is refused rather than defaulted to trusting the file.
        Serving an unvalidated artifact is precisely the failure BR-030 exists
        for, and it surfaces as an empty readout rather than an error.

        Gated on `serviceable`, not `passed`: the two consumer-interop classes
        need a live external consumer, so requiring them here would make the
        Jacobian path unreachable from miStudio itself. `commit` still requires
        the full `passed` before anything is published for handover.
        """
        if report is None or not report.serviceable:
            # NAME THE FAILING CLASS. "No serviceable validation report" is true
            # and useless: it does not distinguish a missing report from a
            # failed check, and it took a log dive plus two wrong diagnoses to
            # learn which one had happened. A refusal the user cannot act on is
            # only half a guard.
            why = ""
            if report is not None:
                detail = getattr(report, "failing_detail", None)
                why = f" Failing: {detail() if callable(detail) else report.summary()}."
            raise ArtifactNotValidated(
                f"{repo_id} has no serviceable validation report; refusing to "
                f"serve it.{why} Run the validation suite first — an "
                "unvalidated artifact reads out plausible nonsense rather "
                "than failing."
            )
        ref = self.find(repo_id)
        if ref is None:
            raise FileNotFoundError(f"no J-lens artifact for {repo_id}")
        payload = self._load_payload(ref)
        if payload is None:
            raise ArtifactNotValidated(f"{ref.lens_path} did not deserialize")
        return {int(k): v for k, v in payload.items()}

    def fitted_layers(self, ref: ArtifactRef) -> List[int]:
        """Layers this artifact covers, from config.yaml — no tensor load.

        Falls back to the `layer_scales` keys, which carry one entry per fitted
        layer, so artifacts written before `fitted_layers` existed still answer.
        An empty list means "unknown", not "none": the caller must not render
        that as an artifact covering nothing.
        """
        if ref.config_path is None or not ref.config_path.is_file():
            return []
        try:
            for raw in ref.config_path.read_text().splitlines():
                key, _, value = raw.partition(":")
                if key.strip() != "fitted_layers":
                    continue
                inner = value.strip().strip("[]")
                if not inner:
                    return []
                return sorted(int(p) for p in inner.split(",") if p.strip())
        except (OSError, ValueError) as exc:  # noqa: BLE001 - reported
            logger.warning("Unreadable fitted_layers in %s: %s", ref.config_path, exc)
            return []
        return sorted(self.layer_scales(ref))

    def degenerate_layers(self, ref: ArtifactRef) -> List[int]:
        """Layers where the fitted J is the identity — the logit lens, exactly.

        Empty means "none recorded", which for an artifact written before this
        was tracked is genuinely unknown rather than a claim that none exist.
        Consumers must not read empty as "every layer is informative".
        """
        if ref.config_path is None or not ref.config_path.is_file():
            return []
        try:
            for raw in ref.config_path.read_text().splitlines():
                key, _, value = raw.partition(":")
                if key.strip() != "degenerate_layers":
                    continue
                inner = value.strip().strip("[]")
                if not inner:
                    return []
                return sorted(int(p) for p in inner.split(",") if p.strip())
        except (OSError, ValueError) as exc:  # noqa: BLE001 - reported
            logger.warning("Unreadable degenerate_layers in %s: %s", ref.config_path, exc)
        return []

    def target_layer(self, ref: ArtifactRef) -> Optional[str]:
        """Which block the Jacobian was taken TO, from config.yaml.

        The coverage strip needs it: with a `penultimate` target a COMPLETE fit
        covers 0..N-2, so comparing against the model's layer count would render
        a full artifact as "25/26" and colour it amber — reporting a recipe
        choice as a defect.
        """
        if ref.config_path is None or not ref.config_path.is_file():
            return None
        try:
            for raw in ref.config_path.read_text().splitlines():
                key, _, value = raw.partition(":")
                if key.strip() == "target_layer":
                    got = value.strip()
                    return got if got in ("final", "penultimate") else None
        except OSError as exc:  # noqa: BLE001
            logger.warning("Could not read %s: %s", ref.config_path, exc)
        return None

    def layer_scales(self, ref: ArtifactRef) -> Dict[int, float]:
        """Per-layer factors the stored matrices were divided by, from config.yaml.

        Empty when the artifact predates the scale being recorded, or declares
        none. Empty means "no rescale to undo", which is the correct reading:
        `_to_storage_dtype` leaves the scale at 1.0 whenever the matrix fits
        fp16 without help, and that is the common case.

        Parsed with a narrow reader rather than a YAML dependency — the file is
        written by `_config_yaml` as flat `  <layer>: <float>` under a
        `layer_scales:` key, and pulling in a parser to read two lines would be
        a larger surface than the thing it reads.
        """
        if ref.config_path is None or not ref.config_path.is_file():
            return {}
        scales: Dict[int, float] = {}
        in_block = False
        try:
            for raw in ref.config_path.read_text().splitlines():
                if raw.strip() == "layer_scales:":
                    in_block = True
                    continue
                if in_block:
                    if raw[:1] not in (" ", "\t") or not raw.strip():
                        break
                    key, _, value = raw.strip().partition(":")
                    try:
                        scales[int(key)] = float(value)
                    except ValueError:
                        # A malformed entry is skipped, not guessed at: a wrong
                        # scale is worse than none, because it silently changes
                        # every magnitude read through this layer.
                        logger.warning(
                            "Unreadable layer scale %r in %s", raw.strip(),
                            ref.config_path,
                        )
        except OSError as exc:  # noqa: BLE001 - reported, not swallowed
            logger.warning("Could not read %s: %s", ref.config_path, exc)
            return {}
        return scales
