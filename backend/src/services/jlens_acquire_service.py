"""
Adopt a J-lens that someone else fitted.

miStudio can otherwise only obtain a lens by fitting one — a long GPU job per
model — while a large body of pre-fitted lenses is already published. The
conformance spec's own recommended sequence (§8) puts "check the repo, download,
validate, mount" FIRST and fitting last; this is that missing front half.

WHAT THIS MODULE MUST NOT DO IS INVENT PROVENANCE. The fit worker's
`_config_yaml` records a recipe — attention-gradient treatment, position scope,
aggregation, differentiation mode, corpus, sequence length, convergence
threshold — because it *performed* those choices. For an acquired lens miStudio
performed none of them. Writing them anyway, even as defaults, puts a recipe
miStudio invented into the file whose stated purpose (BR-007, spec §2.3) is that
the lens be reproducible from it alone, and `ProvenanceStrip` then renders it as
fact. That file already carries a scar of exactly this shape: `target_layer` was
hardcoded to "final" while the real parameter was threaded and dropped, so a
penultimate fit published a recipe claiming otherwise.

So every field below is DERIVED FROM SOMETHING MEASURABLE — the tensors, the
loaded model, or an explicit statement in the publisher's own config — and a
field that cannot be derived is OMITTED rather than defaulted. The readers treat
absence as unknown (`_config_bool`: "None is NOT False"), which is the truth.

UPSTREAM PROVENANCE GOES IN `acquisition.json`, NEVER INTO `config.yaml`. The
config readers are line scanners: they `partition(":")` and match on
`name.strip()`, returning on the first hit AT ANY INDENTATION. A nested
`acquired:` block would therefore not be namespaced at all — verified, a nested
`layer_scales:` is read as real and yields a FABRICATED per-layer rescale that
`JacobianTransport` applies to every probe and intervention magnitude, invisible
in ranked readouts because the model's final norm divides a positive scalar back
out. `acquisition.json` is ignored by `_ref_for` and by `check_naming`, exactly
as `validation.json` and `interventions.json` are.
"""

from __future__ import annotations

import hashlib
import json
import logging
import tempfile
import shutil
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch

logger = logging.getLogger(__name__)

#: Sidecar carrying everything about the TRANSFER rather than about the lens.
#:
#: Named like `validation.json` / `interventions.json` so a reader of the
#: upstream conformance layout ignores it; `_ref_for` requires only the `.pt`
#: and looks for `config.yaml`, and `check_naming` rejects only extra `.pt`s.
ACQUISITION_FILE = "acquisition.json"

#: How close to the identity a matrix must be to count as degenerate.
#:
#: The lens AT its own target layer is the identity by construction — there is no
#: transport to perform. Everything else is a measurement, so this is a
#: tolerance on a float comparison rather than a threshold anyone chose: the
#: fitter writes exact identities and fp16 round-trip perturbs them by ~1e-3.
IDENTITY_TOLERANCE = 5e-3


class WeightIdentity(str, Enum):
    """Whether the lens can be shown to belong to these weights."""

    #: The publisher named the model and it is the one we are attaching to.
    VERIFIED = "verified"
    #: The publisher named a DIFFERENT model. A hard refusal, never a warning.
    MISMATCH = "mismatch"
    #: No config, or no model named in it. Common for community repos; the
    #: user asserts the pairing and the artifact records that they did.
    UNVERIFIED = "unverified"


class AcquisitionRefused(RuntimeError):
    """Raised rather than adopting a lens we cannot honestly describe."""


@dataclass
class IdentityVerdict:
    state: WeightIdentity
    detail: str
    declared: Optional[str] = None
    expected: Optional[str] = None


@dataclass
class LayerVerdict:
    """What the tensors themselves say about indexing and the target layer."""

    fitted: List[int]
    d_model: int
    #: `final` / `penultimate` / None when it cannot be derived.
    target_layer: Optional[str]
    #: Layers whose matrix is the identity — measured, not declared.
    degenerate: List[int]
    #: `||J - I||_F` per layer, the evidence behind `degenerate` and the target.
    identity_distance: Dict[int, float] = field(default_factory=dict)


def check_weight_identity(
    upstream_config: Optional[Dict[str, Any]], repo_id: str
) -> IdentityVerdict:
    """Does the publisher's own config name the model we are attaching to?

    THE FIRST CHECK IN THIS PROJECT AGAINST EVIDENCE INSIDE THE FILE. The
    existing weight-identity check compares a caller-supplied slug against the
    slug of the model already loaded — two views derived from the same live
    record, which cannot detect a lens whose CONTENTS were fitted for other
    weights. A published `config.yaml` states `hf_model_name`, and that is a
    claim by the party who ran the fit.

    A mismatch is a REFUSAL, not a flag: a lens fitted for different weights
    "produces a complete, plausible readout that is wrong".
    """
    from .jlens_artifact_service import slug_for

    declared = None
    if upstream_config:
        declared = upstream_config.get("hf_model_name") or upstream_config.get("model")
    if not declared:
        return IdentityVerdict(
            WeightIdentity.UNVERIFIED,
            "the source names no model, so the pairing rests on the caller's "
            "assertion alone",
            expected=repo_id,
        )

    if str(declared).strip() == repo_id or slug_for(str(declared)) == slug_for(repo_id):
        return IdentityVerdict(
            WeightIdentity.VERIFIED,
            f"the source declares {declared!r}, which is these weights",
            declared=str(declared),
            expected=repo_id,
        )
    return IdentityVerdict(
        WeightIdentity.MISMATCH,
        f"the source was fitted for {declared!r}, not {repo_id!r}. A lens fitted "
        "for different weights produces a complete, plausible readout that is wrong",
        declared=str(declared),
        expected=repo_id,
    )


def inspect_layers(
    payload: Dict[int, torch.Tensor], n_layers: int, d_model: int
) -> LayerVerdict:
    """Read the indexing convention and the target layer OFF THE TENSORS.

    WHY THIS EXISTS RATHER THAN TRUSTING THE CONFIG. `check_semantic`
    deliberately scans EVERY fitted layer, so an artifact using a different
    layer-index convention — 1-based, or counting from the output — still finds
    the expected token somewhere and passes. Semantic discrimination cannot
    catch the failure most likely on a third-party artifact, and this can:

    * a key at or above `n_layers` is impossible for these weights;
    * the matrix at the target layer is the identity by construction, so the
      minimum of `||J - I||_F` locates the target independently of any claim.
    """
    if not payload:
        raise AcquisitionRefused("the lens contains no layers")

    fitted = sorted(payload)
    out_of_range = [l for l in fitted if l < 0 or l >= n_layers]
    if out_of_range:
        raise AcquisitionRefused(
            f"layers {out_of_range} are outside 0..{n_layers - 1}, so this lens "
            f"does not index the same stack as the model ({n_layers} layers). A "
            "different indexing convention still passes a semantic check, which "
            "scans every fitted layer"
        )

    widths = {int(t.shape[0]) for t in payload.values()}
    if widths != {d_model}:
        raise AcquisitionRefused(
            f"the lens matrices are {sorted(widths)} wide but the model's d_model "
            f"is {d_model}; these are not the same weights"
        )

    distance: Dict[int, float] = {}
    eye = torch.eye(d_model, dtype=torch.float32)
    for layer, matrix in payload.items():
        distance[layer] = float(torch.linalg.norm(matrix.float() - eye))

    degenerate = sorted(l for l, d in distance.items() if d <= IDENTITY_TOLERANCE)

    # THE TARGET IS DERIVED, NEVER ASSUMED. The fitted set runs up to the block
    # the Jacobian was taken TO, so its maximum locates the target — and that is
    # what `_target_index` and therefore the coverage gate reads back.
    top = fitted[-1]
    if top == n_layers - 1:
        target = "final"
    elif top == n_layers - 2:
        target = "penultimate"
    else:
        # Omitted rather than guessed. `target_layer()` returns None, and
        # `_coverage_delta` then fails closed when REPLACING an artifact — which
        # is the correct outcome for a lens whose extent we cannot describe.
        target = None
        logger.info(
            "Top fitted layer %d is neither final (%d) nor penultimate (%d); "
            "target_layer omitted",
            top,
            n_layers - 1,
            n_layers - 2,
        )

    return LayerVerdict(
        fitted=fitted,
        d_model=d_model,
        target_layer=target,
        degenerate=degenerate,
        identity_distance=distance,
    )


def _upstream_int(config: Optional[Dict[str, Any]], *path: str) -> Optional[int]:
    node: Any = config or {}
    for key in path:
        if not isinstance(node, dict) or key not in node:
            return None
        node = node[key]
    try:
        return int(node)
    except (TypeError, ValueError):
        return None


def _upstream_float(config: Optional[Dict[str, Any]], *path: str) -> Optional[float]:
    node: Any = config or {}
    for key in path:
        if not isinstance(node, dict) or key not in node:
            return None
        node = node[key]
    try:
        return float(node)
    except (TypeError, ValueError):
        return None


def derive_converged(upstream_config: Optional[Dict[str, Any]]) -> Optional[bool]:
    """Did the publisher's own numbers say the fit converged?

    DERIVED FROM TWO RECORDED VALUES, or omitted. The upstream fitter stops early
    when the mean relative change falls under `stop_at_delta`, and records both,
    so "converged" is a comparison rather than a claim. When either is missing
    this returns None — and `_config_bool` is explicit that "None is NOT False".

    Defaulting to True would let an unconverged third-party lens silently
    displace a converged local fit through the quality gate; defaulting to False
    would block a good one. Absent is the honest third answer.
    """
    delta = _upstream_float(upstream_config, "fit", "stop_at_delta")
    reached = _upstream_float(upstream_config, "results", "final_mean_rel_change")
    if delta is None or reached is None:
        return None
    return reached <= delta


def derive_n_prompts(
    upstream_config: Optional[Dict[str, Any]],
    checkpoint: Optional[Dict[str, Any]] = None,
) -> Optional[int]:
    """How many prompts the fit ACTUALLY ran, not how many were requested.

    `fit.n_prompts` is the cap the operator asked for; `results.prompts_fitted`
    is what ran before the convergence stop fired. On the published gemma-2-2b-it
    lens those are 1000 and 337. `_quality_regression` compares this number
    against the incumbent's, so taking the request would let a 337-prompt fit
    claim to be a 1000-prompt one and displace a genuinely larger local fit.
    """
    ran = _upstream_int(upstream_config, "results", "prompts_fitted")
    if ran is not None:
        return ran
    # A bare `n_prompts` at the top level is what OUR OWN configs write, and it
    # means "prompts seen". Fall back to it, never to `fit.n_prompts`.
    from_config = _upstream_int(upstream_config, "n_prompts")
    if from_config is not None:
        return from_config
    # THE CHECKPOINT CARRIES IT TOO, and that is the only source a community
    # repo shipping a bare `.pt` has. Without this, `n_prompts` is absent for
    # exactly those repos, `_quality_regression` cannot fire, and a 50-prompt
    # foreign lens silently displaces a converged 634-prompt local fit.
    return _upstream_int(checkpoint, "n_prompts")


def config_yaml_for_acquired(
    *,
    repo_id: str,
    layers: LayerVerdict,
    n_vocab: int,
    n_layers: int,
    dtype: str,
    upstream_config: Optional[Dict[str, Any]],
    checkpoint: Optional[Dict[str, Any]] = None,
) -> str:
    """The recipe file for a lens miStudio did not fit.

    ONLY WHAT WAS MEASURED OR EXPLICITLY READ. Compare the fit worker's
    `_config_yaml`, which additionally writes the treatment, position scope,
    aggregation, differentiation mode, corpus, sequence length and convergence
    threshold — every one of which describes a choice miStudio made while
    fitting. Here miStudio made none of them, and a defaulted value in this file
    is indistinguishable from a measured one to every reader downstream.
    """
    lines = [
        "# J-lens ACQUIRED, not fitted here.",
        "# Only fields miStudio could measure from the artifact or read from the",
        "# publisher's own config appear below. The transfer itself — source,",
        f"# revision, digests and identity verdicts — is in {ACQUISITION_FILE}.",
        f"model: {repo_id}",
        f"d_model: {layers.d_model}",
        f"n_layers: {n_layers}",
        f"n_vocab: {n_vocab}",
        f"dtype: {dtype}",
        f"fitted_layers: {layers.fitted}",
        f"degenerate_layers: {layers.degenerate}",
    ]

    # OMITTED WHEN UNDERIVABLE. `target_layer()` accepts only the two literals
    # and returns None otherwise, and `_coverage_delta` fails closed on None —
    # which is the right behaviour for a lens whose extent we cannot state.
    if layers.target_layer is not None:
        lines.append(f"target_layer: {layers.target_layer}")

    n_prompts = derive_n_prompts(upstream_config, checkpoint)
    if n_prompts is not None:
        lines.append(f"# the publisher's figure, not a fit miStudio ran")
        lines.append(f"n_prompts: {n_prompts}")

    converged = derive_converged(upstream_config)
    if converged is not None:
        lines.append(f"converged: {str(converged).lower()}")

    # NO `layer_scales:` BLOCK. An absent block reads as "no rescale to undo",
    # which is correct: the published artifacts store raw fp16 whose entries are
    # O(1), so nothing was scaled down to survive the cast. Writing 1.0s would
    # be equivalent arithmetically and would assert knowledge of a convention
    # the publisher never stated.
    return "\n".join(lines) + "\n"


def publication_blocker(report: Any) -> Optional[str]:
    """Why this report must not be committed, or None.

    A PURE DECISION, EXTRACTED. `commit` raises `ArtifactNotValidated` for any
    report short of a full pass, so a worker that calls it unguarded turns the
    likeliest outcome for a foreign lens — not surfacing the fixture token —
    into a traceback instead of a report carrying the per-layer evidence that
    distinguishes "bad lens" from "wrong fixture".

    Pulling it out of the worker is what makes it testable: an inline `if` is
    only reachable by running the whole task, so a mutation that deletes it
    survives every test that does not.
    """
    if not getattr(report, "serviceable", False):
        return (
            "the artifact did not pass the local checks, so it was staged and "
            "not published: " + report.summary()
        )
    return None


#: Bytes per element to assume when the dtype is NOT YET KNOWN.
#:
#: The preview sees a size and nothing else. Assuming 2 rejected a legitimate
#: full-coverage fp32 lens that `validate` — which derives the real element size
#: from the payload — accepts, and the MCP tool tells an agent to read the
#: preview verdict when choosing a path. The ceiling must therefore be drawn for
#: the WIDEST element a lens could use, or the pre-flight contradicts the check
#: it stands in for.
WIDEST_ELEMENT_BYTES = 4


def model_dims(record: Any) -> Optional[Dict[str, int]]:
    """`d_model`, `n_layers`, `n_vocab` from a Model row, WITHOUT loading weights.

    From `architecture_config`, which records what config.json said at download
    time. NONE when any of the three is missing, and the caller then reports no
    verdict rather than a guessed one — `check_envelope`'s bounds are "derived,
    never constants", so a defaulted dimension produces a bound derived from
    nothing that passes on one model and misses a real materialisation on
    another.
    """
    config = getattr(record, "architecture_config", None) or {}
    if not isinstance(config, dict):
        return None
    try:
        dims = {
            "d_model": int(config.get("hidden_size") or config.get("d_model")),
            "n_layers": int(config.get("num_hidden_layers") or config.get("n_layer")),
            "n_vocab": int(config.get("vocab_size")),
        }
    except (TypeError, ValueError):
        return None
    return dims if all(v > 0 for v in dims.values()) else None


def full_fit_ceiling(
    dims: Dict[str, int], dtype_bytes: int = WIDEST_ELEMENT_BYTES
) -> int:
    """The largest a lens for these weights could legitimately be.

    Mirrors `check_envelope`'s ceiling at FULL coverage, for the two places that
    must bound something before the artifact's own layer count is knowable: the
    pre-flight preview, and the expansion guard that runs before `torch.load`.

    The lower bound is deliberately absent — it scales with the layer count, and
    a partial lens is legitimately far smaller.
    """
    from .jlens_validation import container_allowance

    required = dims["d_model"] * dims["d_model"] * dtype_bytes * dims["n_layers"]
    return int(required * 1.5) + container_allowance(required)


def preview_envelope_verdict(
    size_bytes: int, dims: Dict[str, int], dtype_bytes: int = WIDEST_ELEMENT_BYTES
) -> Dict[str, Any]:
    """Whether a remote file's SIZE alone rules it out for these weights.

    ONLY THE UPPER BOUND. `check_envelope` uses the layer count for both bounds
    and the count is unknowable before the file is opened, so applying the
    model's full stack made the floor far too high — every legitimate PARTIAL
    lens previewed as a failure while the validator accepted it, and the MCP
    tool tells an agent to read this field when choosing a path.

    A file BIGGER than a full fit is out whatever its coverage: that is BR-006's
    materialisation guard and it is monotonic in the layer count. A smaller file
    is simply not decidable from size, and says so rather than guessing.
    """
    required = dims["d_model"] * dims["d_model"] * dtype_bytes * dims["n_layers"]
    ceiling = full_fit_ceiling(dims, dtype_bytes)
    materialised = dims["n_vocab"] * dims["d_model"] * dtype_bytes * dims["n_layers"]

    if size_bytes > ceiling:
        looks_materialised = size_bytes >= materialised * 0.5
        return {
            "fits": False,
            "detail": (
                f"{size_bytes:,} bytes exceeds {ceiling:,}, the most a lens for "
                f"these weights could be"
                + (
                    " — this looks like a materialised token dictionary (BR-006)"
                    if looks_materialised
                    else ""
                )
            ),
        }
    return {
        "fits": True,
        "detail": (
            f"{size_bytes:,} bytes is within a full fit for this model; the "
            "layer count sets the lower bound and is unknown until the file is "
            "opened"
        ),
    }


def file_digest(path: Path) -> str:
    """Streaming SHA-256, matching `_lens_digest`'s chunking."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bytes_identity(
    upstream_sha256: Optional[str], local_sha256: Optional[str]
) -> Optional[bool]:
    """True, False, or None for "the publisher did not say".

    Unknown is not divergence. Collapsing them made every non-LFS lens report
    that what we serve differs from what was published, which the MCP tool
    presents to an agent as a fact.
    """
    if not upstream_sha256 or not local_sha256:
        return None
    return upstream_sha256 == local_sha256


def write_acquisition_record(
    directory: Path,
    *,
    source_repo: str,
    source_path: str,
    revision: Optional[str],
    upstream_sha256: Optional[str],
    local_sha256: str,
    identity: IdentityVerdict,
    layers: LayerVerdict,
    upstream_config: Optional[Dict[str, Any]],
) -> Path:
    """Everything about the TRANSFER, beside the artifact and outside the recipe.

    Two questions the six check classes do not ask, both properties of the
    transfer rather than of the file's conformance:

    * **weight identity** — did the publisher say these are the same weights?
    * **byte identity** — is what we serve bit-for-bit what they published?

    Keeping them here rather than inventing a seventh check class matters: the
    classes answer "is this artifact conformant", and stretching them to cover
    provenance is how a suite starts meaning two things at once.
    """
    record = {
        "source": {
            "repo": source_repo,
            "path": source_path,
            # A REVISION, OR THE STATEMENT IS NOT REPRODUCIBLE. Without it
            # "acquired from <repo>" names a moving target.
            "revision": revision,
        },
        "bytes": {
            "upstream_sha256": upstream_sha256,
            "local_sha256": local_sha256,
            # TRI-STATE. `False` here meant "we did not check" as often as it
            # meant "they differ" — the Hub exposes `lfs.sha256` only for
            # LFS-tracked files, so every small lens reported a positive claim of
            # DIVERGENCE from a measurement that never ran. This module is
            # explicit everywhere else that None is not False.
            "identical": bytes_identity(upstream_sha256, local_sha256),
        },
        "weight_identity": {
            "state": identity.state.value,
            "detail": identity.detail,
            "declared": identity.declared,
            "expected": identity.expected,
        },
        "layers": {
            "fitted": layers.fitted,
            "target_layer": layers.target_layer,
            "degenerate": layers.degenerate,
            # Rounded for legibility; the full float is not evidence anyone reads.
            "identity_distance": {
                str(k): round(v, 6) for k, v in sorted(layers.identity_distance.items())
            },
        },
        # THE PUBLISHER'S OWN CONFIG, VERBATIM AND QUARANTINED. It is the richest
        # provenance available and it must not reach `config.yaml`, whose line
        # scanners would read its nested keys as miStudio's own.
        "upstream_config": upstream_config,
    }
    target = directory / ACQUISITION_FILE
    tmp = target.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(target)
    return target


def read_acquisition_record(directory: Path) -> Optional[Dict[str, Any]]:
    """The transfer record, or None when the artifact was fitted locally."""
    path = directory / ACQUISITION_FILE
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:  # noqa: BLE001 - reported
        logger.warning("Could not read %s: %s", path, exc)
        return None


def parse_upstream_config(text: str) -> Dict[str, Any]:
    """The publisher's `config.yaml`, as a nested dict.

    A REAL PARSER, unlike the line scanners that read miStudio's own configs.
    Those are deliberately narrow because they read a file this project writes
    in a known flat shape; this reads a file written by someone else, whose
    nesting is exactly what carries the fields worth having
    (`results.prompts_fitted`, `fit.stop_at_delta`).
    """
    import yaml

    try:
        loaded = yaml.safe_load(text)
    except yaml.YAMLError as exc:  # noqa: BLE001 - absent config is a real state
        logger.warning("Upstream config did not parse as YAML: %s", exc)
        return {}
    return loaded if isinstance(loaded, dict) else {}


def dtype_of(payload: Dict[int, torch.Tensor]) -> str:
    """The dtype actually on disk, read off a tensor rather than declared."""
    dtypes = {str(t.dtype).replace("torch.", "") for t in payload.values()}
    if len(dtypes) != 1:
        raise AcquisitionRefused(
            f"the lens mixes dtypes {sorted(dtypes)}; a consumer casts on load "
            "and would silently promote part of it"
        )
    return {"float16": "fp16", "bfloat16": "bf16", "float32": "fp32"}.get(
        next(iter(dtypes)), next(iter(dtypes))
    )

# ---------------------------------------------------------------- the source


#: Free space that must remain AFTER a download, on every volume it touches.
#:
#: Mirrors `circuit_capture_service.MIN_FREE_DISK_BYTES`, the only such guard
#: this repo had. No download path checked disk at all — and a J-lens for a 70B
#: model is multiple GB, on a volume already at 83%.
MIN_FREE_DISK_BYTES = 5 * 2**30

#: Extensions that could hold a lens. NOT `*_jacobian_lens.pt`: that glob is the
#: conformant naming, and community repos publish `qwen3_8b_lens.pt`,
#: `gemma2_9b_jlens.pt` and worse. Listing only conformant names would make the
#: generic path useless for exactly the repos it exists to reach.
CANDIDATE_SUFFIXES = (".pt", ".pth", ".safetensors", ".bin")


@dataclass
class RemoteFile:
    path: str
    size_bytes: Optional[int]
    sha256: Optional[str]
    #: Sits beside a `config.yaml` — i.e. probably a self-describing artifact
    #: whose weight identity can be checked rather than asserted.
    has_config: bool
    #: Sits beside a `*_convergence.csv`.
    has_convergence: bool


@dataclass
class RepoPreview:
    repo_id: str
    revision: str
    candidates: List[RemoteFile]


def preview_repo(
    repo_id: str, revision: Optional[str] = None, token: Optional[str] = None
) -> RepoPreview:
    """List the files in a repo that could be a lens, with sizes.

    READ-ONLY, AND THE POINT IS TO SPEND A REQUEST INSTEAD OF A DOWNLOAD. A
    mistyped path would otherwise cost a multi-GB fetch and a slot on the
    single-GPU queue before anything noticed.

    THE REVISION IS RESOLVED HERE. `hf_hub_download` without one takes `main`,
    which moves — so "acquired from <repo>" would not be a reproducible
    statement. The caller passes the resolved sha back when it downloads, so the
    file that was previewed is the file that arrives.
    """
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    info = api.repo_info(repo_id=repo_id, revision=revision, repo_type="model", files_metadata=True)
    resolved = info.sha or revision or "main"

    siblings = list(getattr(info, "siblings", None) or [])
    all_paths = {s.rfilename for s in siblings}
    # AN EXACT BASENAME. `endswith("config.yaml")` also matched
    # `adapter_config.yaml` and `model_config.yaml`, so a repo holding one of
    # those previewed as self-describing — and `has_config` is the field the MCP
    # tool calls "the field to read first", and the sort key that floats a
    # candidate to the top. The worker then asks for `<parent>/config.yaml`,
    # which is not there, and the acquisition comes back unverified.
    dirs_with_config = {
        p.rsplit("/", 1)[0] if "/" in p else ""
        for p in all_paths
        if p.rsplit("/", 1)[-1] == "config.yaml"
    }
    dirs_with_csv = {
        p.rsplit("/", 1)[0] if "/" in p else ""
        for p in all_paths
        if p.endswith("_convergence.csv")
    }

    candidates: List[RemoteFile] = []
    for sibling in siblings:
        name = sibling.rfilename
        if not name.endswith(CANDIDATE_SUFFIXES):
            continue
        parent = name.rsplit("/", 1)[0] if "/" in name else ""
        lfs = getattr(sibling, "lfs", None)
        candidates.append(
            RemoteFile(
                path=name,
                # LFS carries the real size; `size` is the pointer file's.
                size_bytes=(getattr(lfs, "size", None) if lfs else None)
                or getattr(sibling, "size", None),
                sha256=getattr(lfs, "sha256", None) if lfs else None,
                has_config=parent in dirs_with_config,
                has_convergence=parent in dirs_with_csv,
            )
        )
    candidates.sort(key=lambda c: (not c.has_config, c.path))
    return RepoPreview(repo_id=repo_id, revision=resolved, candidates=candidates)


def download_footprint(size_bytes: int) -> int:
    """How much disk a download of this size actually consumes: TWICE its size.

    `fetch_file` writes the blob into the HuggingFace cache and `stage_from_file`
    then copies it into the registry, so both exist simultaneously. Reserving
    one copy passes and then fills the volume with the second.
    """
    return 2 * max(0, int(size_bytes))


def check_free_space(*paths: Path, needed_bytes: int) -> None:
    """Refuse a download that would not fit, BEFORE fetching a byte.

    EVERY VOLUME IT TOUCHES. The HuggingFace cache and the artifact registry can
    be different mounts, and the file lands on both — once as a cached blob and
    once as the staged artifact. Checking only the destination passes and then
    fills the cache volume instead.
    """
    import os
    import shutil as _shutil

    # ONE PROBE PER DISTINCT MOUNT. The registry lives INSIDE the data dir
    # (`jlens_artifacts_dir = data_dir / "jlens"`), so passing both probed the
    # same filesystem twice while the HuggingFace cache — the volume the
    # download hits FIRST — went unchecked, which is precisely what this
    # function's docstring says it exists to prevent.
    seen: set = set()
    for path in paths:
        probe = path
        while not probe.exists() and probe != probe.parent:
            probe = probe.parent
        # KEYED ON THE DEVICE. `(total, free)` collapses two genuinely distinct
        # mounts whose usage happens to coincide — two same-size volumes from
        # one StorageClass, both freshly provisioned, are identical on both
        # numbers — and the one it would skip is the cache volume this guard
        # exists to cover.
        device = os.stat(probe).st_dev
        if device in seen:
            continue
        seen.add(device)
        free = _shutil.disk_usage(probe).free
        if free < needed_bytes + MIN_FREE_DISK_BYTES:
            raise AcquisitionRefused(
                f"{path} has {free / 2**30:.1f} GiB free; this needs "
                f"{needed_bytes / 2**30:.1f} GiB plus a "
                f"{MIN_FREE_DISK_BYTES / 2**30:.0f} GiB floor"
            )


def declared_uncompressed_size(path: Path) -> Optional[int]:
    """What a `torch.save` archive says it will expand to, WITHOUT expanding it.

    EVERY OTHER GUARD IN THIS PATH BOUNDS THE COMPRESSED SIZE. `check_envelope`
    compares `size_bytes` off the filesystem, `preview_envelope_verdict` compares
    the size the Hub reports, and `download_footprint` reserves twice the
    download — so nothing bounds what `torch.load` will allocate. Measured: a
    34 KB archive of zeros expands to 33.5 MB, a factor of 986, and loads without
    complaint. At that ratio a file small enough to pass every check above
    OOM-kills the worker — which is the single-GPU queue, head-of-line for every
    fit, readout and intervention.

    A zip's central directory carries each member's uncompressed length, so this
    is a header read rather than a decompression. Returns None when the file is
    not a zip — an old-format `.pt` or a `.safetensors` — and the caller must
    then decide, rather than being handed a zero that reads as "no risk".
    """
    import zipfile

    try:
        with zipfile.ZipFile(path) as archive:
            return sum(info.file_size for info in archive.infolist())
    except (zipfile.BadZipFile, OSError):
        return None


def check_expansion(path: Path, ceiling_bytes: int) -> int:
    """Refuse a checkpoint that would expand past what these weights could need.

    The ceiling is the caller's envelope for a FULL fit — the same bound
    `check_envelope` applies to the file, applied instead to what the file
    becomes. An archive that declares no size is refused too when it is already
    over the ceiling on disk, and otherwise allowed: a non-zip checkpoint cannot
    hide expansion, because it has none.
    """
    declared = declared_uncompressed_size(path)
    on_disk = path.stat().st_size
    effective = declared if declared is not None else on_disk
    if effective > ceiling_bytes:
        raise AcquisitionRefused(
            f"the checkpoint expands to {effective:,} bytes, above the "
            f"{ceiling_bytes:,} a lens for these weights could need"
            + (
                f" — it is only {on_disk:,} bytes on disk, a "
                f"{effective / max(on_disk, 1):.0f}x expansion"
                if declared is not None and declared > on_disk * 2
                else ""
            )
        )
    return effective


def fetch_file(
    repo_id: str,
    path_in_repo: str,
    revision: str,
    token: Optional[str] = None,
    cache_dir: Optional[Path] = None,
) -> Path:
    """Download one file at a PINNED revision, into the HF cache.

    NOT INTO THE ARTIFACT REGISTRY. `list_artifacts` excludes only `.staging`,
    `.superseded` and `.swap` — a scratch directory anywhere else under the root
    holding a conformant `*_jacobian_lens.pt` would be discovered and SERVED as
    a second artifact under a bogus slug. The registry is the filesystem, so
    anything written there is published.

    The cache also gives resume, dedup and etag validation for free.
    """
    from huggingface_hub import hf_hub_download

    return Path(
        hf_hub_download(
            repo_id=repo_id,
            filename=path_in_repo,
            revision=revision,
            token=token,
            cache_dir=str(cache_dir) if cache_dir else None,
        )
    )


def fetch_optional(
    repo_id: str,
    path_in_repo: str,
    revision: str,
    token: Optional[str] = None,
    cache_dir: Optional[Path] = None,
) -> Optional[Path]:
    """A sibling file that may not exist. Absence is a real state, not an error.

    Community repos ship a bare `.pt` with no config and no convergence trace,
    and that is a lens miStudio can still adopt — as UNVERIFIED, which is the
    honest record.
    """
    try:
        return fetch_file(repo_id, path_in_repo, revision, token, cache_dir)
    except Exception as exc:  # noqa: BLE001 - absence is expected, not a failure
        logger.info("No %s in %s@%s (%s)", path_in_repo, repo_id, revision[:8], exc)
        return None


def sibling_paths(lens_path: str) -> Dict[str, str]:
    """Where the config and convergence trace sit relative to a lens file.

    Spec §2.1 puts all three in one directory, so this is a directory join
    rather than a search. The convergence file's stem follows the LENS file's,
    not the directory's — `gpt2-small/` holds `gpt2_convergence.csv` — because
    upstream derives both from the HuggingFace id while the directory carries
    the publisher's own model name.
    """
    parent = lens_path.rsplit("/", 1)[0] if "/" in lens_path else ""
    stem = lens_path.rsplit("/", 1)[-1]
    for suffix in ("_jacobian_lens.pt", ".pt", ".pth", ".safetensors", ".bin"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    join = (lambda name: f"{parent}/{name}") if parent else (lambda name: name)
    return {
        "config": join("config.yaml"),
        "convergence": join(f"{stem}_convergence.csv"),
    }

# ------------------------------------------------------------------ publishing


#: Files that travel when a lens is published, in the layout spec §2.1 defines.
#:
#: `interventions.json` IS INCLUDED, because carrying it is the entire reason it
#: exists: "a lens published to HuggingFace and pulled down by a serving runtime
#: arrives as files and nothing else", and a result that does not make that
#: journey leaves the consumer "with a dictionary it can read and no measurements
#: of what happened when it was applied". Publish is the only mechanism that
#: could carry it, and the MCP tool already promises agents that it does.
#:
#: `validation.json` and `acquisition.json` are DELIBERATELY EXCLUDED. The first
#: is miStudio's verdict on its own copy — including two classes recorded as
#: DEFERRED, which is a statement about this installation and not about the
#: artifact — and the second describes a transfer that has nothing to do with a
#: downstream consumer. A reader of the conformance format ignores both, and
#: shipping them invites someone to read our local verdict as the lens's own.
PUBLISHED_FILES = ("config.yaml", "interventions.json")


def published_path(repo_id: str, dataset: str = "mistudio") -> str:
    """Where a lens for `repo_id` goes inside a repo.

    Mirrors spec §2.1 — `<model>/jlens/<dataset>/` — so a consumer that already
    resolves published lenses finds ours without being told anything new. The
    dataset segment names the corpus the fit was drawn from; `mistudio` is the
    honest default for a fit whose corpus was supplied ad hoc rather than being
    a named public dataset.
    """
    from .jlens_artifact_service import slug_for

    return f"{slug_for(repo_id)}/jlens/{dataset}"


def model_card(
    repo_id: str,
    recipe: Dict[str, Any],
    validation: Optional[Dict[str, Any]] = None,
    dataset: str = "mistudio",
) -> str:
    """The README that travels with a published lens.

    STATES WHAT WAS AND WAS NOT CHECKED. A lens published from here carries two
    consumer-interop classes recorded as DEFERRED — nothing has ever run them,
    because doing so needs a live external consumer — and a reader who assumes
    a green suite means interoperability proven would be reading something this
    project has never measured.
    """
    lines = [
        "---",
        "library_name: jacobian-lens",
        "tags:",
        "- jacobian_lens",
        "- interpretability",
        "---",
        "",
        f"# Jacobian lens for `{repo_id}`",
        "",
        "A training-free dictionary that reads what this model is *poised to say*",
        "at each layer and token position. Fitted with MechInterp Studio.",
        "",
        "## Layout",
        "",
        "```",
        f"{published_path(repo_id, dataset)}/",
        f"    {slug_of(repo_id)}_jacobian_lens.pt",
        "    config.yaml",
        "```",
        "",
        "The checkpoint is a `torch.save` dict loadable with `weights_only=True`:",
        "",
        "```python",
        '{"J": {layer: Tensor[d_model, d_model]}, "d_model": int,',
        ' "source_layers": [int], "n_prompts": int}',
        "```",
        "",
        "## Recipe",
        "",
    ]
    for key in (
        "n_prompts",
        "converged",
        "target_layer",
        "fitted_layers",
        "d_model",
        "n_layers",
        "dtype",
    ):
        if recipe.get(key) is not None:
            lines.append(f"- **{key}**: {recipe[key]}")

    lines += [
        "",
        "## What has been checked",
        "",
    ]
    if validation:
        for row in validation.get("results", []):
            lines.append(
                f"- `{row.get('check')}`: **{row.get('status')}** — {row.get('detail')}"
            )
        lines += [
            "",
            "`deferred` means the check needs a live external consumer and was "
            "not run here. It is not a pass.",
        ]
    else:
        lines.append("- no validation report travelled with this artifact")

    lines += [
        "",
        "## Caveat",
        "",
        "Band boundaries are NOT included and must not be inferred: the published",
        "sensory/workspace/motor figures were measured on one specific model, and",
        "porting them is exactly the error this project forbids by construction.",
        "Measure them for the model in front of you.",
        "",
    ]
    return "\n".join(lines)


def slug_of(repo_id: str) -> str:
    from .jlens_artifact_service import slug_for

    return slug_for(repo_id)


def _rebind_intervention_digests(path: Path, old: str, new: str) -> None:
    """Point recorded evidence at the republished lens, and SAY that it moved.

    `record_intervention_result` binds each record to the lens's digest so a
    measurement cannot be read against a different artifact. Republishing
    rewrites the container — the same matrices in the conformant wrapper — so
    the binding must follow or every record is dropped on arrival by the rule
    that protects it.
    """
    try:
        records = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:  # noqa: BLE001 - reported
        logger.warning("Could not rebind %s: %s", path, exc)
        return
    if not isinstance(records, list):
        return
    for record in records:
        if isinstance(record, dict) and record.get("lens_sha256") == old:
            record["lens_sha256"] = new
            record["rebound_from"] = old
            record["rebound_reason"] = (
                "republished in the conformant wrapper; the matrices are "
                "unchanged and only the container differs"
            )
    path.write_text(json.dumps(records, indent=2, sort_keys=True), encoding="utf-8")


def publish_artifact(
    directory: Path,
    repo_id: str,
    target_repo: str,
    token: str,
    *,
    dataset: str = "mistudio",
    create: bool = False,
    private: bool = False,
    recipe: Optional[Dict[str, Any]] = None,
    validation: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Upload a validated lens to HuggingFace, in the conformant layout.

    RETURNS THE COMMIT SHA. The SAE uploader this follows returns
    `commit_hash: None` with a comment saying it "would need to get [it] from
    the API response" — `upload_folder` returns a `CommitInfo` carrying `oid`,
    and for an artifact whose entire purpose is portable evidence, "published at
    X" without a revision names a moving target.
    """
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    if create:
        api.create_repo(
            repo_id=target_repo, repo_type="model", private=private, exist_ok=True
        )

    path_in_repo = published_path(repo_id, dataset)
    lens_files = list(directory.glob("*_jacobian_lens.pt"))
    if len(lens_files) != 1:
        raise AcquisitionRefused(
            f"{directory} holds {len(lens_files)} lens files; exactly one is "
            "required and a consumer globs for it"
        )

    # NORMALISED AT PUBLISH TIME, NOT ASSUMED. `write_staged` emits the
    # conformant wrapper now, but every artifact fitted BEFORE that change is
    # the bare `{layer: matrix}` form — including both lenses on the cluster —
    # and they carry a matching validation.json, so every gate here passes and
    # the upload proceeds. The consumer then reads `payload["J"]`, raises, and
    # is holding a README that describes the wrapper. Publishing is the one
    # moment the on-disk vintage stops being a local detail.
    from .jlens_artifact_service import INTERVENTION_FILE, normalise_payload

    # BOUNDED BEFORE IT IS OPENED, like the acquisition path. `check_free_space`
    # covers disk and runs AFTER `torch.load` has already committed the memory —
    # a 70B lens is ~10.7 GB resident in the worker that is head-of-line for
    # every fit on the single-GPU queue.
    if recipe and recipe.get("d_model") and recipe.get("n_layers"):
        check_expansion(
            lens_files[0],
            full_fit_ceiling(
                {
                    "d_model": int(recipe["d_model"]),
                    "n_layers": int(recipe["n_layers"]),
                    "n_vocab": int(recipe.get("n_vocab") or 0) or 1,
                }
            ),
        )

    raw = torch.load(lens_files[0], map_location="cpu", weights_only=True)
    layers = normalise_payload(raw)
    widths = {int(t.shape[0]) for t in layers.values()}
    if len(widths) != 1:
        raise AcquisitionRefused(
            f"the lens matrices are {sorted(widths)} wide; a consumer reads one "
            "d_model without a fallback"
        )
    n_prompts = raw.get("n_prompts") if isinstance(raw, dict) else None
    if n_prompts is None:
        n_prompts = (recipe or {}).get("n_prompts") or 0

    # SPACE FOR THE REWRITE. In the pod only /data is a volume; the temp copy
    # lands on the node's ephemeral layer, and a multi-GB lens there risks
    # evicting a pod that is also running GPU fits. This module refuses before a
    # byte moves on the acquire path and did not on this one.
    needed = lens_files[0].stat().st_size
    with tempfile.TemporaryDirectory() as staging:
        outbox = Path(staging)
        check_free_space(outbox, needed_bytes=needed)

        # EVERY OTHER FIELD THE ORIGINAL CARRIED SURVIVES. Rebuilding only the
        # four spec fields silently dropped anything else the publisher had put
        # there — `layer_convention` and `target_layer` most importantly, which
        # are exactly what a consumer needs to know whether the lens targets the
        # penultimate block or the final one. The checkpoint is the only place
        # those can travel for a repo that republishes just the `.pt`.
        preserved = {
            k: v
            for k, v in (raw.items() if isinstance(raw, dict) else [])
            if k not in {"J", "d_model", "source_layers", "n_prompts"}
            and not isinstance(k, int)
        }
        published_lens = outbox / lens_files[0].name
        torch.save(
            {
                **preserved,
                "J": layers,
                "d_model": widths.pop(),
                "source_layers": sorted(layers),
                "n_prompts": int(n_prompts),
            },
            published_lens,
        )
        # THE EVIDENCE IS REBOUND TO THE FILE THAT SHIPS.
        #
        # Re-saving changes the container's bytes, so every `lens_sha256` in
        # `interventions.json` would name a file that is not at the destination
        # — and miStudio's own rule drops a record whose digest does not match
        # the current lens. The evidence would arrive and self-invalidate, for
        # exactly the old-format artifacts the re-save exists to fix.
        #
        # Rebinding is honest because the TENSORS are identical: `normalise_payload`
        # reshapes the container and touches no matrix. Each record says so, so a
        # reader can tell a rebound digest from an original one.
        local_digest = file_digest(lens_files[0])
        shipped_digest = file_digest(published_lens)
        for name in PUBLISHED_FILES:
            source = directory / name
            if not source.is_file():
                continue
            if name == INTERVENTION_FILE and shipped_digest != local_digest:
                shutil.copyfile(source, outbox / name)
                _rebind_intervention_digests(
                    outbox / name, local_digest, shipped_digest
                )
                continue
            shutil.copyfile(source, outbox / name)
        # The convergence trace rides along per spec §2.1 — it is the
        # publisher's own record of whether the fit plateaued, and it is what
        # miStudio itself reads off an acquired artifact.
        for csv in directory.glob("*_convergence.csv"):
            shutil.copyfile(csv, outbox / csv.name)
        (outbox / "README.md").write_text(
            model_card(repo_id, recipe or {}, validation, dataset=dataset),
            encoding="utf-8",
        )

        info = api.upload_folder(
            folder_path=str(outbox),
            repo_id=target_repo,
            path_in_repo=path_in_repo,
            commit_message=f"Add Jacobian lens for {repo_id}",
        )

    revision = getattr(info, "oid", None) or getattr(info, "commit_id", None)
    return {
        "repo": target_repo,
        "path_in_repo": path_in_repo,
        "revision": revision,
        "url": f"https://huggingface.co/{target_repo}/tree/{revision or 'main'}/{path_in_repo}",
        "files": sorted(p.name for p in directory.glob("*") if p.name in PUBLISHED_FILES)
        + [lens_files[0].name, "README.md"],
    }

