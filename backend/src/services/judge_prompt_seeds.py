"""Judge-prompt seed FILES: the loader. No prompt text lives in this module, or in any module.

Operator request (2026-10-21): adding or changing a behaviour-monitoring topic must take
configuration, never code. So every starter judge prompt is a DATA FILE in the library's own
export format (`mistudio.judge-prompt/v1`, what Export writes and Import reads), and this module
only reads files:

* the built-in folder, ``src/data/judge_prompt_seeds/*.json``, shipped in the image; and
* an optional extra folder, ``JUDGE_PROMPT_SEED_DIR`` (Settings → Environment shows it; in
  Kubernetes a ConfigMap mounted there, see ``k8s/base/backend.yaml``).

`ensure_seeds` runs at API startup (`main.lifespan`). For each file, in name order, built-in first:

1. it is checked exactly as an import is (`judge_prompt_library.parse_document`); a file that does
   not parse or validate is SKIPPED with an error naming the file, and the others still load;
2. a ``name/vN`` not in the library is created (source ``seed``);
3. a ``name/vN`` already stored with the SAME content hash is left alone;
4. a ``name/vN`` already stored with OTHER content is never touched — a version is immutable — and a
   warning names the file and both hashes. To change a prompt, give the file a new version.

Nothing is ever deleted: removing a file leaves its prompt in the library. Two API replicas starting
together are safe: ``(name, version)`` is unique, and the loser of an insert race re-reads the
winner's row and compares hashes like any other existing row.

The migration that created the library (``d5f8a2c41e7b``) carries its own frozen copy of the two
original seeds, as migrations must; `test_judge_prompt_seed_files` asserts that copy equals the
built-in files byte for byte.
"""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

#: The starter prompts shipped in the image.
BUILTIN_SEED_DIR = Path(__file__).resolve().parent.parent / "data" / "judge_prompt_seeds"
#: Which files in a seed folder are read. Anything else (a README, a .bak) is ignored.
SEED_PATTERN = "*.json"

#: The prompt a judge run uses when it names none — the pre-library behaviour. A NAME, not text:
#: its text is in ``stakes-rating.v1.json``.
DEFAULT_PROMPT_NAME = "stakes-rating"
DEFAULT_PROMPT_VERSION = 1

#: Provenance of a row this loader created.
SEED_SOURCE = "seed"
SEED_ACTOR = "seed"


class SeedFileInvalid(ValueError):
    """A seed file that cannot be read as a judge-prompt document. Names the file."""

    def __init__(self, path: Path, reason: str):
        super().__init__(f"judge-prompt seed file {path}: {reason}")
        self.path = path
        self.reason = reason


def seed_id(name: str, version: int) -> str:
    """A FIXED row id for a seeded version, so the migration's rows, this loader's rows and every
    replica's rows are the same rows. Readable when it fits the 36-character id column
    (``jp_stakes_rating_v1``, as the migration wrote); otherwise derived from the key."""
    readable = f"jp_{name.replace('-', '_')}_v{int(version)}"
    if len(readable) <= 36:
        return readable
    return "jp_" + hashlib.sha256(f"{name}/v{int(version)}".encode("utf-8")).hexdigest()[:32]


def read_seed_file(path: Path) -> Dict[str, Any]:
    """The JSON object in ``path``, or `SeedFileInvalid` naming the file and what is wrong."""
    try:
        text = Path(path).read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as bad:
        raise SeedFileInvalid(Path(path), f"cannot be read ({bad})") from bad
    try:
        document = json.loads(text)
    except json.JSONDecodeError as bad:
        raise SeedFileInvalid(
            Path(path), f"is not valid JSON (line {bad.lineno}, column {bad.colno}: {bad.msg})"
        ) from bad
    if not isinstance(document, dict):
        raise SeedFileInvalid(Path(path), f"must hold one JSON object, not a {type(document).__name__}")
    return document


def seed_files(directory: Path) -> List[Path]:
    return sorted(p for p in Path(directory).glob(SEED_PATTERN) if p.is_file())


def builtin_document(name: str, version: int) -> Dict[str, Any]:
    """The built-in seed document for ``name/vN`` — how code that needs a starter prompt's text
    (`probe_monitor_judge`'s pinned stakes prompt) reads it from data instead of holding a copy."""
    for path in seed_files(BUILTIN_SEED_DIR):
        document = read_seed_file(path)
        if document.get("name") == name and document.get("version") == version:
            return document
    raise LookupError(
        f"no built-in judge-prompt seed file for {name}/v{version} in {BUILTIN_SEED_DIR}"
    )


def seed_directories(extra: Any = ...) -> List[Tuple[Path, bool]]:
    """``[(folder, is_builtin), ...]``: the built-in folder, then ``JUDGE_PROMPT_SEED_DIR`` if set.

    ``extra`` overrides the setting (tests); ``...`` reads it.
    """
    if extra is ...:
        from ..core.config import settings

        extra = settings.judge_prompt_seed_dir
    folders: List[Tuple[Path, bool]] = [(BUILTIN_SEED_DIR, True)]
    if extra:
        folders.append((Path(extra), False))
    return folders


@dataclass
class SeedReport:
    """What one `ensure_seeds` pass did. Every list holds plain dicts so it can be logged/served."""

    created: List[str] = field(default_factory=list)
    unchanged: List[str] = field(default_factory=list)
    refused: List[Dict[str, str]] = field(default_factory=list)
    invalid: List[Dict[str, str]] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "created": list(self.created),
            "unchanged": list(self.unchanged),
            "refused": list(self.refused),
            "invalid": list(self.invalid),
        }


async def _stored(db: Any, name: str, version: int) -> Optional[Any]:
    from sqlalchemy import select

    from ..models.probe_judge_prompt import ProbeJudgePrompt

    return (
        await db.execute(
            select(ProbeJudgePrompt).where(
                ProbeJudgePrompt.name == name, ProbeJudgePrompt.version == version
            )
        )
    ).scalar_one_or_none()


def _compare(report: SeedReport, path: Path, parsed: Any, existing: Any) -> None:
    if existing.content_hash == parsed.content_hash:
        report.unchanged.append(parsed.key)
        return
    report.refused.append(
        {
            "file": str(path),
            "prompt": parsed.key,
            "existing_hash": existing.content_hash,
            "file_hash": parsed.content_hash,
        }
    )
    logger.warning(
        "Judge-prompt seed file %s was NOT loaded: %s is already in the library with content hash "
        "%s, and the file's text hashes to %s. A version never changes once stored; give the file "
        "a new version number to add the new text.",
        path, parsed.key, existing.content_hash, parsed.content_hash,
    )


async def _load_one(db: Any, path: Path, builtin: bool, report: SeedReport) -> None:
    from sqlalchemy.exc import IntegrityError

    from ..models.probe_judge_prompt import ProbeJudgePrompt
    from .judge_prompt_library import PromptRefused, parse_document

    try:
        parsed = parse_document(read_seed_file(path))
    except SeedFileInvalid as bad:
        report.invalid.append({"file": str(path), "error": bad.reason})
        logger.error("Judge-prompt seed file %s was skipped: %s", path, bad.reason)
        return
    except PromptRefused as bad:
        report.invalid.append({"file": str(path), "error": f"{bad.code}: {bad.message}"})
        logger.error("Judge-prompt seed file %s was skipped: %s (%s)", path, bad.message, bad.code)
        return

    existing = await _stored(db, parsed.name, parsed.version)
    if existing is not None:
        _compare(report, path, parsed, existing)
        return

    db.add(
        ProbeJudgePrompt(
            id=seed_id(parsed.name, parsed.version),
            name=parsed.name,
            version=parsed.version,
            title=parsed.title.strip(),
            description=parsed.description or None,
            behaviour=parsed.behaviour,
            behaviour_label=parsed.behaviour_label or None,
            content_hash=parsed.content_hash,
            source=SEED_SOURCE,
            source_ref=None if builtin else {"seed_file": path.name},
            created_by=SEED_ACTOR,
            **parsed.content,
        )
    )
    try:
        await db.commit()
    except IntegrityError:
        # Another replica (or a concurrent request) stored this key first. Its row is now the
        # stored version; compare against it like any other.
        await db.rollback()
        winner = await _stored(db, parsed.name, parsed.version)
        if winner is None:
            raise
        _compare(report, path, parsed, winner)
        return
    report.created.append(parsed.key)
    logger.info("Judge-prompt seed %s added from %s", parsed.key, path)


async def ensure_seeds(db: Any, directories: Optional[Sequence[Tuple[Path, bool]]] = None) -> SeedReport:
    """Load every seed file into the library. Idempotent; never deletes; never changes a version.

    ``directories`` defaults to `seed_directories()` (built-in, then ``JUDGE_PROMPT_SEED_DIR``).
    """
    report = SeedReport()
    for directory, builtin in (directories if directories is not None else seed_directories()):
        if not Path(directory).is_dir():
            reason = "is not a folder (JUDGE_PROMPT_SEED_DIR)" if not builtin else "is missing from the image"
            report.invalid.append({"file": str(directory), "error": reason})
            logger.error("Judge-prompt seed folder %s %s; nothing loaded from it", directory, reason)
            continue
        for path in seed_files(directory):
            await _load_one(db, path, builtin, report)
    if report.created or report.refused or report.invalid:
        logger.info("Judge-prompt seeds: %s", report.as_dict())
    return report
