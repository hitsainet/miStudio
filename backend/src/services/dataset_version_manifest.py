"""Read a miDataworks dataset-version manifest at probe-dataset registration (Feature 034, FR-26).

miDataworks publishes each dataset version with a manifest (`midataworks.dataset-version`,
schema `docs/schemas/midataworks-dataset-version-v1.json`, owned by miDataworks). When a probe
dataset is registered WITH one, miStudio checks it and records what it verified:

* the manifest is valid against the VENDORED schema (a byte-identical package copy — the image's
  build context is `backend/`, so `docs/schemas/` never reaches it; pinned by
  `test_vendored_dataset_schema.py`);
* the manifest names the split being registered, and its row count equals the downloaded split's
  row count BEFORE miStudio's own keyword filter and label mapping narrow it.

miStudio holds NO pydantic model of this schema (IDL-55): it is miDataworks' document, read with
`jsonschema` and a small table of paths. Every fact read is in `FACT_PATHS` below, and a test checks
each path exists in the committed schema — a renamed field fails at test time, not as a `KeyError`
in production.

**A missing fact is recorded as `null`, never defaulted.** A version still in a repository and not
yet published (M-14 "published state only") has no commit; writing a default there would turn an
absence into a recorded claim.
"""

from __future__ import annotations

import functools
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib import resources
from typing import Any, Optional

#: The package copy the image reads.
SCHEMA_PACKAGE = "src.schemas.vendored"
SCHEMA_FILE = "midataworks-dataset-version-v1.json"

#: Every fact this module reads, as a path through the manifest (008 FPRD §5.2 M-2, M-10, M-14).
#: `[]` marks an array whose items are searched. Checked against the schema's own `properties` /
#: `$defs` by `test_probe_dataset_manifest.py::test_every_fact_path_exists_in_the_schema`.
FACT_PATHS: dict[str, tuple[str, ...]] = {
    "dataset_name": ("version", "dataset"),           # M-2
    "version_id": ("version", "version_id"),          # M-2
    "version_number": ("version", "number"),          # M-2
    "split_name": ("content", "splits", "[]", "name"),  # M-10
    "split_rows": ("content", "splits", "[]", "rows"),  # M-10
    "publication_state": ("publication", "state"),    # M-14
    "repository": ("publication", "repo_id"),         # M-14
    "commit": ("publication", "commit"),              # M-14 (published state only)
}

RECORD_KIND = "mistudio.verified-dataset-version/1"


class ManifestInvalid(ValueError):
    """The manifest does not validate; `path` is the JSON path of the failing element."""

    def __init__(self, path: str, message: str):
        self.path = path
        super().__init__(f"dataset-version manifest is invalid at {path}: {message}")


@functools.cache
def load_schema() -> dict:
    text = resources.files(SCHEMA_PACKAGE).joinpath(SCHEMA_FILE).read_text(encoding="utf-8")
    return json.loads(text)


@functools.cache
def load_validator():
    """A compiled validator for the draft the FILE declares (`$schema`), so miDataworks picks it."""
    from jsonschema.validators import validator_for

    schema = load_schema()
    cls = validator_for(schema)
    cls.check_schema(schema)
    return cls(schema)


def canonical_sha256(document: dict) -> str:
    """SHA-256 of the request's canonical JSON. It identifies WHAT WAS VERIFIED — the manifest as
    sent — not miDataworks' file bytes, and not the split's content (deferred, IDL-55)."""
    canonical = json.dumps(document, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _get(document: Any, path: tuple[str, ...]) -> Any:
    node = document
    for key in path:
        if not isinstance(node, dict) or key not in node:
            return None
        node = node[key]
    return node


@dataclass(frozen=True)
class ManifestFacts:
    """What miStudio reads from a valid manifest."""

    document: dict

    def _fact(self, name: str) -> Any:
        return _get(self.document, FACT_PATHS[name])

    def _splits(self) -> list[dict]:
        path = FACT_PATHS["split_name"]
        array = _get(self.document, path[: path.index("[]")])
        return [s for s in array if isinstance(s, dict)] if isinstance(array, list) else []

    def split_names(self) -> list[str]:
        name_key = FACT_PATHS["split_name"][-1]
        return [s[name_key] for s in self._splits() if name_key in s]

    def row_count(self, split: str) -> Optional[int]:
        """The manifest's row count for `split`; `None` when the manifest does not name it.

        A manifest whose content is not rows (a config, a reward bundle) describes no splits, so
        every split reads as absent — which is the truth.
        """
        name_key = FACT_PATHS["split_name"][-1]
        rows_key = FACT_PATHS["split_rows"][-1]
        for entry in self._splits():
            if entry.get(name_key) == split:
                return entry.get(rows_key)
        return None

    def record(self, split: str, manifest_rows: int, downloaded_rows: int) -> dict:
        """The `probe_monitor_datasets.dataset_version` value: exactly what was verified."""
        published = self._fact("publication_state") == "published"
        return {
            "kind": RECORD_KIND,
            "version_id": self._fact("version_id"),
            "version_number": self._fact("version_number"),
            "dataset_name": self._fact("dataset_name"),
            "split": split,
            "manifest_rows": manifest_rows,
            "downloaded_rows": downloaded_rows,
            # Unknown is explicit: no publication → no repository; not published → no commit.
            "repository": self._fact("repository"),
            "commit": self._fact("commit") if published else None,
            "publication_state": self._fact("publication_state"),
            "manifest_sha256": canonical_sha256(self.document),
            "recorded_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        }


def _json_path(error) -> str:
    parts = "".join(f"[{p}]" if isinstance(p, int) else f".{p}" for p in error.absolute_path)
    return "$" + parts


def validate(document: Any) -> ManifestFacts:
    """Validate against the vendored schema; raise `ManifestInvalid` naming the failing path."""
    from jsonschema.exceptions import best_match

    error = best_match(load_validator().iter_errors(document))
    if error is not None:
        raise ManifestInvalid(_json_path(error), error.message)
    return ManifestFacts(document)
