"""Repair `models.architecture_config` rows that record no layer count.

Kept out of the migration files so more than one revision can run it. The first
attempt (c4d8e1f60a92) shipped before `extract_architecture_config` tolerated a
config that REFUSES a global answer, so it skipped every heterogeneous model
and repaired nothing -- and an applied migration does not run again. Correcting
the extractor therefore needs a second revision, and two revisions must not
mean two copies of this logic.

Idempotent by construction: only rows MISSING `num_hidden_layers` are selected,
so a correct row is never overwritten and a re-run is free.
"""

import json
import logging
from pathlib import Path
from typing import Optional

import sqlalchemy as sa

logger = logging.getLogger("alembic.runtime.migration")


def config_dir(file_path: str) -> Optional[Path]:
    """The directory holding config.json, in either cache layout."""
    root = Path(file_path)
    if not root.is_dir():
        return None
    snapshots = sorted(root.glob("**/snapshots/*/config.json"))
    if snapshots:
        return snapshots[-1].parent
    return root if (root / "config.json").is_file() else None


def describe_model(directory: Path) -> dict:
    """Describe the model with the production extractor, not a copy of it."""
    from transformers import AutoConfig

    from src.ml.model_loader import extract_architecture_config

    config = AutoConfig.from_pretrained(str(directory), local_files_only=True)
    return extract_architecture_config(config)


def affected_rows(conn):
    """Rows whose stored config records no layer count.

    `models` is keyed by `id`; there is no `model_id` column. Naming one made
    this SELECT raise, and because the entrypoint refuses to serve without
    successful migrations, that took the API down rather than skipping a
    backfill (11 restarts, 2026-08-25).
    """
    return conn.execute(
        sa.text(
            "SELECT id, file_path, architecture_config FROM models "
            "WHERE file_path IS NOT NULL "
            "AND NOT jsonb_exists(COALESCE(architecture_config, '{}'::jsonb), "
            "                     'num_hidden_layers')"
        )
    ).fetchall()


def write_config(conn, model_id: str, config: dict) -> None:
    """Store a rebuilt config. Keyed by `id`."""
    conn.execute(
        sa.text(
            "UPDATE models SET architecture_config = CAST(:cfg AS jsonb) "
            "WHERE id = :mid"
        ),
        {"cfg": json.dumps(config, default=str), "mid": model_id},
    )


def backfill(conn) -> int:
    """Repair what can be repaired. Returns the number of rows updated."""
    try:
        rows = affected_rows(conn)
    except Exception as exc:                            # pragma: no cover
        # A backfill of descriptive metadata must never be why the API cannot
        # start. The per-row handler below cannot help when the failure is the
        # query itself.
        logger.warning("architecture_config backfill: skipped entirely (%s)", exc)
        return 0

    repaired = 0
    for model_id, file_path, existing in rows:
        try:
            directory = config_dir(file_path)
            if directory is None:
                logger.warning(
                    "architecture_config backfill: no config.json under %s for %s",
                    file_path, model_id,
                )
                continue

            rebuilt = describe_model(directory)
            if "num_hidden_layers" not in rebuilt:
                logger.warning(
                    "architecture_config backfill: %s exposes no layer count "
                    "on any tower; leaving it alone", model_id,
                )
                continue

            merged = dict(existing or {})
            merged.update(rebuilt)
            write_config(conn, model_id, merged)

            repaired += 1
            logger.info(
                "architecture_config backfill: %s -> %s layers (towers: %s)",
                model_id,
                rebuilt["num_hidden_layers"],
                sorted(rebuilt.get("towers") or {}) or "flat",
            )
        except Exception as exc:                        # pragma: no cover
            logger.warning(
                "architecture_config backfill: skipped %s (%s)", model_id, exc
            )

    logger.info("architecture_config backfill: repaired %d model(s)", repaired)
    return repaired
