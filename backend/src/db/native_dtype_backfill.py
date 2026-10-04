"""Fill `models.native_dtype` from each downloaded checkpoint's own config.json.

A FACT READ FROM A FILE, not an inference — the opposite of what the same migration refuses to do
for `model_dtype`, which would have to be guessed. Rows whose files are gone or record no dtype stay
NULL ("not recorded").

⚠ IT CANNOT FAIL THE MIGRATION. The API entrypoint refuses to serve when migrations fail, and a
backfill raising inside one took the API down once (11 restarts, 2026-08-25, `architecture_backfill`).
Every row is isolated: an unreadable file or an unexpected config is logged and skipped. Idempotent:
only NULL rows are selected.
"""

import logging

import sqlalchemy as sa

from src.db.architecture_backfill import config_dir

logger = logging.getLogger("alembic.runtime.migration")


def backfill_native_dtype(conn) -> int:
    """Set `native_dtype` on every model row that lacks it and whose config.json is readable."""
    from src.ml.native_dtype import checkpoint_dtype_of, read_config_json

    filled = 0
    rows = conn.execute(
        sa.text("SELECT id, file_path FROM models WHERE native_dtype IS NULL AND file_path IS NOT NULL")
    ).fetchall()
    for model_id, file_path in rows:
        try:
            directory = config_dir(file_path)
            if directory is None:
                continue
            recorded, _source = checkpoint_dtype_of(read_config_json(directory))
            if recorded is None:
                continue
            conn.execute(
                sa.text("UPDATE models SET native_dtype = :dtype WHERE id = :id"),
                {"dtype": recorded, "id": model_id},
            )
            filled += 1
        except Exception as exc:  # noqa: BLE001 - one bad row must not fail the migration
            logger.warning("native_dtype backfill skipped model %s: %s", model_id, exc)
    logger.info("native_dtype backfill: %d of %d model rows read from their config.json", filled, len(rows))
    return filled
