"""Database schema validation for startup health checks.

This module verifies that all required database tables and columns exist,
helping catch migration issues early before they cause runtime errors.
"""

import logging
from typing import Optional

from sqlalchemy import inspect, text
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)


# DERIVED FROM THE ORM, NOT HAND-MAINTAINED.
#
# This used to be a literal dict of 17 tables. The ORM declares 36, so every
# table added in the last year — circuits, validation_manifests,
# cluster_profiles, steering_record_runs, agent_approval_requests,
# dismissed_operations — went unchecked, and the validator logged
# "Schema validation passed" on a database missing all of them (MIS-E2E-032).
# The anti-drift tool was itself drifting, and by exactly the mechanism it
# existed to detect.
#
# PADR IDL-16 claims this validator "compares live DB schema against SQLAlchemy
# model metadata". It now does.
def _required_tables() -> dict:
    """Every mapped table, with its non-nullable columns as the required set.

    Non-nullable is the right bar: a nullable column can legitimately be absent
    from an older database mid-migration, but a NOT NULL column that is missing
    means an INSERT will fail at runtime.
    """
    from ..core.database import Base
    from .. import models  # noqa: F401 — import registers every table

    required = {}
    for name, table in Base.metadata.tables.items():
        cols = [c.name for c in table.columns if not c.nullable]
        required[name] = cols or [c.name for c in table.primary_key]
    return required


# NOT evaluated at import. `schema_validator` is imported early — before every
# model module has been loaded — so an eager call here saw only 15 of the 36
# tables, silently reproducing the very gap this change removes. It is resolved
# at validation time instead, when the registry is complete.
REQUIRED_TABLES = {
    # Core tables
    "models": ["id", "name", "status", "created_at"],
    "datasets": ["id", "name", "status", "created_at"],
    "trainings": ["id", "status", "created_at"],
    "features": ["id", "neuron_index", "created_at"],

    # SAE tables
    "external_saes": ["id", "name", "status", "created_at"],
    "extraction_jobs": ["id", "status", "created_at"],

    # Analysis tables
    "feature_analysis_cache": ["id", "feature_id"],
    "feature_dashboard_data": ["id", "feature_id", "logit_lens_data", "histogram_data"],

    # Export tables
    "neuronpedia_export_jobs": ["id", "sae_id", "status"],

    # Template tables
    "training_templates": ["id", "name"],
    "extraction_templates": ["id", "name"],
    "steering_experiments": ["id", "name"],

    # Labeling tables
    "labeling_jobs": ["id", "status"],

    # Checkpoint tables
    "checkpoints": ["id", "training_id"],

    # Tokenization tables
    "dataset_tokenizations": ["id", "dataset_id"],
}


class SchemaValidationError(Exception):
    """Raised when schema validation fails."""

    def __init__(self, missing_tables: list[str], missing_columns: dict[str, list[str]]):
        self.missing_tables = missing_tables
        self.missing_columns = missing_columns

        errors = []
        if missing_tables:
            errors.append(f"Missing tables: {', '.join(missing_tables)}")
        for table, columns in missing_columns.items():
            errors.append(f"Table '{table}' missing columns: {', '.join(columns)}")

        super().__init__(f"Schema validation failed: {'; '.join(errors)}")


async def get_existing_tables(db: AsyncSession) -> set[str]:
    """Get all existing table names in the public schema."""
    result = await db.execute(text("""
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = 'public'
        AND table_type = 'BASE TABLE'
    """))
    return {row[0] for row in result.fetchall()}


async def get_table_columns(db: AsyncSession, table_name: str) -> set[str]:
    """Get all column names for a table."""
    result = await db.execute(text("""
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema = 'public'
        AND table_name = :table_name
    """), {"table_name": table_name})
    return {row[0] for row in result.fetchall()}


async def validate_schema(
    db: AsyncSession,
    required_tables: Optional[dict[str, list[str]]] = None,
    raise_on_error: bool = True
) -> tuple[bool, list[str], dict[str, list[str]]]:
    """
    Validate that all required tables and columns exist.

    Args:
        db: Database session
        required_tables: Optional custom table requirements (defaults to REQUIRED_TABLES)
        raise_on_error: If True, raise SchemaValidationError on failure

    Returns:
        Tuple of (is_valid, missing_tables, missing_columns)

    Raises:
        SchemaValidationError: If validation fails and raise_on_error is True
    """
    if required_tables is None:
        required_tables = _required_tables()

    existing_tables = await get_existing_tables(db)

    missing_tables = []
    missing_columns = {}

    for table_name, required_cols in required_tables.items():
        if table_name not in existing_tables:
            missing_tables.append(table_name)
            logger.warning(f"Schema validation: Missing table '{table_name}'")
        else:
            # Check columns
            existing_cols = await get_table_columns(db, table_name)
            missing = [col for col in required_cols if col not in existing_cols]
            if missing:
                missing_columns[table_name] = missing
                logger.warning(f"Schema validation: Table '{table_name}' missing columns: {missing}")

    is_valid = len(missing_tables) == 0 and len(missing_columns) == 0

    if is_valid:
        logger.info("Schema validation passed: All required tables and columns exist")
    else:
        logger.error(f"Schema validation failed: {len(missing_tables)} missing tables, "
                    f"{len(missing_columns)} tables with missing columns")
        if raise_on_error:
            raise SchemaValidationError(missing_tables, missing_columns)

    return is_valid, missing_tables, missing_columns


async def validate_schema_on_startup(db: AsyncSession) -> bool:
    """
    Validate schema on application startup.

    This is a softer check that logs warnings but doesn't crash the application.
    Critical tables that would cause immediate failures will log errors.

    Returns:
        True if schema is valid, False otherwise
    """
    try:
        is_valid, missing_tables, missing_columns = await validate_schema(
            db,
            raise_on_error=False
        )

        if not is_valid:
            # Log detailed report
            logger.error("=" * 60)
            logger.error("DATABASE SCHEMA VALIDATION FAILED")
            logger.error("=" * 60)

            if missing_tables:
                logger.error(f"Missing tables ({len(missing_tables)}):")
                for table in missing_tables:
                    logger.error(f"  - {table}")

            if missing_columns:
                logger.error(f"Tables with missing columns ({len(missing_columns)}):")
                for table, cols in missing_columns.items():
                    logger.error(f"  - {table}: {', '.join(cols)}")

            logger.error("=" * 60)
            logger.error("Run 'alembic upgrade head' to apply missing migrations")
            logger.error("=" * 60)

        return is_valid

    except Exception as e:
        logger.error(f"Schema validation encountered an error: {e}")
        return False


def get_schema_report(
    missing_tables: list[str],
    missing_columns: dict[str, list[str]]
) -> dict:
    """Generate a structured schema validation report."""
    return {
        "valid": len(missing_tables) == 0 and len(missing_columns) == 0,
        "missing_tables": missing_tables,
        "missing_columns": missing_columns,
        "total_missing_tables": len(missing_tables),
        "total_tables_with_missing_columns": len(missing_columns),
    }
