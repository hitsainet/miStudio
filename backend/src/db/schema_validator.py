"""Database schema validation for startup health checks.

This module verifies that all required database tables and columns exist,
helping catch migration issues early before they cause runtime errors.
"""

import logging
from typing import Optional

from sqlalchemy import inspect, text
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)


# Required tables and their critical columns
# Format: {table_name: [required_columns]}
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
        required_tables = REQUIRED_TABLES

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
