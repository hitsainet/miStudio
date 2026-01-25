#!/usr/bin/env python3
"""Schema verification script for CI/CD post-deploy checks.

This script verifies that all required database tables and columns exist.
It can be run after deployments to catch migration issues early.

Usage:
    python scripts/verify_schema.py [--fix]

    --fix    Attempt to create missing tables (requires appropriate permissions)

Exit codes:
    0 - Schema is valid
    1 - Schema validation failed
    2 - Error during validation
"""

import argparse
import asyncio
import os
import sys
from pathlib import Path

# Add the backend src directory to the path
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker


# Required tables and their critical columns
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


# SQL to create missing tables (for --fix mode)
FIX_SQL = {
    "feature_dashboard_data": """
        CREATE TABLE IF NOT EXISTS feature_dashboard_data (
            id BIGSERIAL PRIMARY KEY,
            feature_id VARCHAR(255) NOT NULL REFERENCES features(id) ON DELETE CASCADE,
            logit_lens_data JSONB,
            histogram_data JSONB,
            top_tokens JSONB,
            computed_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            computation_version VARCHAR(50),
            CONSTRAINT uq_feature_dashboard_feature_id UNIQUE (feature_id)
        );
        CREATE INDEX IF NOT EXISTS idx_feature_dashboard_feature_id ON feature_dashboard_data(feature_id);
    """,
    "neuronpedia_export_jobs": """
        DO $$
        BEGIN
            IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'export_status') THEN
                CREATE TYPE export_status AS ENUM (
                    'pending', 'computing', 'packaging', 'completed', 'failed', 'cancelled'
                );
            END IF;
        END
        $$;
        CREATE TABLE IF NOT EXISTS neuronpedia_export_jobs (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            sae_id VARCHAR(255) NOT NULL,
            source_type VARCHAR(50) NOT NULL,
            config JSONB NOT NULL DEFAULT '{}',
            status export_status NOT NULL DEFAULT 'pending',
            progress FLOAT NOT NULL DEFAULT 0.0,
            current_stage VARCHAR(100),
            output_path TEXT,
            file_size_bytes BIGINT,
            feature_count INTEGER,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            started_at TIMESTAMPTZ,
            completed_at TIMESTAMPTZ,
            error_message TEXT,
            error_details JSONB
        );
        CREATE INDEX IF NOT EXISTS idx_export_jobs_sae_id ON neuronpedia_export_jobs(sae_id);
        CREATE INDEX IF NOT EXISTS idx_export_jobs_status ON neuronpedia_export_jobs(status);
        CREATE INDEX IF NOT EXISTS idx_export_jobs_created_at ON neuronpedia_export_jobs(created_at);
    """,
}


async def get_existing_tables(session: AsyncSession) -> set[str]:
    """Get all existing table names in the public schema."""
    result = await session.execute(text("""
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = 'public'
        AND table_type = 'BASE TABLE'
    """))
    return {row[0] for row in result.fetchall()}


async def get_table_columns(session: AsyncSession, table_name: str) -> set[str]:
    """Get all column names for a table."""
    result = await session.execute(text("""
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema = 'public'
        AND table_name = :table_name
    """), {"table_name": table_name})
    return {row[0] for row in result.fetchall()}


async def verify_schema(database_url: str, fix: bool = False) -> tuple[bool, dict]:
    """
    Verify the database schema.

    Args:
        database_url: PostgreSQL connection URL
        fix: If True, attempt to create missing tables

    Returns:
        Tuple of (is_valid, report_dict)
    """
    # Create async engine
    engine = create_async_engine(database_url)
    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    missing_tables = []
    missing_columns = {}
    fixed_tables = []

    async with async_session() as session:
        existing_tables = await get_existing_tables(session)

        for table_name, required_cols in REQUIRED_TABLES.items():
            if table_name not in existing_tables:
                missing_tables.append(table_name)

                # Attempt to fix if requested
                if fix and table_name in FIX_SQL:
                    print(f"  Attempting to create missing table: {table_name}")
                    try:
                        await session.execute(text(FIX_SQL[table_name]))
                        await session.commit()
                        fixed_tables.append(table_name)
                        print(f"  ✓ Created table: {table_name}")
                    except Exception as e:
                        print(f"  ✗ Failed to create table {table_name}: {e}")
            else:
                # Check columns
                existing_cols = await get_table_columns(session, table_name)
                missing = [col for col in required_cols if col not in existing_cols]
                if missing:
                    missing_columns[table_name] = missing

    await engine.dispose()

    # Remove fixed tables from missing list
    for table in fixed_tables:
        if table in missing_tables:
            missing_tables.remove(table)

    is_valid = len(missing_tables) == 0 and len(missing_columns) == 0

    report = {
        "valid": is_valid,
        "missing_tables": missing_tables,
        "missing_columns": missing_columns,
        "fixed_tables": fixed_tables,
    }

    return is_valid, report


def print_report(report: dict) -> None:
    """Print a formatted schema validation report."""
    print("\n" + "=" * 60)
    print("DATABASE SCHEMA VALIDATION REPORT")
    print("=" * 60)

    if report["valid"]:
        print("\n✓ Schema is valid - all required tables and columns exist\n")
    else:
        print("\n✗ Schema validation FAILED\n")

        if report["missing_tables"]:
            print(f"Missing tables ({len(report['missing_tables'])}):")
            for table in report["missing_tables"]:
                print(f"  - {table}")
            print()

        if report["missing_columns"]:
            print(f"Tables with missing columns ({len(report['missing_columns'])}):")
            for table, cols in report["missing_columns"].items():
                print(f"  - {table}: {', '.join(cols)}")
            print()

    if report.get("fixed_tables"):
        print(f"Fixed tables ({len(report['fixed_tables'])}):")
        for table in report["fixed_tables"]:
            print(f"  ✓ {table}")
        print()

    print("=" * 60)


async def main():
    parser = argparse.ArgumentParser(description="Verify database schema")
    parser.add_argument("--fix", action="store_true", help="Attempt to create missing tables")
    parser.add_argument("--database-url", help="Database URL (defaults to DATABASE_URL env var)")
    args = parser.parse_args()

    # Get database URL
    database_url = args.database_url or os.environ.get("DATABASE_URL")
    if not database_url:
        print("Error: No database URL provided. Set DATABASE_URL env var or use --database-url")
        sys.exit(2)

    # Convert sync URL to async if needed
    if database_url.startswith("postgresql://"):
        database_url = database_url.replace("postgresql://", "postgresql+asyncpg://", 1)
    elif database_url.startswith("postgresql+psycopg2://"):
        database_url = database_url.replace("postgresql+psycopg2://", "postgresql+asyncpg://", 1)

    print(f"\nVerifying schema for database...")
    if args.fix:
        print("(--fix mode enabled: will attempt to create missing tables)")

    try:
        is_valid, report = await verify_schema(database_url, fix=args.fix)
        print_report(report)

        if not is_valid:
            print("\nTo fix missing tables, run migrations:")
            print("  alembic upgrade head")
            print("\nOr run this script with --fix to auto-create missing tables:")
            print("  python scripts/verify_schema.py --fix\n")

        sys.exit(0 if is_valid else 1)

    except Exception as e:
        print(f"\nError during schema verification: {e}")
        sys.exit(2)


if __name__ == "__main__":
    asyncio.run(main())
