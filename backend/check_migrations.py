#!/usr/bin/env python3
"""
Compare SQLAlchemy models with actual database schema to find migration gaps.
Run from backend directory: python check_migrations.py
"""
import asyncio
import sys
import os

# Add backend to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine
from src.core.config import settings


async def get_model_columns():
    """Get all columns defined in SQLAlchemy models."""
    from src.core.database import Base

    # Import all models to register them with Base.metadata
    from src.models.model import Model
    from src.models.dataset import Dataset
    from src.models.training import Training
    from src.models.checkpoint import Checkpoint
    from src.models.feature import Feature
    from src.models.feature_activation import FeatureActivation
    from src.models.training_metric import TrainingMetric
    from src.models.extraction_job import ExtractionJob
    from src.models.dataset_tokenization import DatasetTokenization
    from src.models.external_sae import ExternalSAE
    from src.models.training_template import TrainingTemplate
    from src.models.extraction_template import ExtractionTemplate
    from src.models.steering_experiment import SteeringExperiment

    model_info = {}
    for table_name, table in Base.metadata.tables.items():
        model_info[table_name] = {
            'columns': {col.name: str(col.type) for col in table.columns},
        }
    return model_info


async def get_db_columns():
    """Get all columns from actual database."""
    engine = create_async_engine(str(settings.database_url))

    async with engine.connect() as conn:
        # Get all tables and columns
        result = await conn.execute(text("""
            SELECT table_name, column_name, data_type, is_nullable
            FROM information_schema.columns
            WHERE table_schema = 'public'
            ORDER BY table_name, ordinal_position
        """))
        rows = result.fetchall()

        db_info = {}
        for row in rows:
            table_name = row[0]
            if table_name not in db_info:
                db_info[table_name] = {'columns': {}}
            db_info[table_name]['columns'][row[1]] = row[2]

    await engine.dispose()
    return db_info


async def main():
    model_info = await get_model_columns()
    db_info = await get_db_columns()

    print("=" * 80)
    print("MIGRATION GAP ANALYSIS")
    print("=" * 80)

    gaps_found = False
    migration_needed = []

    # Check for missing tables
    model_tables = set(model_info.keys())
    db_tables = set(db_info.keys())

    missing_tables = model_tables - db_tables
    if missing_tables:
        gaps_found = True
        print("\n🔴 MISSING TABLES (in models but not in DB):")
        for table in sorted(missing_tables):
            print(f"  - {table}")
            cols = list(model_info[table]['columns'].keys())
            print(f"    Columns: {cols}")
            migration_needed.append(f"CREATE TABLE {table}")

    extra_tables = db_tables - model_tables - {'alembic_version'}
    if extra_tables:
        print("\n🟡 EXTRA TABLES (in DB but not in models - may be intentional):")
        for table in sorted(extra_tables):
            print(f"  - {table}")

    # Check for missing columns in each table
    print("\n" + "-" * 80)
    print("COLUMN ANALYSIS BY TABLE")
    print("-" * 80)

    tables_with_issues = []
    for table_name in sorted(model_tables & db_tables):
        model_cols = set(model_info[table_name]['columns'].keys())
        db_cols = set(db_info[table_name]['columns'].keys())

        missing_cols = model_cols - db_cols
        extra_cols = db_cols - model_cols

        if missing_cols or extra_cols:
            tables_with_issues.append(table_name)
            print(f"\n📋 {table_name}:")
            if missing_cols:
                gaps_found = True
                print(f"  🔴 MISSING COLUMNS (need migration):")
                for col in sorted(missing_cols):
                    col_type = model_info[table_name]['columns'][col]
                    print(f"      - {col} ({col_type})")
                    migration_needed.append(f"ALTER TABLE {table_name} ADD COLUMN {col} {col_type}")
            if extra_cols:
                print(f"  🟡 EXTRA COLUMNS (in DB, not in model - may be deprecated):")
                for col in sorted(extra_cols):
                    print(f"      - {col}")

    # Print tables that are in sync
    synced_tables = []
    for table_name in sorted(model_tables & db_tables):
        model_cols = set(model_info[table_name]['columns'].keys())
        db_cols = set(db_info[table_name]['columns'].keys())
        if model_cols == db_cols:
            synced_tables.append(table_name)

    print(f"\n✅ Tables fully in sync ({len(synced_tables)}):")
    for t in synced_tables:
        print(f"    - {t}")

    if migration_needed:
        print("\n" + "=" * 80)
        print("🔴 MIGRATIONS NEEDED:")
        print("=" * 80)
        for m in migration_needed:
            print(f"  {m}")
        print("\nGenerate SQL for missing columns:")
        print("-" * 40)
        for table_name in sorted(model_tables & db_tables):
            model_cols = set(model_info[table_name]['columns'].keys())
            db_cols = set(db_info[table_name]['columns'].keys())
            missing_cols = model_cols - db_cols
            for col in sorted(missing_cols):
                col_type = model_info[table_name]['columns'][col]
                # Map SQLAlchemy types to PostgreSQL types
                pg_type = col_type.upper()
                if 'VARCHAR' in pg_type:
                    pg_type = 'VARCHAR(255)'
                elif 'INTEGER' in pg_type:
                    pg_type = 'INTEGER'
                elif 'FLOAT' in pg_type:
                    pg_type = 'FLOAT'
                elif 'BOOLEAN' in pg_type:
                    pg_type = 'BOOLEAN'
                elif 'TEXT' in pg_type:
                    pg_type = 'TEXT'
                elif 'DATETIME' in pg_type:
                    pg_type = 'TIMESTAMP'
                elif 'JSON' in pg_type:
                    pg_type = 'JSONB'
                print(f"ALTER TABLE {table_name} ADD COLUMN IF NOT EXISTS {col} {pg_type};")
    else:
        print("\n" + "=" * 80)
        print("✅ No migration gaps found! All models match the database schema.")
        print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
