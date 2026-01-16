#!/usr/bin/env python3
"""
Check for type mismatches between SQLAlchemy models and database schema.
"""

import asyncio
import re
from collections import defaultdict

from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine
from src.core.config import settings
from src.core.database import Base

# Import all models to register them
from src.models.model import Model
from src.models.dataset import Dataset
from src.models.training import Training
from src.models.checkpoint import Checkpoint
from src.models.feature import Feature
from src.models.feature_activation import FeatureActivation
from src.models.extraction_job import ExtractionJob
from src.models.extraction_template import ExtractionTemplate
from src.models.training_template import TrainingTemplate
from src.models.training_metric import TrainingMetric
from src.models.activation_extraction import ActivationExtraction
from src.models.dataset_tokenization import DatasetTokenization
from src.models.labeling_job import LabelingJob
from src.models.labeling_prompt_template import LabelingPromptTemplate
from src.models.external_sae import ExternalSAE
from src.models.prompt_template import PromptTemplate
from src.models.neuronpedia_export import NeuronpediaExportJob
from src.models.steering_experiment import SteeringExperiment
from src.models.feature_analysis_cache import FeatureAnalysisCache
from src.models.feature_dashboard import FeatureDashboardData


# Type mappings from SQLAlchemy to PostgreSQL
SQLALCHEMY_TO_PG = {
    'INTEGER': ['integer', 'int4', 'serial'],
    'BIGINT': ['bigint', 'int8', 'bigserial'],
    'VARCHAR': ['character varying', 'varchar', 'text'],
    'TEXT': ['text', 'character varying'],
    'STRING': ['character varying', 'varchar', 'text'],
    'BOOLEAN': ['boolean', 'bool'],
    'FLOAT': ['double precision', 'float8', 'real', 'float4'],
    'DOUBLE': ['double precision', 'float8'],
    'DOUBLE_PRECISION': ['double precision', 'float8'],
    'TIMESTAMP': ['timestamp with time zone', 'timestamp without time zone', 'timestamptz'],
    'DATETIME': ['timestamp with time zone', 'timestamp without time zone'],
    'JSON': ['json', 'jsonb'],
    'JSONB': ['jsonb', 'json'],
    'UUID': ['uuid'],
    'ARRAY': ['ARRAY'],
}


def normalize_sa_type(sa_type_str):
    """Normalize SQLAlchemy type string."""
    base = re.match(r'(\w+)', str(sa_type_str))
    if base:
        return base.group(1).upper()
    return sa_type_str.upper()


def types_compatible(sa_type, pg_type):
    """Check if SQLAlchemy type is compatible with PostgreSQL type."""
    sa_normalized = normalize_sa_type(sa_type)
    pg_lower = pg_type.lower()
    
    if sa_normalized in SQLALCHEMY_TO_PG:
        return any(pg_lower.startswith(allowed) for allowed in SQLALCHEMY_TO_PG[sa_normalized])
    
    if sa_normalized.lower() in pg_lower:
        return True
    
    if sa_normalized == 'ENUM':
        return True
        
    return False


async def main():
    print("=" * 80)
    print("TYPE MISMATCH ANALYSIS")
    print("=" * 80)
    print()
    
    # Get model column types
    print("📋 Collecting SQLAlchemy model types...")
    model_types = {}
    for table in Base.metadata.sorted_tables:
        model_types[table.name] = {}
        for col in table.columns:
            model_types[table.name][col.name] = str(col.type)
    
    # Get database column types
    print("📋 Collecting database column types...")
    engine = create_async_engine(str(settings.database_url))
    
    async with engine.connect() as conn:
        result = await conn.execute(text("""
            SELECT table_name, column_name, data_type, udt_name, character_maximum_length
            FROM information_schema.columns
            WHERE table_schema = 'public'
            ORDER BY table_name, ordinal_position
        """))
        rows = result.fetchall()
    
    await engine.dispose()
    
    db_types = defaultdict(dict)
    db_lengths = defaultdict(dict)
    for row in rows:
        table_name, col_name, data_type, udt_name, max_length = row
        if data_type == 'USER-DEFINED':
            db_types[table_name][col_name] = f"enum({udt_name})"
        elif data_type == 'ARRAY':
            db_types[table_name][col_name] = f"ARRAY({udt_name})"
        else:
            db_types[table_name][col_name] = data_type
        db_lengths[table_name][col_name] = max_length
    
    # Compare types
    print()
    print("🔍 Checking for type mismatches...")
    print()
    
    mismatches = []
    for table_name in sorted(model_types.keys()):
        if table_name not in db_types:
            continue
            
        for col_name, sa_type in model_types[table_name].items():
            if col_name not in db_types[table_name]:
                continue
                
            pg_type = db_types[table_name][col_name]
            
            if not types_compatible(sa_type, pg_type):
                mismatches.append({
                    'table': table_name,
                    'column': col_name,
                    'model_type': sa_type,
                    'db_type': pg_type,
                })
    
    if mismatches:
        print("⚠️  TYPE MISMATCHES FOUND:")
        print("-" * 80)
        for m in mismatches:
            print(f"Table: {m['table']}")
            print(f"  Column: {m['column']}")
            print(f"  Model type:  {m['model_type']}")
            print(f"  DB type:     {m['db_type']}")
            print()
    else:
        print("✅ No type mismatches found between models and database!")
    
    # Check for ID columns that should be String but might be Integer
    print()
    print("=" * 80)
    print("ID COLUMN TYPE CHECK")
    print("=" * 80)
    print()
    
    id_issues = []
    for table_name in sorted(model_types.keys()):
        if table_name not in db_types:
            continue
        
        for col_name in ['id', 'model_id', 'dataset_id', 'training_id', 'extraction_id', 
                         'external_sae_id', 'sae_id', 'feature_id', 'extraction_job_id']:
            if col_name in model_types[table_name] and col_name in db_types[table_name]:
                sa_type = model_types[table_name][col_name]
                pg_type = db_types[table_name][col_name]
                max_len = db_lengths[table_name].get(col_name)
                
                # Check for String/VARCHAR columns
                if 'VARCHAR' in sa_type.upper() or 'STRING' in sa_type.upper():
                    if 'character varying' not in pg_type.lower() and 'varchar' not in pg_type.lower():
                        id_issues.append({
                            'table': table_name,
                            'column': col_name,
                            'expected': sa_type,
                            'actual': pg_type,
                            'max_length': max_len
                        })
                    else:
                        print(f"✅ {table_name}.{col_name}: {pg_type} (len={max_len})")
    
    if id_issues:
        print()
        print("⚠️  ID COLUMN TYPE ISSUES:")
        for issue in id_issues:
            print(f"  {issue['table']}.{issue['column']}")
            print(f"    Expected: {issue['expected']}")
            print(f"    Actual:   {issue['actual']}")


if __name__ == "__main__":
    asyncio.run(main())
