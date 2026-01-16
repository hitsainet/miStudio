#!/usr/bin/env python3
"""Find columns in models that might not have migrations."""
import re
from pathlib import Path
from collections import defaultdict

# Get all columns defined in models
model_columns = defaultdict(set)
for model_file in Path("src/models").glob("*.py"):
    if model_file.name.startswith("__"):
        continue
    content = model_file.read_text()
    
    # Find table name
    table_match = re.search(r'__tablename__\s*=\s*["\'](\w+)["\']', content)
    if not table_match:
        continue
    table_name = table_match.group(1)
    
    # Find Column definitions
    col_matches = re.findall(r'^\s+(\w+)\s*=\s*Column\s*\(', content, re.MULTILINE)
    for col in col_matches:
        model_columns[table_name].add(col)

# Get all columns mentioned in migrations
migration_columns = defaultdict(set)
for mig_file in Path("alembic/versions").glob("*.py"):
    if mig_file.name.startswith("__"):
        continue
    content = mig_file.read_text()
    
    # Find add_column
    for match in re.findall(r'op\.add_column\s*\(\s*["\'](\w+)["\']\s*,\s*sa\.Column\s*\(\s*["\'](\w+)["\']', content):
        table, col = match
        migration_columns[table].add(col)
    
    # Find create_table columns
    for table_match in re.finditer(r'op\.create_table\s*\(\s*["\'](\w+)["\']([^)]+\))+', content, re.DOTALL):
        table_name = table_match.group(1)
        table_block = table_match.group(0)
        for col_match in re.findall(r'sa\.Column\s*\(\s*["\'](\w+)["\']', table_block):
            migration_columns[table_name].add(col_match)

# Compare
print("COLUMN MIGRATION GAP ANALYSIS")
print("=" * 60)
print()

gaps_found = False
for table in sorted(model_columns.keys()):
    model_cols = model_columns[table]
    mig_cols = migration_columns.get(table, set())
    
    # Standard columns always present
    standard = {'id', 'created_at', 'updated_at'}
    
    missing = model_cols - mig_cols - standard
    if missing:
        gaps_found = True
        print(f"📋 Table: {table}")
        print(f"   Model columns not found in migrations:")
        for col in sorted(missing):
            print(f"     - {col}")
        print()

if not gaps_found:
    print("✅ All model columns appear to have migrations!")
    print()
    print("Note: This checks column names against migration files.")
    print("Run check_migrations.py for authoritative DB comparison.")
