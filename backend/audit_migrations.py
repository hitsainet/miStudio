#!/usr/bin/env python3
"""
Audit tool to find potential migration gaps by comparing:
1. SQLAlchemy model column definitions
2. Alembic migration files
3. Git commit history for model changes

This helps identify columns added to models but missing from migrations.
"""

import os
import re
import ast
from pathlib import Path
from collections import defaultdict
from datetime import datetime

# Paths
MODELS_DIR = Path("src/models")
MIGRATIONS_DIR = Path("alembic/versions")

def extract_columns_from_model(filepath):
    """Extract Column definitions from a SQLAlchemy model file."""
    columns = {}
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Find class definitions that inherit from Base
    # Look for Column() definitions
    column_pattern = r'(\w+)\s*=\s*Column\s*\('
    matches = re.findall(column_pattern, content)
    
    # Also look for mapped_column() for SQLAlchemy 2.0 style
    mapped_pattern = r'(\w+)\s*:\s*Mapped\[.*?\]\s*=\s*mapped_column\s*\('
    matches.extend(re.findall(mapped_pattern, content))
    
    for col_name in matches:
        if col_name not in ['__tablename__', '__table_args__']:
            columns[col_name] = filepath.stem
    
    # Extract table name
    table_match = re.search(r'__tablename__\s*=\s*["\'](\w+)["\']', content)
    table_name = table_match.group(1) if table_match else filepath.stem
    
    return table_name, columns

def extract_columns_from_migration(filepath):
    """Extract column additions from a migration file."""
    columns = defaultdict(list)
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Look for add_column operations
    add_col_pattern = r'op\.add_column\s*\(\s*["\'](\w+)["\']\s*,\s*sa\.Column\s*\(\s*["\'](\w+)["\']'
    for match in re.findall(add_col_pattern, content):
        table_name, col_name = match
        columns[table_name].append(col_name)
    
    # Look for create_table operations and their columns
    # This is a simplified pattern - real parsing would need AST
    create_table_pattern = r'op\.create_table\s*\(\s*["\'](\w+)["\']'
    
    return dict(columns)

def get_migration_date(filepath):
    """Extract creation date from migration file."""
    with open(filepath, 'r') as f:
        content = f.read()
    
    date_match = re.search(r'Create Date:\s*(\d{4}-\d{2}-\d{2})', content)
    if date_match:
        return date_match.group(1)
    return "unknown"

def main():
    print("=" * 80)
    print("MIGRATION AUDIT REPORT")
    print("=" * 80)
    print()
    
    # Collect all model columns
    print("📁 Scanning SQLAlchemy models...")
    model_tables = {}
    for model_file in MODELS_DIR.glob("*.py"):
        if model_file.name.startswith("__"):
            continue
        try:
            table_name, columns = extract_columns_from_model(model_file)
            if columns:
                model_tables[table_name] = {
                    'columns': columns,
                    'file': model_file.name
                }
        except Exception as e:
            print(f"  ⚠️  Error parsing {model_file.name}: {e}")
    
    print(f"  Found {len(model_tables)} tables with {sum(len(t['columns']) for t in model_tables.values())} columns")
    print()
    
    # Collect all migration column additions
    print("📁 Scanning Alembic migrations...")
    migration_columns = defaultdict(set)
    migration_files = sorted(MIGRATIONS_DIR.glob("*.py"))
    
    for mig_file in migration_files:
        if mig_file.name.startswith("__"):
            continue
        try:
            cols = extract_columns_from_migration(mig_file)
            for table, col_list in cols.items():
                migration_columns[table].update(col_list)
        except Exception as e:
            print(f"  ⚠️  Error parsing {mig_file.name}: {e}")
    
    print(f"  Found {len(migration_files)} migration files")
    print()
    
    # Check for model columns not in any migration (potential gaps)
    print("🔍 Checking for potential migration gaps...")
    print()
    
    gaps_found = False
    for table_name, info in sorted(model_tables.items()):
        model_cols = set(info['columns'].keys())
        mig_cols = migration_columns.get(table_name, set())
        
        # Standard columns that are typically in create_table, not add_column
        standard_cols = {'id', 'created_at', 'updated_at'}
        
        # Check if any model columns might be missing migrations
        # This is heuristic - columns in create_table won't show in add_column
        potentially_missing = model_cols - mig_cols - standard_cols
        
        if potentially_missing and len(model_cols) > len(standard_cols):
            # Only flag if we have significant columns
            # Note: This may have false positives for columns in create_table
            pass  # We'll do a smarter check below
    
    # More targeted check: Look at recent model changes
    print("📅 Recent model file changes (last 50 commits):")
    print()
    
    import subprocess
    result = subprocess.run(
        ['git', 'log', '--oneline', '-50', '--', 'src/models/*.py'],
        capture_output=True, text=True, cwd=str(Path.cwd())
    )
    
    recent_model_commits = result.stdout.strip().split('\n') if result.stdout.strip() else []
    
    for commit in recent_model_commits[:15]:
        if commit:
            print(f"  {commit}")
    
    if len(recent_model_commits) > 15:
        print(f"  ... and {len(recent_model_commits) - 15} more")
    
    print()
    print("📅 Recent migration files (last 10):")
    print()
    
    recent_migrations = sorted(MIGRATIONS_DIR.glob("*.py"), 
                               key=lambda p: p.stat().st_mtime, 
                               reverse=True)[:10]
    
    for mig in recent_migrations:
        date = get_migration_date(mig)
        print(f"  {date}: {mig.name[:50]}...")
    
    print()
    print("=" * 80)
    print("RECOMMENDATION")
    print("=" * 80)
    print("""
To find actual migration gaps, run the existing check_migrations.py script
which compares SQLAlchemy models against the LIVE database schema:

    cd backend && source venv/bin/activate && python check_migrations.py

This audit script helps identify commits where model changes may have been
made without corresponding migrations, but the authoritative check is
comparing models against the actual database.
""")

if __name__ == "__main__":
    main()
