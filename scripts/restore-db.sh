#!/bin/bash
# Database restore script for MiStudio
# Usage: ./scripts/restore-db.sh <backup_file>

set -e

if [ -z "$1" ]; then
    echo "Usage: $0 <backup_file.sql.gz>"
    echo ""
    echo "Available backups:"
    ls -lh /home/x-sean/app/miStudio/backups/mistudio_db_*.sql.gz 2>/dev/null || echo "  No backups found"
    exit 1
fi

BACKUP_FILE="$1"

if [ ! -f "$BACKUP_FILE" ]; then
    echo "Error: Backup file not found: $BACKUP_FILE"
    exit 1
fi

echo "WARNING: This will overwrite the current database!"
echo "Backup file: $BACKUP_FILE"
read -p "Are you sure you want to continue? (y/N) " -n 1 -r
echo

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Restore cancelled."
    exit 0
fi

echo "Restoring database from backup..."

# Drop and recreate the database
docker exec mistudio-postgres psql -U postgres -c "DROP DATABASE IF EXISTS mistudio;"
docker exec mistudio-postgres psql -U postgres -c "CREATE DATABASE mistudio;"

# Restore from backup
if [[ "$BACKUP_FILE" == *.gz ]]; then
    gunzip -c "$BACKUP_FILE" | docker exec -i mistudio-postgres psql -U postgres -d mistudio
else
    docker exec -i mistudio-postgres psql -U postgres -d mistudio < "$BACKUP_FILE"
fi

echo "Database restored successfully!"
echo ""
echo "Please restart the backend to reconnect:"
echo "  docker-compose restart backend"
