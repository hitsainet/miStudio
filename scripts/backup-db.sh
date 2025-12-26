#!/bin/bash
# Database backup script for MiStudio
# Usage: ./scripts/backup-db.sh [backup_dir]

set -e

BACKUP_DIR="${1:-/home/x-sean/app/miStudio/backups}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="$BACKUP_DIR/mistudio_db_$TIMESTAMP.sql"

# Create backup directory if it doesn't exist
mkdir -p "$BACKUP_DIR"

echo "Creating database backup..."
docker exec mistudio-postgres pg_dump -U postgres -d mistudio > "$BACKUP_FILE"

# Compress the backup
gzip "$BACKUP_FILE"
BACKUP_FILE="$BACKUP_FILE.gz"

# Keep only last 10 backups
cd "$BACKUP_DIR"
ls -t mistudio_db_*.sql.gz 2>/dev/null | tail -n +11 | xargs -r rm --

echo "Backup created: $BACKUP_FILE"
echo "Size: $(du -h "$BACKUP_FILE" | cut -f1)"

# List recent backups
echo -e "\nRecent backups:"
ls -lh "$BACKUP_DIR"/mistudio_db_*.sql.gz 2>/dev/null | tail -5
