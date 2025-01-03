#!/bin/bash

# TALD UNIA Database Migration Script
# Version: 1.0.0
# SQLite Version: 3.42.0

set -euo pipefail
IFS=$'\n\t'

# Configuration
readonly DB_PATH="${DB_PATH:-./data/tald_unia.db}"
readonly MIGRATIONS_DIR="${MIGRATIONS_DIR:-./database/migrations}"
readonly SCHEMA_VERSION_TABLE="schema_versions"
readonly LOG_DIR="${LOG_DIR:-./logs/migrations}"
readonly BACKUP_DIR="${BACKUP_DIR:-./backups}"
readonly SQLITE_MIN_VERSION="3.42.0"

# Logging setup
mkdir -p "$LOG_DIR"
readonly LOG_FILE="$LOG_DIR/migration_$(date +%Y%m%d_%H%M%S).log"
exec 1> >(tee -a "$LOG_FILE")
exec 2>&1

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

error() {
    log "ERROR: $1" >&2
    exit 1
}

check_prerequisites() {
    # Check SQLite version
    if ! command -v sqlite3 >/dev/null; then
        error "sqlite3 is not installed"
    fi
    
    local sqlite_version
    sqlite_version=$(sqlite3 --version | cut -d' ' -f1)
    if [[ "$sqlite_version" < "$SQLITE_MIN_VERSION" ]]; then
        error "SQLite version $sqlite_version is below minimum required version $SQLITE_MIN_VERSION"
    }

    # Check directories
    [[ -d "$MIGRATIONS_DIR" ]] || error "Migrations directory $MIGRATIONS_DIR does not exist"
    [[ -r "$MIGRATIONS_DIR" ]] || error "Migrations directory $MIGRATIONS_DIR is not readable"
    
    # Create/check database directory
    mkdir -p "$(dirname "$DB_PATH")"
    touch "$DB_PATH" || error "Cannot create/access database file $DB_PATH"
    
    # Create backup directory
    mkdir -p "$BACKUP_DIR" || error "Cannot create backup directory $BACKUP_DIR"
    
    # Validate migration files
    local prev_version=0
    for f in "$MIGRATIONS_DIR"/*.sql; do
        [[ -f "$f" ]] || continue
        local version
        version=$(basename "$f" | cut -d'_' -f1)
        if ! [[ "$version" =~ ^[0-9]+$ ]]; then
            error "Invalid migration file name format: $f"
        fi
        if ((version <= prev_version)); then
            error "Migration versions must be strictly increasing: $f"
        fi
        prev_version=$version
    done
    
    return 0
}

init_version_table() {
    sqlite3 "$DB_PATH" <<EOF
        PRAGMA foreign_keys = ON;
        BEGIN TRANSACTION;
        CREATE TABLE IF NOT EXISTS $SCHEMA_VERSION_TABLE (
            version INTEGER PRIMARY KEY,
            migration_file TEXT NOT NULL,
            checksum TEXT NOT NULL,
            applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            execution_time INTEGER NOT NULL,
            status TEXT NOT NULL CHECK(status IN ('success', 'failed', 'rolled_back')),
            error_message TEXT,
            metadata TEXT CHECK(json_valid(metadata) OR metadata IS NULL)
        );
        CREATE INDEX IF NOT EXISTS idx_schema_versions_status ON $SCHEMA_VERSION_TABLE(status);
        COMMIT;
EOF
}

get_current_version() {
    local version
    version=$(sqlite3 "$DB_PATH" "SELECT COALESCE(MAX(version), 0) FROM $SCHEMA_VERSION_TABLE WHERE status = 'success';")
    echo "$version"
}

calculate_checksum() {
    local file="$1"
    sha256sum "$file" | cut -d' ' -f1
}

run_migration() {
    local migration_file="$1"
    local version="$2"
    local start_time
    local end_time
    local duration
    local checksum
    local backup_file
    
    checksum=$(calculate_checksum "$migration_file")
    backup_file="$BACKUP_DIR/backup_v${version}_$(date +%Y%m%d_%H%M%S).db"
    
    # Create backup
    cp "$DB_PATH" "$backup_file" || error "Failed to create backup before migration $version"
    
    log "Running migration $version from file: $migration_file"
    
    start_time=$(date +%s%N)
    
    # Execute migration within a transaction
    if ! sqlite3 "$DB_PATH" <<EOF
        PRAGMA foreign_keys = ON;
        PRAGMA journal_mode = WAL;
        PRAGMA synchronous = FULL;
        BEGIN TRANSACTION;
        $(cat "$migration_file");
        INSERT INTO $SCHEMA_VERSION_TABLE (version, migration_file, checksum, execution_time, status)
        VALUES ($version, '$(basename "$migration_file")', '$checksum', 0, 'success');
        COMMIT;
EOF
    then
        local error_msg="Migration $version failed"
        log "$error_msg"
        
        # Restore from backup
        cp "$backup_file" "$DB_PATH" || error "Failed to restore backup after failed migration"
        
        # Record failure
        sqlite3 "$DB_PATH" <<EOF
            INSERT INTO $SCHEMA_VERSION_TABLE (version, migration_file, checksum, execution_time, status, error_message)
            VALUES ($version, '$(basename "$migration_file")', '$checksum', 0, 'failed', '$error_msg');
EOF
        return 1
    fi
    
    end_time=$(date +%s%N)
    duration=$(( (end_time - start_time) / 1000000 )) # Convert to milliseconds
    
    # Update execution time
    sqlite3 "$DB_PATH" "UPDATE $SCHEMA_VERSION_TABLE SET execution_time = $duration WHERE version = $version;"
    
    log "Successfully completed migration $version (${duration}ms)"
    return 0
}

migrate() {
    check_prerequisites || exit 1
    
    init_version_table
    
    local current_version
    current_version=$(get_current_version)
    log "Current schema version: $current_version"
    
    local success=true
    
    for migration_file in "$MIGRATIONS_DIR"/*.sql; do
        [[ -f "$migration_file" ]] || continue
        
        local version
        version=$(basename "$migration_file" | cut -d'_' -f1)
        
        if ((version <= current_version)); then
            continue
        fi
        
        if ! run_migration "$migration_file" "$version"; then
            success=false
            break
        fi
    done
    
    if $success; then
        log "Migration completed successfully"
        return 0
    else
        error "Migration failed"
    fi
}

# Execute migration
migrate