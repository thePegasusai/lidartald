#!/bin/bash

# TALD UNIA Database Backup Script
# Version: 1.0.0
# Dependencies: aws-cli (2.x), postgresql-client (14.x)

set -euo pipefail

# Global Configuration
BACKUP_RETENTION_DAYS=7
BACKUP_PATH="/var/backups/tald-unia/db"
MAX_PARALLEL_STREAMS=4
COMPRESSION_LEVEL=9
RETRY_ATTEMPTS=3
BACKUP_CHUNK_SIZE="1GB"
LOG_FILE="/var/log/tald-unia/db-backup.log"

# Source AWS credentials and configuration
source /etc/tald-unia/aws-credentials.env

# Logging function
log() {
    local level="$1"
    local message="$2"
    echo "{\"timestamp\":\"$(date -u +"%Y-%m-%dT%H:%M:%SZ")\",\"level\":\"${level}\",\"message\":\"${message}\"}" >> "${LOG_FILE}"
}

# Error handling function
handle_error() {
    local exit_code=$?
    local line_number=$1
    log "ERROR" "Backup failed at line ${line_number} with exit code ${exit_code}"
    cleanup_temp_files
    exit ${exit_code}
}

trap 'handle_error ${LINENO}' ERR

# Verify prerequisites
verify_prerequisites() {
    command -v aws >/dev/null 2>&1 || { log "ERROR" "aws-cli is required but not installed"; exit 5; }
    command -v pg_dump >/dev/null 2>&1 || { log "ERROR" "postgresql-client is required but not installed"; exit 5; }
}

# Create backup directory with proper permissions
create_backup_directory() {
    local timestamp=$(date +%Y%m%d_%H%M%S)
    local backup_dir="${BACKUP_PATH}/${timestamp}"
    mkdir -p "${backup_dir}"
    chmod 700 "${backup_dir}"
    echo "${backup_dir}"
}

# Create encrypted backup with parallel processing
create_backup() {
    local environment="$1"
    local backup_type="$2"
    local parallel_streams="$3"
    local backup_dir=$(create_backup_directory)
    local backup_file="${backup_dir}/tald_unia_${environment}_${backup_type}.sql.gz"
    
    log "INFO" "Starting backup: ${backup_file}"

    # Get RDS instance details from Terraform state
    local db_host=$(aws rds describe-db-instances --db-instance-identifier "${environment}-tald-unia" --query 'DBInstances[0].Endpoint.Address' --output text)
    local db_port=5432
    local db_name="tald_unia"

    # Execute parallel backup with compression
    PGPASSWORD="${DB_PASSWORD}" pg_dump \
        -h "${db_host}" \
        -p "${db_port}" \
        -U "tald_admin" \
        -d "${db_name}" \
        -j "${parallel_streams}" \
        -F directory \
        -Z "${COMPRESSION_LEVEL}" \
        -f "${backup_file}" \
        --verbose

    # Encrypt backup using AWS KMS
    aws kms encrypt \
        --key-id "${ENCRYPTION_KEY}" \
        --plaintext fileb://"${backup_file}" \
        --output text \
        --query CiphertextBlob > "${backup_file}.encrypted"

    # Calculate and store checksum
    sha256sum "${backup_file}.encrypted" > "${backup_file}.sha256"

    # Upload to S3 with server-side encryption
    aws s3 cp \
        "${backup_file}.encrypted" \
        "s3://tald-unia-backups-${environment}/db/${backup_type}/$(basename ${backup_file}).encrypted" \
        --sse aws:kms \
        --sse-kms-key-id "${ENCRYPTION_KEY}"

    log "INFO" "Backup completed successfully: ${backup_file}"
    return 0
}

# Verify backup integrity
verify_backup() {
    local backup_file="$1"
    local perform_restore_test="$2"

    log "INFO" "Verifying backup: ${backup_file}"

    # Verify checksum
    sha256sum -c "${backup_file}.sha256" || return 1

    if [ "${perform_restore_test}" = true ]; then
        # Create temporary database for restore testing
        local test_db="tald_unia_verify_$(date +%s)"
        PGPASSWORD="${DB_PASSWORD}" createdb -h localhost -U tald_admin "${test_db}"

        # Decrypt and restore backup
        aws kms decrypt \
            --ciphertext-blob fileb://"${backup_file}.encrypted" \
            --output text \
            --query Plaintext | base64 --decode | \
        PGPASSWORD="${DB_PASSWORD}" pg_restore \
            -h localhost \
            -U tald_admin \
            -d "${test_db}" \
            --verbose

        # Cleanup test database
        PGPASSWORD="${DB_PASSWORD}" dropdb -h localhost -U tald_admin "${test_db}"
    fi

    log "INFO" "Backup verification completed: ${backup_file}"
    return 0
}

# Cleanup old backups
cleanup_old_backups() {
    local backup_path="$1"
    local parallel_deletes="$2"

    log "INFO" "Starting cleanup of old backups"

    # Find and remove local backups older than retention period
    find "${backup_path}" -type f -mtime +${BACKUP_RETENTION_DAYS} -print0 | \
        xargs -0 -P "${parallel_deletes}" rm -f

    # Remove old backups from S3
    aws s3 ls "s3://tald-unia-backups-${environment}/db/" | \
        while read -r line; do
            createDate=$(echo "${line}" | awk {'print $1" "$2'})
            createDate=$(date -d "${createDate}" +%s)
            olderThan=$(date -d "-${BACKUP_RETENTION_DAYS} days" +%s)
            if [[ ${createDate} -lt ${olderThan} ]]; then
                aws s3 rm "s3://tald-unia-backups-${environment}/db/${line##* }"
            fi
        done

    log "INFO" "Cleanup completed"
    return 0
}

# Main execution
main() {
    verify_prerequisites

    # Parse command line arguments
    local environment="${1:-production}"
    local backup_type="${2:-full}"
    local parallel_streams="${3:-${MAX_PARALLEL_STREAMS}}"

    log "INFO" "Starting backup process for environment: ${environment}"

    # Create backup
    create_backup "${environment}" "${backup_type}" "${parallel_streams}"

    # Verify backup
    verify_backup "${BACKUP_PATH}/latest/tald_unia_${environment}_${backup_type}.sql.gz.encrypted" true

    # Cleanup old backups
    cleanup_old_backups "${BACKUP_PATH}" "${parallel_streams}"

    log "INFO" "Backup process completed successfully"
    exit 0
}

# Execute main function with parameters
main "$@"