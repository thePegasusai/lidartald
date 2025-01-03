#!/bin/bash

# TALD UNIA Secret Rotation Script
# Version: 1.0.0
# Dependencies:
# - aws-cli v2.0
# - vault v1.13.1
# - kubectl v1.27

set -euo pipefail

# Global variables
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly TIMESTAMP=$(date +%Y%m%d_%H%M%S)
readonly ROTATION_LOCK_FILE="/var/run/tald-unia-rotation.lock"
readonly LOG_DIR="/var/log/tald-unia/rotation"
readonly LOG_FILE="${LOG_DIR}/rotation-${TIMESTAMP}.log"

# Environment validation
[[ -z "${AWS_REGION}" ]] && { echo "ERROR: AWS_REGION not set"; exit 1; }
[[ -z "${VAULT_ADDR}" ]] && { echo "ERROR: VAULT_ADDR not set"; exit 1; }
[[ -z "${ENVIRONMENT}" ]] && { echo "ERROR: ENVIRONMENT not set"; exit 1; }

# Initialize logging
setup_logging() {
    mkdir -p "${LOG_DIR}"
    exec 1> >(tee -a "${LOG_FILE}")
    exec 2> >(tee -a "${LOG_FILE}" >&2)
    echo "=== Secret rotation started at $(date) ==="
}

# Acquire rotation lock
acquire_lock() {
    if ! mkdir "${ROTATION_LOCK_FILE}" 2>/dev/null; then
        echo "ERROR: Another rotation process is running"
        exit 1
    }
    trap 'cleanup' EXIT
}

# Cleanup function
cleanup() {
    rm -rf "${ROTATION_LOCK_FILE}"
    echo "=== Secret rotation completed at $(date) ==="
}

# Verify Vault HA cluster health
verify_ha_status() {
    local cluster_name="$1"
    echo "Verifying Vault HA cluster health..."
    
    # Check if HA is enabled
    local ha_enabled=$(vault read sys/ha-status -format=json | jq -r '.data.ha_enabled')
    [[ "${ha_enabled}" != "true" ]] && { echo "ERROR: HA not enabled"; return 1; }
    
    # Verify all replicas are healthy
    local unhealthy_pods=$(kubectl get pods -l app.kubernetes.io/name=vault \
        -n vault --no-headers | grep -v "Running" | wc -l)
    [[ "${unhealthy_pods}" -gt 0 ]] && { echo "ERROR: Unhealthy Vault pods detected"; return 1; }
    
    # Check replication status
    vault read -format=json sys/replication/status | jq -e '.data.ok' >/dev/null || \
        { echo "ERROR: Replication check failed"; return 1; }
    
    echo "Vault HA cluster health verified"
    return 0
}

# Rotate AWS KMS keys
rotate_kms_keys() {
    local key_id="$1"
    echo "Rotating KMS key: ${key_id}"
    
    # Verify AWS credentials
    aws sts get-caller-identity >/dev/null || { echo "ERROR: AWS authentication failed"; return 1; }
    
    # Enable key rotation
    aws kms enable-key-rotation --key-id "${key_id}" || \
        { echo "ERROR: Failed to enable key rotation"; return 1; }
    
    # Create rotation snapshot
    local snapshot_id=$(aws kms create-key-rotation-snapshot \
        --key-id "${key_id}" --query 'SnapshotId' --output text)
    
    # Wait for rotation completion
    aws kms wait key-rotated --key-id "${key_id}" || \
        { echo "ERROR: Key rotation failed"; return 1; }
    
    # Verify new key version
    aws kms get-key-rotation-status --key-id "${key_id}" | \
        jq -e '.KeyRotationEnabled' >/dev/null || { echo "ERROR: Key rotation verification failed"; return 1; }
    
    echo "KMS key rotation completed successfully"
    return 0
}

# Rotate Vault encryption keys
rotate_vault_keys() {
    local mount_point="$1"
    echo "Rotating Vault keys for mount: ${mount_point}"
    
    # Create pre-rotation backup
    local backup_path="${LOG_DIR}/vault-${TIMESTAMP}.snap"
    vault operator raft snapshot save "${backup_path}" || \
        { echo "ERROR: Backup failed"; return 1; }
    
    # Rotate encryption key
    vault write -f "sys/rotate/${mount_point}" || \
        { echo "ERROR: Key rotation failed"; return 1; }
    
    # Wait for replication sync
    sleep 10
    
    # Verify rotation
    vault read "sys/key-status" | grep -q "Term" || \
        { echo "ERROR: Key rotation verification failed"; return 1; }
    
    echo "Vault key rotation completed successfully"
    return 0
}

# Main rotation function
rotate_all_secrets() {
    local retries=3
    local retry_delay=5
    
    echo "Starting secret rotation for environment: ${ENVIRONMENT}"
    
    # Verify HA cluster health
    verify_ha_status "vault" || { echo "ERROR: HA cluster health check failed"; exit 1; }
    
    # Rotate KMS keys with retries
    local attempt=1
    while [[ ${attempt} -le ${retries} ]]; do
        if rotate_kms_keys "alias/tald-unia-${ENVIRONMENT}"; then
            break
        fi
        echo "Retry ${attempt}/${retries} for KMS key rotation..."
        sleep $((retry_delay * attempt))
        ((attempt++))
    done
    [[ ${attempt} -gt ${retries} ]] && { echo "ERROR: KMS key rotation failed after ${retries} attempts"; exit 1; }
    
    # Rotate Vault keys with retries
    attempt=1
    while [[ ${attempt} -le ${retries} ]]; do
        if rotate_vault_keys "transit"; then
            break
        fi
        echo "Retry ${attempt}/${retries} for Vault key rotation..."
        sleep $((retry_delay * attempt))
        ((attempt++))
    done
    [[ ${attempt} -gt ${retries} ]] && { echo "ERROR: Vault key rotation failed after ${retries} attempts"; exit 1; }
    
    echo "Secret rotation completed successfully"
}

# Main execution
main() {
    setup_logging
    acquire_lock
    rotate_all_secrets
}

main "$@"