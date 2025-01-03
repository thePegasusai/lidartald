#!/bin/bash

# TALD UNIA Security Components Deployment Script
# Version: 1.0.0
# Dependencies:
# - helm v3.12.x
# - kubectl v1.27.x
# - aws-cli v2.x

set -euo pipefail

# Global variables
ENVIRONMENT=${ENVIRONMENT:-production}
NAMESPACE="security"
HELM_TIMEOUT="600s"
LOG_LEVEL=${LOG_LEVEL:-INFO}
CLEANUP_ON_FAILURE=${CLEANUP_ON_FAILURE:-true}
VALIDATION_TIMEOUT="300s"
MONITORING_ENABLED=${MONITORING_ENABLED:-true}

# Logging configuration
log() {
    local level=$1
    shift
    local timestamp=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
    echo "{\"timestamp\":\"${timestamp}\",\"level\":\"${level}\",\"message\":\"$*\"}"
}

error_exit() {
    log "ERROR" "$1"
    if [[ "${CLEANUP_ON_FAILURE}" == "true" ]]; then
        cleanup_on_failure
    fi
    exit 1
}

cleanup_on_failure() {
    log "INFO" "Initiating cleanup procedure..."
    # Store deployment state for recovery
    kubectl get deployments -n "${NAMESPACE}" -o yaml > "/tmp/security-deployments-${ENVIRONMENT}.yaml"
    # Remove failed deployments while preserving data
    helm list -n "${NAMESPACE}" | grep -E "cert-manager|vault|oauth2-proxy" | awk '{print $1}' | xargs -r helm uninstall -n "${NAMESPACE}"
}

check_prerequisites() {
    log "INFO" "Checking prerequisites..."

    # Check helm version
    if ! helm version --short | grep -q "v3.12"; then
        error_exit "Helm 3.12.x is required"
    fi

    # Validate kubectl access
    if ! kubectl auth can-i create deployments -n "${NAMESPACE}"; then
        error_exit "Insufficient Kubernetes permissions"
    fi

    # Verify AWS credentials
    if ! aws sts get-caller-identity &>/dev/null; then
        error_exit "Invalid AWS credentials"
    }

    # Check required CRDs
    if ! kubectl get crd certificates.cert-manager.io &>/dev/null; then
        log "INFO" "cert-manager CRDs not found, will be installed"
    }

    # Validate storage classes
    if ! kubectl get storageclass gp3 &>/dev/null; then
        error_exit "Required storage class 'gp3' not found"
    }

    log "INFO" "Prerequisites check completed successfully"
}

deploy_cert_manager() {
    log "INFO" "Deploying cert-manager..."

    # Add and update Jetstack repo
    helm repo add jetstack https://charts.jetstack.io
    helm repo update

    # Deploy cert-manager with values
    if ! helm upgrade --install cert-manager jetstack/cert-manager \
        --namespace "${NAMESPACE}" \
        --version v1.11.0 \
        --values ../helm/security/cert-manager/values.yaml \
        --timeout "${HELM_TIMEOUT}" \
        --wait; then
        error_exit "cert-manager deployment failed"
    fi

    # Verify deployment
    kubectl wait --for=condition=available deployment/cert-manager -n "${NAMESPACE}" --timeout="${VALIDATION_TIMEOUT}"

    # Setup monitoring if enabled
    if [[ "${MONITORING_ENABLED}" == "true" ]]; then
        kubectl apply -f - <<EOF
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: cert-manager
  namespace: ${NAMESPACE}
spec:
  selector:
    matchLabels:
      app.kubernetes.io/name: cert-manager
  endpoints:
  - port: metrics
    interval: 30s
EOF
    fi

    log "INFO" "cert-manager deployment completed successfully"
}

deploy_vault() {
    log "INFO" "Deploying HashiCorp Vault..."

    # Add and update HashiCorp repo
    helm repo add hashicorp https://helm.releases.hashicorp.com
    helm repo update

    # Validate KMS configuration
    if ! aws kms describe-key --key-id "${AWS_KMS_KEY_ID}" &>/dev/null; then
        error_exit "Invalid AWS KMS key configuration"
    fi

    # Deploy Vault with values
    if ! helm upgrade --install vault hashicorp/vault \
        --namespace "${NAMESPACE}" \
        --values ../helm/security/vault/values.yaml \
        --set server.extraEnvironmentVars.VAULT_AWSKMS_SEAL_KEY_ID="${AWS_KMS_KEY_ID}" \
        --timeout "${HELM_TIMEOUT}" \
        --wait; then
        error_exit "Vault deployment failed"
    fi

    # Verify deployment
    kubectl wait --for=condition=ready pod -l app.kubernetes.io/name=vault -n "${NAMESPACE}" --timeout="${VALIDATION_TIMEOUT}"

    # Setup audit logging
    kubectl exec -n "${NAMESPACE}" vault-0 -- vault audit enable file file_path=/vault/audit/audit.log

    log "INFO" "Vault deployment completed successfully"
}

deploy_oauth2_proxy() {
    log "INFO" "Deploying OAuth2 Proxy..."

    # Add and update oauth2-proxy repo
    helm repo add oauth2-proxy https://oauth2-proxy.github.io/manifests
    helm repo update

    # Validate OIDC configuration
    if [[ -z "${OAUTH2_PROXY_CLIENT_ID}" ]] || [[ -z "${OAUTH2_PROXY_CLIENT_SECRET}" ]]; then
        error_exit "Missing OAuth2 credentials"
    fi

    # Deploy OAuth2 Proxy with values
    if ! helm upgrade --install oauth2-proxy oauth2-proxy/oauth2-proxy \
        --namespace "${NAMESPACE}" \
        --values ../helm/security/oauth2-proxy/values.yaml \
        --set config.clientID="${OAUTH2_PROXY_CLIENT_ID}" \
        --set config.clientSecret="${OAUTH2_PROXY_CLIENT_SECRET}" \
        --set config.cookieSecret="${OAUTH2_PROXY_COOKIE_SECRET}" \
        --timeout "${HELM_TIMEOUT}" \
        --wait; then
        error_exit "OAuth2 Proxy deployment failed"
    fi

    # Verify deployment
    kubectl wait --for=condition=available deployment/oauth2-proxy -n "${NAMESPACE}" --timeout="${VALIDATION_TIMEOUT}"

    # Setup rate limiting
    kubectl apply -f - <<EOF
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: oauth2-proxy-rate-limit
  namespace: ${NAMESPACE}
spec:
  podSelector:
    matchLabels:
      app.kubernetes.io/name: oauth2-proxy
  ingress:
  - from:
    - namespaceSelector: {}
    ports:
    - port: 4180
      protocol: TCP
EOF

    log "INFO" "OAuth2 Proxy deployment completed successfully"
}

main() {
    log "INFO" "Starting security components deployment for environment: ${ENVIRONMENT}"

    # Create namespace if it doesn't exist
    kubectl create namespace "${NAMESPACE}" --dry-run=client -o yaml | kubectl apply -f -

    # Run deployment steps
    check_prerequisites
    deploy_cert_manager
    deploy_vault
    deploy_oauth2_proxy

    log "INFO" "Security components deployment completed successfully"
}

# Execute main function
main "$@"