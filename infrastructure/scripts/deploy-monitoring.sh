#!/usr/bin/env bash

# TALD UNIA Monitoring Stack Deployment Script
# Version: 1.0.0
# Dependencies:
# - helm v3.12.0
# - kubectl v1.27.0

set -euo pipefail

# Global variables
MONITORING_NAMESPACE="monitoring"
HELM_TIMEOUT="600s"
PROMETHEUS_RELEASE="prometheus"
GRAFANA_RELEASE="grafana"
ELASTICSEARCH_RELEASE="elasticsearch"

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Logging functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Create monitoring namespace with enhanced security
create_monitoring_namespace() {
    log_info "Creating monitoring namespace with security controls..."
    
    if ! kubectl get namespace "$MONITORING_NAMESPACE" &> /dev/null; then
        kubectl create namespace "$MONITORING_NAMESPACE"
        
        # Apply security labels
        kubectl label namespace "$MONITORING_NAMESPACE" \
            pod-security.kubernetes.io/enforce=restricted \
            pod-security.kubernetes.io/audit=restricted \
            pod-security.kubernetes.io/warn=restricted
        
        # Apply network policies
        cat <<EOF | kubectl apply -f -
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: default-deny
  namespace: $MONITORING_NAMESPACE
spec:
  podSelector: {}
  policyTypes:
  - Ingress
  - Egress
EOF
        
        # Apply resource quota
        kubectl apply -f - <<EOF
apiVersion: v1
kind: ResourceQuota
metadata:
  name: monitoring-quota
  namespace: $MONITORING_NAMESPACE
spec:
  hard:
    requests.cpu: "8"
    requests.memory: 16Gi
    limits.cpu: "16"
    limits.memory: 32Gi
EOF
    else
        log_warn "Namespace $MONITORING_NAMESPACE already exists"
    fi
}

# Deploy Prometheus with security hardening
deploy_prometheus() {
    local values_file=$1
    log_info "Deploying Prometheus..."
    
    # Add and update Helm repo
    helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
    helm repo update
    
    # Deploy Prometheus with security configurations
    helm upgrade --install "$PROMETHEUS_RELEASE" prometheus-community/prometheus \
        --namespace "$MONITORING_NAMESPACE" \
        --values "$values_file" \
        --timeout "$HELM_TIMEOUT" \
        --wait \
        --atomic
    
    # Verify deployment
    kubectl rollout status statefulset/"$PROMETHEUS_RELEASE"-server \
        -n "$MONITORING_NAMESPACE" --timeout="$HELM_TIMEOUT"
    
    # Verify ServiceMonitor configuration
    if ! kubectl get servicemonitor -n "$MONITORING_NAMESPACE" | grep -q "$PROMETHEUS_RELEASE"; then
        log_error "ServiceMonitor not found for Prometheus"
        return 1
    fi
}

# Deploy Grafana with enhanced security
deploy_grafana() {
    local values_file=$1
    log_info "Deploying Grafana..."
    
    # Add and update Helm repo
    helm repo add grafana https://grafana.github.io/helm-charts
    helm repo update
    
    # Deploy Grafana with security configurations
    helm upgrade --install "$GRAFANA_RELEASE" grafana/grafana \
        --namespace "$MONITORING_NAMESPACE" \
        --values "$values_file" \
        --timeout "$HELM_TIMEOUT" \
        --wait \
        --atomic
    
    # Verify deployment
    kubectl rollout status deployment/"$GRAFANA_RELEASE" \
        -n "$MONITORING_NAMESPACE" --timeout="$HELM_TIMEOUT"
    
    # Import dashboards
    local dashboard_configmap="grafana-dashboards"
    kubectl create configmap "$dashboard_configmap" \
        --from-file=lidar-metrics.json=./grafana-dashboards/lidar-metrics.json \
        --from-file=fleet-metrics.json=./grafana-dashboards/fleet-metrics.json \
        -n "$MONITORING_NAMESPACE" --dry-run=client -o yaml | kubectl apply -f -
}

# Deploy Elasticsearch with security features
deploy_elasticsearch() {
    local values_file=$1
    log_info "Deploying Elasticsearch..."
    
    # Add and update Helm repo
    helm repo add elastic https://helm.elastic.co
    helm repo update
    
    # Deploy Elasticsearch with security configurations
    helm upgrade --install "$ELASTICSEARCH_RELEASE" elastic/elasticsearch \
        --namespace "$MONITORING_NAMESPACE" \
        --values "$values_file" \
        --timeout "$HELM_TIMEOUT" \
        --wait \
        --atomic
    
    # Verify deployment
    kubectl rollout status statefulset/"$ELASTICSEARCH_RELEASE"-master \
        -n "$MONITORING_NAMESPACE" --timeout="$HELM_TIMEOUT"
    
    # Wait for cluster health
    local retries=0
    while [[ $retries -lt 30 ]]; do
        if kubectl exec -n "$MONITORING_NAMESPACE" "$ELASTICSEARCH_RELEASE"-master-0 \
            -- curl -s -k https://localhost:9200/_cluster/health | grep -q '"status":"green"'; then
            break
        fi
        retries=$((retries + 1))
        sleep 10
    done
}

# Verify monitoring stack health
verify_monitoring_stack() {
    log_info "Verifying monitoring stack health..."
    local failed=0
    
    # Check Prometheus
    if ! kubectl get pods -n "$MONITORING_NAMESPACE" -l "app=prometheus" \
        -o jsonpath='{.items[*].status.containerStatuses[0].ready}' | grep -q "true"; then
        log_error "Prometheus health check failed"
        failed=1
    fi
    
    # Check Grafana
    if ! kubectl get pods -n "$MONITORING_NAMESPACE" -l "app.kubernetes.io/name=grafana" \
        -o jsonpath='{.items[*].status.containerStatuses[0].ready}' | grep -q "true"; then
        log_error "Grafana health check failed"
        failed=1
    fi
    
    # Check Elasticsearch
    if ! kubectl get pods -n "$MONITORING_NAMESPACE" -l "app=elasticsearch-master" \
        -o jsonpath='{.items[*].status.containerStatuses[0].ready}' | grep -q "true"; then
        log_error "Elasticsearch health check failed"
        failed=1
    fi
    
    # Verify metrics collection
    if ! kubectl exec -n "$MONITORING_NAMESPACE" "$PROMETHEUS_RELEASE"-server-0 \
        -- curl -s localhost:9090/api/v1/targets | grep -q '"health":"up"'; then
        log_error "Metrics collection verification failed"
        failed=1
    fi
    
    return $failed
}

# Main deployment function
main() {
    log_info "Starting TALD UNIA monitoring stack deployment..."
    
    # Create namespace with security controls
    create_monitoring_namespace
    
    # Deploy monitoring components
    deploy_prometheus "infrastructure/helm/monitoring/prometheus/values.yaml"
    deploy_grafana "infrastructure/helm/monitoring/grafana/values.yaml"
    deploy_elasticsearch "infrastructure/helm/monitoring/elasticsearch/values.yaml"
    
    # Verify deployment
    if verify_monitoring_stack; then
        log_info "Monitoring stack deployment completed successfully"
    else
        log_error "Monitoring stack deployment verification failed"
        exit 1
    fi
}

# Execute main function
main "$@"