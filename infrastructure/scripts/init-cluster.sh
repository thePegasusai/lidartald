#!/bin/bash

# TALD UNIA Kubernetes Cluster Initialization Script
# Version: 1.0.0
# This script initializes and configures a Kubernetes cluster for the TALD UNIA platform
# with specialized support for LiDAR processing, fleet mesh networking, and GPU workloads.

set -euo pipefail

# Global variables
CLUSTER_NAME="${ENVIRONMENT}-tald-unia-cluster"
NAMESPACE="tald-unia"
MONITORING_NAMESPACE="monitoring"
GPU_RESOURCE_QUOTA="nvidia.com/gpu: 4"
MESH_MAX_NODES="32"

# Function to check prerequisites
check_prerequisites() {
    command -v kubectl >/dev/null 2>&1 || { echo "kubectl is required but not installed. Aborting." >&2; exit 1; }
    command -v helm >/dev/null 2>&1 || { echo "helm is required but not installed. Aborting." >&2; exit 1; }
    command -v aws >/dev/null 2>&1 || { echo "aws-cli is required but not installed. Aborting." >&2; exit 1; }
}

# Function to initialize cluster connection
init_cluster_connection() {
    echo "Configuring cluster connection..."
    aws eks update-kubeconfig --name "${CLUSTER_NAME}" --region "${REGION}"
    kubectl cluster-info || { echo "Failed to connect to cluster. Aborting." >&2; exit 1; }
}

# Function to create namespaces
create_namespaces() {
    echo "Creating namespaces..."
    kubectl create namespace "${NAMESPACE}" --dry-run=client -o yaml | kubectl apply -f -
    kubectl create namespace "${MONITORING_NAMESPACE}" --dry-run=client -o yaml | kubectl apply -f -
    
    # Apply resource quotas for GPU workloads
    cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: ResourceQuota
metadata:
  name: gpu-quota
  namespace: ${NAMESPACE}
spec:
  hard:
    ${GPU_RESOURCE_QUOTA}
EOF
}

# Function to install NVIDIA GPU operator
install_gpu_operator() {
    echo "Installing NVIDIA GPU operator..."
    helm repo add nvidia https://nvidia.github.io/gpu-operator
    helm repo update
    
    helm upgrade --install gpu-operator nvidia/gpu-operator \
        --namespace "${NAMESPACE}" \
        --set driver.enabled=true \
        --set toolkit.enabled=true \
        --set devicePlugin.enabled=true \
        --version 1.11.1
}

# Function to install cert-manager
install_cert_manager() {
    echo "Installing cert-manager..."
    helm repo add jetstack https://charts.jetstack.io
    helm repo update
    
    helm upgrade --install cert-manager jetstack/cert-manager \
        --namespace "${NAMESPACE}" \
        --set installCRDs=true \
        --version v1.11.0
}

# Function to setup monitoring stack
setup_monitoring() {
    echo "Setting up monitoring stack..."
    helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
    helm repo update
    
    # Install Prometheus with LiDAR-specific configurations
    helm upgrade --install prometheus prometheus-community/kube-prometheus-stack \
        --namespace "${MONITORING_NAMESPACE}" \
        --values ../helm/monitoring/prometheus/values.yaml \
        --set prometheus.prometheusSpec.serviceMonitorSelectorNilUsesHelmValues=false
        
    # Configure custom metrics for LiDAR and fleet operations
    kubectl apply -f - <<EOF
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: lidar-metrics
  namespace: ${MONITORING_NAMESPACE}
spec:
  selector:
    matchLabels:
      app: lidar-core
  endpoints:
  - port: metrics
EOF
}

# Function to install Redis operator
install_redis_operator() {
    echo "Installing Redis operator..."
    helm repo add redis-operator https://spotahome.github.io/redis-operator
    helm repo update
    
    helm upgrade --install redis-operator redis-operator/redis-operator \
        --namespace "${NAMESPACE}" \
        --set image.tag=v1.2.0 \
        --set resources.limits.cpu=1 \
        --set resources.limits.memory=1Gi
}

# Function to configure network policies
configure_network_policies() {
    echo "Configuring network policies..."
    kubectl apply -f - <<EOF
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: fleet-mesh-policy
  namespace: ${NAMESPACE}
spec:
  podSelector:
    matchLabels:
      app: fleet-manager
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - podSelector:
        matchLabels:
          app: fleet-manager
  egress:
  - to:
    - podSelector:
        matchLabels:
          app: fleet-manager
EOF
}

# Function to deploy core services
deploy_core_services() {
    echo "Deploying core TALD UNIA services..."
    helm upgrade --install tald-unia ../helm/tald-unia \
        --namespace "${NAMESPACE}" \
        --values ../helm/tald-unia/values.yaml \
        --set global.environment="${ENVIRONMENT}" \
        --set fleetManager.meshNetwork.maxDevices="${MESH_MAX_NODES}"
}

# Function to setup RBAC
setup_rbac() {
    echo "Setting up RBAC..."
    kubectl apply -f - <<EOF
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  name: fleet-manager-role
rules:
- apiGroups: [""]
  resources: ["pods", "services"]
  verbs: ["get", "list", "watch"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRoleBinding
metadata:
  name: fleet-manager-binding
subjects:
- kind: ServiceAccount
  name: fleet-manager
  namespace: ${NAMESPACE}
roleRef:
  kind: ClusterRole
  name: fleet-manager-role
  apiGroup: rbac.authorization.k8s.io
EOF
}

# Main initialization function
init_cluster() {
    local environment=$1
    local region=$2
    
    export ENVIRONMENT="${environment}"
    export REGION="${region}"
    
    echo "Initializing TALD UNIA cluster in ${ENVIRONMENT} environment..."
    
    check_prerequisites
    init_cluster_connection
    create_namespaces
    install_gpu_operator
    install_cert_manager
    setup_monitoring
    install_redis_operator
    configure_network_policies
    deploy_core_services
    setup_rbac
    
    echo "Cluster initialization completed successfully!"
}

# Script execution
if [[ $# -ne 2 ]]; then
    echo "Usage: $0 <environment> <region>"
    exit 1
fi

init_cluster "$1" "$2"