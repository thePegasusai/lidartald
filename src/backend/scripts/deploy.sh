#!/bin/bash

# TALD UNIA Backend Deployment Script
# Version: 1.0.0
# Dependencies:
# - kubectl v1.27
# - helm v3.12

set -euo pipefail

# Global Constants
readonly DEPLOYMENT_TIMEOUT=900  # 15 minutes SLA
readonly HEALTH_CHECK_INTERVAL=10
readonly MAX_RETRY_ATTEMPTS=3
readonly REQUIRED_SERVICES='["database", "redis", "fleet-manager", "game-engine", "lidar-core"]'
readonly PERFORMANCE_THRESHOLDS='{"scan_rate": 30, "ui_fps": 60, "network_latency": 50}'
readonly RESOURCE_REQUIREMENTS='{"gpu": "nvidia-tesla-v100", "cuda_version": "11.8"}'

# Logging functions
log_info() {
    echo "[INFO] $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

log_error() {
    echo "[ERROR] $(date '+%Y-%m-%d %H:%M:%S') - $1" >&2
}

# Deploy database service
deploy_database() {
    local namespace=$1
    local environment=$2
    
    log_info "Deploying database service in namespace: $namespace"
    
    # Validate storage class
    if ! kubectl get storageclass gp2 &>/dev/null; then
        log_error "Required storage class 'gp2' not found"
        return 1
    }
    
    # Apply database configuration
    kubectl apply -f src/backend/k8s/database.yaml -n "$namespace"
    
    # Wait for StatefulSet rollout
    if ! kubectl rollout status statefulset/tald-db -n "$namespace" --timeout="${DEPLOYMENT_TIMEOUT}s"; then
        log_error "Database StatefulSet rollout failed"
        return 1
    }
    
    # Verify database health
    if ! kubectl exec -n "$namespace" tald-db-0 -- sqlite3 -version &>/dev/null; then
        log_error "Database health check failed"
        return 1
    }
    
    log_info "Database deployment successful"
    return 0
}

# Deploy fleet manager service
deploy_fleet_manager() {
    local namespace=$1
    local environment=$2
    
    log_info "Deploying fleet manager service in namespace: $namespace"
    
    # Apply fleet manager configuration
    kubectl apply -f src/backend/k8s/fleet-manager.yaml -n "$namespace"
    
    # Wait for deployment rollout
    if ! kubectl rollout status deployment/fleet-manager -n "$namespace" --timeout="${DEPLOYMENT_TIMEOUT}s"; then
        log_error "Fleet manager deployment rollout failed"
        return 1
    }
    
    # Verify mesh network connectivity
    if ! kubectl exec -n "$namespace" deploy/fleet-manager -- curl -s http://localhost:8080/health | grep -q "ok"; then
        log_error "Fleet manager health check failed"
        return 1
    }
    
    log_info "Fleet manager deployment successful"
    return 0
}

# Deploy game engine service
deploy_game_engine() {
    local namespace=$1
    local environment=$2
    
    log_info "Deploying game engine service in namespace: $namespace"
    
    # Verify GPU node availability
    if ! kubectl get nodes -l gpu=nvidia-tesla &>/dev/null; then
        log_error "No GPU nodes available"
        return 1
    }
    
    # Apply game engine configuration
    kubectl apply -f src/backend/k8s/game-engine.yaml -n "$namespace"
    
    # Wait for deployment rollout
    if ! kubectl rollout status deployment/game-engine -n "$namespace" --timeout="${DEPLOYMENT_TIMEOUT}s"; then
        log_error "Game engine deployment rollout failed"
        return 1
    }
    
    # Verify Vulkan compatibility
    if ! kubectl exec -n "$namespace" deploy/game-engine -- vulkaninfo | grep -q "Vulkan Instance Version"; then
        log_error "Game engine Vulkan verification failed"
        return 1
    }
    
    log_info "Game engine deployment successful"
    return 0
}

# Deploy LiDAR core service
deploy_lidar_core() {
    local namespace=$1
    local environment=$2
    
    log_info "Deploying LiDAR core service in namespace: $namespace"
    
    # Apply LiDAR core configuration
    kubectl apply -f src/backend/k8s/lidar-core.yaml -n "$namespace"
    
    # Wait for deployment rollout
    if ! kubectl rollout status deployment/lidar-core -n "$namespace" --timeout="${DEPLOYMENT_TIMEOUT}s"; then
        log_error "LiDAR core deployment rollout failed"
        return 1
    }
    
    # Verify scanning rate
    if ! kubectl exec -n "$namespace" deploy/lidar-core -- curl -s http://localhost:8080/metrics | grep -q "scan_rate.*30"; then
        log_error "LiDAR core performance verification failed"
        return 1
    }
    
    log_info "LiDAR core deployment successful"
    return 0
}

# Verify deployment status
verify_deployment() {
    local namespace=$1
    local service_name=$2
    
    log_info "Verifying deployment status for $service_name"
    
    # Check pod status
    local ready_pods
    ready_pods=$(kubectl get pods -n "$namespace" -l app="$service_name" -o jsonpath='{.items[*].status.containerStatuses[*].ready}' | tr ' ' '\n' | grep -c "true")
    local total_pods
    total_pods=$(kubectl get pods -n "$namespace" -l app="$service_name" --no-headers | wc -l)
    
    if [ "$ready_pods" -ne "$total_pods" ]; then
        log_error "Not all pods are ready for $service_name"
        return 1
    }
    
    # Verify service endpoints
    if ! kubectl get endpoints -n "$namespace" "$service_name" -o jsonpath='{.subsets[*].addresses[*]}' | grep -q .; then
        log_error "No endpoints available for $service_name"
        return 1
    }
    
    # Check resource allocation
    if ! kubectl describe pods -n "$namespace" -l app="$service_name" | grep -q "Resources"; then
        log_error "Resource allocation verification failed for $service_name"
        return 1
    }
    
    return 0
}

# Rollback deployment
rollback_deployment() {
    local namespace=$1
    local service_name=$2
    local revision=$3
    
    log_info "Rolling back $service_name to revision $revision"
    
    # Stop ongoing deployment
    kubectl rollout pause deployment/"$service_name" -n "$namespace"
    
    # Rollback to previous version
    if ! kubectl rollout undo deployment/"$service_name" -n "$namespace" --to-revision="$revision"; then
        log_error "Rollback failed for $service_name"
        return 1
    }
    
    # Wait for rollback completion
    if ! kubectl rollout status deployment/"$service_name" -n "$namespace" --timeout="${DEPLOYMENT_TIMEOUT}s"; then
        log_error "Rollback status check failed for $service_name"
        return 1
    }
    
    # Resume deployment
    kubectl rollout resume deployment/"$service_name" -n "$namespace"
    
    log_info "Rollback completed successfully for $service_name"
    return 0
}

# Main deployment function
deploy_all() {
    local namespace=$1
    local environment=$2
    
    log_info "Starting deployment process for environment: $environment"
    
    # Create namespace if it doesn't exist
    kubectl create namespace "$namespace" --dry-run=client -o yaml | kubectl apply -f -
    
    # Deploy services in sequence
    services=(
        "deploy_database"
        "deploy_fleet_manager"
        "deploy_game_engine"
        "deploy_lidar_core"
    )
    
    for service in "${services[@]}"; do
        if ! $service "$namespace" "$environment"; then
            log_error "Deployment failed for $service"
            return 1
        fi
        
        # Verify deployment
        service_name=${service#deploy_}
        if ! verify_deployment "$namespace" "$service_name"; then
            log_error "Deployment verification failed for $service_name"
            return 1
        fi
    done
    
    log_info "All services deployed successfully"
    return 0
}

# Script entry point
main() {
    if [ "$#" -ne 2 ]; then
        echo "Usage: $0 <namespace> <environment>"
        exit 1
    }
    
    local namespace=$1
    local environment=$2
    
    # Validate environment
    case "$environment" in
        development|staging|production)
            ;;
        *)
            log_error "Invalid environment. Must be one of: development, staging, production"
            exit 1
            ;;
    esac
    
    # Check kubectl access
    if ! kubectl cluster-info &>/dev/null; then
        log_error "Unable to access Kubernetes cluster"
        exit 1
    }
    
    # Execute deployment
    if ! deploy_all "$namespace" "$environment"; then
        log_error "Deployment failed"
        exit 1
    fi
    
    log_info "Deployment completed successfully"
}

# Execute main function
main "$@"