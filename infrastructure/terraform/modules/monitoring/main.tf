# Core Terraform configuration with required providers
terraform {
  required_version = "~> 1.0"
  
  required_providers {
    helm = {
      source  = "hashicorp/helm"
      version = "~> 2.9"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.20"
    }
  }

  backend "s3" {
    # Backend configuration should be provided during terraform init
  }
}

# Configure Helm provider
provider "helm" {
  kubernetes {
    config_path = "~/.kube/config"
    cluster_name = var.cluster_name
  }
}

# Configure Kubernetes provider
provider "kubernetes" {
  config_path = "~/.kube/config"
  cluster_name = var.cluster_name
}

# Create dedicated monitoring namespace
resource "kubernetes_namespace" "monitoring" {
  metadata {
    name = var.monitoring_namespace
    
    labels = {
      environment = var.environment
      "app.kubernetes.io/managed-by" = "terraform"
      "monitoring.tald.unia/enabled" = "true"
    }

    annotations = {
      "monitoring.tald.unia/description" = "Namespace for TALD UNIA monitoring infrastructure"
    }
  }
}

# Deploy Prometheus stack using Helm
resource "helm_release" "prometheus" {
  name       = "prometheus"
  repository = "https://prometheus-community.github.io/helm-charts"
  chart      = "kube-prometheus-stack"
  namespace  = kubernetes_namespace.monitoring.metadata[0].name
  version    = "45.7.1"  # Specify exact version for production stability
  
  values = [
    file("../../helm/monitoring/prometheus/values.yaml")
  ]

  set {
    name  = "prometheus.retention"
    value = "${var.prometheus_retention_days}d"
  }

  set {
    name  = "prometheus.scrapeInterval"
    value = var.metrics_scrape_interval
  }

  set {
    name  = "prometheus.resources.requests.cpu"
    value = var.monitoring_resources.prometheus_cpu
  }

  set {
    name  = "prometheus.resources.requests.memory"
    value = var.monitoring_resources.prometheus_memory
  }

  set {
    name  = "prometheus.storage.size"
    value = var.storage_config.prometheus_storage_size
  }

  set {
    name  = "prometheus.storage.storageClass"
    value = var.storage_config.storage_class
  }

  set {
    name  = "alertmanager.enabled"
    value = var.enable_alerting
  }
}

# Deploy Grafana using Helm
resource "helm_release" "grafana" {
  name       = "grafana"
  repository = "https://grafana.github.io/helm-charts"
  chart      = "grafana"
  namespace  = kubernetes_namespace.monitoring.metadata[0].name
  version    = "6.50.7"  # Specify exact version for production stability
  
  values = [
    file("../../helm/monitoring/grafana/values.yaml")
  ]

  set {
    name  = "admin.password"
    value = var.grafana_admin_password
  }

  set {
    name  = "resources.requests.cpu"
    value = var.monitoring_resources.grafana_cpu
  }

  set {
    name  = "resources.requests.memory"
    value = var.monitoring_resources.grafana_memory
  }

  # Configure Prometheus data source
  set {
    name  = "datasources.datasources\\.yaml.apiVersion"
    value = "1"
  }

  set {
    name  = "datasources.datasources\\.yaml.datasources[0].name"
    value = "Prometheus"
  }

  set {
    name  = "datasources.datasources\\.yaml.datasources[0].type"
    value = "prometheus"
  }

  set {
    name  = "datasources.datasources\\.yaml.datasources[0].url"
    value = "http://prometheus-server.${kubernetes_namespace.monitoring.metadata[0].name}.svc.cluster.local"
  }
}

# Deploy Elasticsearch using Helm
resource "helm_release" "elasticsearch" {
  name       = "elasticsearch"
  repository = "https://helm.elastic.co"
  chart      = "elasticsearch"
  namespace  = kubernetes_namespace.monitoring.metadata[0].name
  version    = "7.17.3"  # Specify exact version for production stability

  set {
    name  = "retention.days"
    value = var.log_retention_days
  }

  set {
    name  = "resources.requests.cpu"
    value = var.monitoring_resources.elasticsearch_cpu
  }

  set {
    name  = "resources.requests.memory"
    value = var.monitoring_resources.elasticsearch_memory
  }

  set {
    name  = "persistence.size"
    value = var.storage_config.elasticsearch_storage_size
  }

  set {
    name  = "persistence.storageClass"
    value = var.storage_config.storage_class
  }
}

# Output endpoints for external access
output "prometheus_endpoint" {
  description = "Prometheus server endpoint URL"
  value       = "http://prometheus-server.${kubernetes_namespace.monitoring.metadata[0].name}.svc.cluster.local"
}

output "grafana_endpoint" {
  description = "Grafana dashboard endpoint URL"
  value       = "http://grafana.${kubernetes_namespace.monitoring.metadata[0].name}.svc.cluster.local"
}

output "elasticsearch_endpoint" {
  description = "Elasticsearch endpoint URL"
  value       = "http://elasticsearch-master.${kubernetes_namespace.monitoring.metadata[0].name}.svc.cluster.local:9200"
}