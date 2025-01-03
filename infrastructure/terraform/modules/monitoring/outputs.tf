# Output monitoring service endpoints
output "monitoring_endpoints" {
  description = "Map of monitoring service endpoints"
  value = {
    prometheus    = "${helm_release_prometheus.metadata[0].name}.${var.monitoring_namespace}.svc.cluster.local"
    grafana      = "${helm_release_grafana.metadata[0].name}.${var.monitoring_namespace}.svc.cluster.local"
    elasticsearch = "${helm_release_elasticsearch.metadata[0].name}.${var.monitoring_namespace}.svc.cluster.local"
  }
  sensitive = false
}

# Output monitoring stack configuration details
output "monitoring_config" {
  description = "Monitoring stack configuration details"
  value = {
    namespace        = var.monitoring_namespace
    retention_days   = var.prometheus_retention_days
    scrape_interval = var.metrics_scrape_interval
    performance_targets = {
      lidar_scan_rate = "30Hz"
      network_latency = "50ms"
      ui_framerate    = "60fps"
    }
  }
  sensitive = false
}

# Output monitoring stack deployment status
output "monitoring_status" {
  description = "Status of monitoring stack components"
  value = {
    prometheus_ready    = helm_release_prometheus.status == "deployed"
    grafana_ready      = helm_release_grafana.status == "deployed"
    elasticsearch_ready = helm_release_elasticsearch.status == "deployed"
  }
  sensitive = false
}