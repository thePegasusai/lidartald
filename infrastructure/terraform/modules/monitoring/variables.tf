# Core environment variables
variable "environment" {
  description = "Deployment environment (staging/production)"
  type        = string
  
  validation {
    condition     = can(regex("^(staging|production)$", var.environment))
    error_message = "Environment must be either staging or production"
  }
}

variable "cluster_name" {
  description = "Name of the EKS cluster where monitoring stack will be deployed"
  type        = string
}

variable "monitoring_namespace" {
  description = "Kubernetes namespace for monitoring components"
  type        = string
  default     = "monitoring"
}

# Prometheus configuration
variable "prometheus_retention_days" {
  description = "Number of days to retain Prometheus metrics"
  type        = number
  default     = 90

  validation {
    condition     = var.prometheus_retention_days >= 30
    error_message = "Prometheus retention must be at least 30 days"
  }
}

variable "metrics_scrape_interval" {
  description = "Interval for scraping metrics from targets"
  type        = string
  default     = "30s"

  validation {
    condition     = can(regex("^[0-9]+(ms|s|m|h)$", var.metrics_scrape_interval))
    error_message = "Metrics scrape interval must be a valid time duration (e.g., 30s, 1m)"
  }
}

# Grafana configuration
variable "grafana_admin_password" {
  description = "Admin password for Grafana dashboard"
  type        = string
  sensitive   = true

  validation {
    condition     = length(var.grafana_admin_password) >= 12
    error_message = "Grafana admin password must be at least 12 characters long"
  }
}

# Alerting configuration
variable "enable_alerting" {
  description = "Enable Prometheus alerting and AlertManager"
  type        = bool
  default     = true
}

# Logging configuration
variable "log_retention_days" {
  description = "Number of days to retain logs in Elasticsearch"
  type        = number
  default     = 30

  validation {
    condition     = var.log_retention_days >= 7
    error_message = "Log retention must be at least 7 days"
  }
}

# Performance monitoring thresholds
variable "performance_thresholds" {
  description = "Performance monitoring thresholds for critical metrics"
  type = object({
    lidar_scan_rate_hz    = number
    network_latency_ms    = number
    ui_framerate_min_fps  = number
  })
  default = {
    lidar_scan_rate_hz    = 30
    network_latency_ms    = 50
    ui_framerate_min_fps  = 60
  }

  validation {
    condition     = var.performance_thresholds.lidar_scan_rate_hz >= 30
    error_message = "LiDAR scan rate must be at least 30Hz"
  }

  validation {
    condition     = var.performance_thresholds.network_latency_ms <= 50
    error_message = "Network latency threshold must not exceed 50ms"
  }

  validation {
    condition     = var.performance_thresholds.ui_framerate_min_fps >= 60
    error_message = "UI framerate threshold must be at least 60 FPS"
  }
}

# Resource allocation
variable "monitoring_resources" {
  description = "Resource allocation for monitoring components"
  type = object({
    prometheus_cpu      = string
    prometheus_memory   = string
    grafana_cpu        = string
    grafana_memory     = string
    elasticsearch_cpu  = string
    elasticsearch_memory = string
  })
  default = {
    prometheus_cpu      = "1000m"
    prometheus_memory   = "2Gi"
    grafana_cpu        = "500m"
    grafana_memory     = "1Gi"
    elasticsearch_cpu  = "2000m"
    elasticsearch_memory = "4Gi"
  }
}

# Storage configuration
variable "storage_config" {
  description = "Storage configuration for monitoring components"
  type = object({
    prometheus_storage_size     = string
    elasticsearch_storage_size  = string
    storage_class              = string
  })
  default = {
    prometheus_storage_size     = "50Gi"
    elasticsearch_storage_size  = "100Gi"
    storage_class              = "gp2"
  }
}