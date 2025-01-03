# Core Terraform configuration
terraform {
  required_version = ">= 1.0.0"

  # Configure S3 backend for production state
  backend "s3" {
    bucket         = "tald-unia-terraform-state-prod"
    key            = "production/terraform.tfstate"
    region         = "us-west-2"
    encrypt        = true
    dynamodb_table = "terraform-state-lock-prod"
  }
}

# Define local variables
locals {
  environment           = "production"
  aws_region           = "us-west-2"
  domain_name          = "prod.tald-unia.com"
  monitoring_namespace = "monitoring-prod"
  retention_days       = 90
  alert_channels       = ["email", "pagerduty"]
  high_availability    = true
  cross_zone_lb_enabled = true
}

# Core AWS infrastructure module
module "aws_infrastructure" {
  source = "../../aws"
  
  environment            = local.environment
  aws_region            = local.aws_region
  high_availability     = true
  auto_scaling          = true
  cross_zone_lb_enabled = local.cross_zone_lb_enabled

  # EKS configuration
  cluster_version     = "1.27"
  node_instance_types = ["t3.2xlarge"]
  node_desired_size   = 3
  node_min_size      = 2
  node_max_size      = 5

  # Redis configuration for fleet coordination
  redis_node_type       = "cache.r6g.xlarge"
  redis_num_cache_nodes = 3

  # DynamoDB configuration
  dynamodb_billing_mode = "PROVISIONED"

  # S3 and CloudFront configuration
  s3_versioning         = true
  cloudfront_price_class = "PriceClass_All"

  # Cognito configuration
  cognito_password_policy = {
    minimum_length    = 12
    require_lowercase = true
    require_numbers   = true
    require_symbols   = true
    require_uppercase = true
  }
}

# Monitoring infrastructure module
module "monitoring" {
  source = "../../modules/monitoring"

  environment          = local.environment
  cluster_name         = module.aws_infrastructure.eks_cluster_name
  monitoring_namespace = local.monitoring_namespace
  retention_days       = local.retention_days
  alert_channels       = local.alert_channels

  # Performance monitoring thresholds
  performance_thresholds = {
    lidar_scan_rate_hz   = 30  # 30Hz scan rate requirement
    network_latency_ms   = 50  # <50ms latency requirement
    ui_framerate_min_fps = 60  # 60 FPS UI responsiveness
  }

  # Resource allocation for monitoring components
  monitoring_resources = {
    prometheus_cpu       = "2000m"
    prometheus_memory    = "4Gi"
    grafana_cpu         = "1000m"
    grafana_memory      = "2Gi"
    elasticsearch_cpu   = "4000m"
    elasticsearch_memory = "8Gi"
  }

  # Storage configuration
  storage_config = {
    prometheus_storage_size    = "500Gi"
    elasticsearch_storage_size = "1000Gi"
    storage_class             = "gp3"
  }

  # Alert configuration from variables
  custom_thresholds = var.alert_thresholds
}

# Security infrastructure module
module "security" {
  source = "../../modules/security"

  environment         = local.environment
  domain_name        = local.domain_name
  vpc_id             = module.aws_infrastructure.vpc_id
  
  # Enhanced security settings for production
  enable_key_rotation   = true
  key_deletion_window  = 30
  
  # Security policies from variables
  security_policies    = var.security_policies

  # Production-specific CIDR blocks
  allowed_cidr_blocks = [
    "10.0.0.0/8",    # Internal VPC CIDR
    "172.16.0.0/12", # VPN CIDR
    "192.168.0.0/16" # Management CIDR
  ]
}

# Output critical infrastructure information
output "vpc_id" {
  description = "Production VPC identifier"
  value       = module.aws_infrastructure.vpc_id
}

output "monitoring_endpoints" {
  description = "Production monitoring service endpoints"
  value = {
    prometheus    = module.monitoring.prometheus_endpoint
    grafana       = module.monitoring.grafana_endpoint
    alert_manager = module.monitoring.alert_manager_endpoint
  }
}

output "security_resources" {
  description = "Production security resource identifiers"
  value = {
    certificate_arn    = module.security.certificate_arn
    kms_key_arn       = module.security.kms_key_arn
    security_group_ids = module.security.security_group_ids
  }
  sensitive = true
}