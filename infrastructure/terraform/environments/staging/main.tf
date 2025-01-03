# Core Terraform configuration
terraform {
  required_version = ">= 1.0.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 4.0"
    }
  }

  backend "s3" {
    bucket         = "tald-unia-terraform-state"
    key            = "staging/terraform.tfstate"
    region         = "us-west-2"
    encrypt        = true
    dynamodb_table = "terraform-state-lock"
    kms_key_id     = "arn:aws:kms:us-west-2:123456789012:key/tald-unia-terraform-key"
  }
}

# Local variables for environment configuration
locals {
  environment                  = "staging"
  aws_region                  = "us-west-2"
  domain_name                 = "staging.tald-unia.com"
  monitoring_namespace        = "monitoring-staging"
  lidar_processing_namespace  = "lidar-staging"
  fleet_coordination_namespace = "fleet-staging"
  game_processing_namespace   = "game-staging"

  common_tags = {
    Environment = local.environment
    Project     = "TALD-UNIA"
    ManagedBy   = "Terraform"
    Component   = "Staging Infrastructure"
  }
}

# AWS Core Infrastructure Module
module "aws_infrastructure" {
  source = "../../aws"

  environment                  = local.environment
  aws_region                  = local.aws_region
  lidar_processing_namespace  = local.lidar_processing_namespace
  fleet_coordination_namespace = local.fleet_coordination_namespace
  game_processing_namespace   = local.game_processing_namespace

  # EKS Configuration for LiDAR Processing
  cluster_version     = "1.27"
  node_instance_types = ["t3.2xlarge"]  # For 30Hz LiDAR processing
  node_desired_size   = 3
  node_min_size      = 2
  node_max_size      = 5

  # Redis Configuration for Fleet State
  redis_node_type       = "cache.t3.medium"
  redis_num_cache_nodes = 3

  # DynamoDB Configuration
  dynamodb_billing_mode = "PAY_PER_REQUEST"

  tags = local.common_tags
}

# Monitoring Infrastructure Module
module "monitoring" {
  source = "../../modules/monitoring"

  environment           = local.environment
  cluster_name         = module.aws_infrastructure.eks_cluster_name
  monitoring_namespace = local.monitoring_namespace

  # LiDAR-specific monitoring configuration
  performance_thresholds = {
    lidar_scan_rate_hz   = 30  # Required 30Hz scan rate
    network_latency_ms   = 50  # Maximum 50ms latency
    ui_framerate_min_fps = 60  # Minimum 60 FPS
  }

  # Resource allocation for monitoring components
  monitoring_resources = {
    prometheus_cpu       = "1000m"
    prometheus_memory    = "2Gi"
    grafana_cpu         = "500m"
    grafana_memory      = "1Gi"
    elasticsearch_cpu   = "2000m"
    elasticsearch_memory = "4Gi"
  }

  # Storage configuration
  storage_config = {
    prometheus_storage_size    = "50Gi"
    elasticsearch_storage_size = "100Gi"
    storage_class             = "gp2"
  }

  prometheus_retention_days = 30
  log_retention_days       = 30
  enable_alerting         = true
}

# Security Infrastructure Module
module "security" {
  source = "../../modules/security"

  environment     = local.environment
  aws_region     = local.aws_region
  domain_name    = local.domain_name
  vpc_id         = module.aws_infrastructure.vpc_id

  # Security-specific configuration
  key_deletion_window = 30
  enable_key_rotation = true
  allowed_cidr_blocks = ["0.0.0.0/0"]  # Restrict in production

  # Additional security features for LiDAR data protection
  kms_key_policy = {
    enable_key_rotation = true
    deletion_window    = 30
  }
}

# Outputs for cross-module reference
output "vpc_id" {
  description = "VPC ID for the staging environment"
  value       = module.aws_infrastructure.vpc_id
}

output "monitoring_endpoints" {
  description = "Monitoring service endpoints"
  value = {
    prometheus = module.monitoring.prometheus_endpoint
    grafana    = module.monitoring.grafana_endpoint
    lidar_metrics = module.monitoring.lidar_metrics_endpoint
  }
}

output "security_resources" {
  description = "Security resource identifiers"
  value = {
    certificate_arn       = module.security.certificate_arn
    kms_key_arn          = module.security.kms_key_arn
    fleet_security_group_id = module.security.fleet_security_group_id
  }
  sensitive = true
}