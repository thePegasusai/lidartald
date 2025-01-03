# Core Environment Configuration
environment = "staging"
aws_region  = "us-west-2"

# VPC Configuration
vpc_cidr = "10.0.0.0/16"

# EKS Cluster Configuration
cluster_version     = "1.27"
node_instance_types = ["t3.xlarge", "t3.2xlarge"]
node_desired_size   = 3
node_min_size      = 1
node_max_size      = 5

# ElastiCache Configuration
redis_node_type       = "cache.t3.medium"
redis_num_cache_nodes = 3

# DynamoDB Configuration
dynamodb_billing_mode = "PAY_PER_REQUEST"

# S3 Configuration
s3_versioning = true

# CloudFront Configuration
cloudfront_price_class = "PriceClass_100"

# Cognito Configuration
cognito_password_policy = {
  minimum_length    = 12
  require_lowercase = true
  require_numbers   = true
  require_symbols   = true
  require_uppercase = true
}

# Monitoring Configuration
monitoring_namespace = "monitoring-staging"

# Domain Configuration
domain_name = "staging.tald-unia.com"

# Common Resource Tags
tags = {
  Project     = "TALD-UNIA"
  ManagedBy   = "Terraform"
  Environment = "staging"
  Component   = "Gaming Platform"
}

# AppSync Configuration
appsync_authentication_type = "AMAZON_COGNITO_USER_POOLS"

# ECS Fargate Configuration
ecs_task_cpu    = 1024  # 1 vCPU
ecs_task_memory = 2048  # 2 GB

# Additional Environment-Specific Variables
enable_detailed_monitoring = true
backup_retention_days     = 7
log_retention_days       = 30

# Fleet Network Configuration
fleet_max_size           = 32  # Maximum devices in fleet as per technical spec
fleet_connection_timeout = 300 # 5 minutes in seconds

# Game Asset Configuration
game_asset_max_size      = 1073741824  # 1GB in bytes
asset_cache_ttl_seconds  = 3600        # 1 hour

# Performance Configuration
api_throttling_rate_limit = 1000
api_throttling_burst_limit = 2000

# Security Configuration
enable_waf                = true
enable_shield            = false  # Disabled for staging
ssl_policy              = "TLSv1.2_2021"