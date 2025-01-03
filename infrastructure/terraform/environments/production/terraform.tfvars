# Production Environment Configuration
environment = "production"
aws_region  = "us-west-2"

# VPC Configuration
vpc_cidr = "10.0.0.0/16"

# EKS Cluster Configuration
cluster_version     = "1.27"
node_instance_types = ["t3.2xlarge", "t3.4xlarge"]
node_desired_size   = 5
node_min_size      = 3
node_max_size      = 10

# ElastiCache Redis Configuration
redis_node_type       = "cache.r6g.xlarge"
redis_num_cache_nodes = 5

# DynamoDB Configuration
dynamodb_billing_mode = "PROVISIONED"

# S3 Configuration
s3_versioning = true

# CloudFront Configuration
cloudfront_price_class = "PriceClass_All"

# Cognito Configuration
cognito_password_policy = {
  minimum_length    = 16
  require_lowercase = true
  require_numbers   = true
  require_symbols   = true
  require_uppercase = true
}

# Common Resource Tags
tags = {
  Project     = "TALD-UNIA"
  ManagedBy   = "Terraform"
  Environment = "production"
  CostCenter  = "Gaming-Platform"
  Compliance  = "GDPR-CCPA"
}

# AppSync API Configuration
appsync_authentication_type = "AMAZON_COGNITO_USER_POOLS"

# ECS Fargate Configuration
ecs_task_cpu    = 2048  # 2 vCPU
ecs_task_memory = 4096  # 4 GB RAM