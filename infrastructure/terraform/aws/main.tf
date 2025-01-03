# Configure Terraform settings and backend
terraform {
  required_version = ">= 1.0.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 4.0"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.0"
    }
  }

  backend "s3" {
    bucket               = "tald-unia-terraform-state"
    key                  = "aws/terraform.tfstate"
    region              = "us-west-2"
    encrypt             = true
    dynamodb_table      = "terraform-state-lock"
    kms_key_id          = "arn:aws:kms:us-west-2:ACCOUNT_ID:key/KEY_ID"
    workspace_key_prefix = "env"
  }
}

# Configure AWS Provider
provider "aws" {
  region = var.aws_region

  default_tags {
    tags = {
      Project             = "TALD-UNIA"
      Environment         = var.environment
      ManagedBy          = "Terraform"
      SecurityCompliance = "Required"
      DataClassification = "Confidential"
    }
  }
}

# Configure Kubernetes Provider
provider "kubernetes" {
  host                   = module.eks.cluster_endpoint
  cluster_ca_certificate = base64decode(module.eks.cluster_ca_certificate)
  exec {
    api_version = "client.authentication.k8s.io/v1beta1"
    command     = "aws"
    args        = ["eks", "get-token", "--cluster-name", module.eks.cluster_name]
  }
}

# Data source for EKS cluster authentication
data "aws_eks_cluster_auth" "cluster" {
  name = module.eks.cluster_name
}

# ElastiCache Redis Cluster
resource "aws_elasticache_cluster" "fleet_cache" {
  cluster_id           = "tald-unia-${var.environment}-fleet"
  engine              = "redis"
  node_type           = var.redis_node_type
  num_cache_nodes     = var.redis_num_cache_nodes
  parameter_group_family = "redis7"
  port                = 6379
  security_group_ids  = [aws_security_group.redis.id]
  subnet_group_name   = aws_elasticache_subnet_group.redis.name
  
  at_rest_encryption_enabled = true
  transit_encryption_enabled = true
  
  tags = {
    Name = "tald-unia-${var.environment}-fleet-cache"
  }
}

# DynamoDB Tables
resource "aws_dynamodb_table" "fleet_state" {
  name           = "tald-unia-fleet-${var.environment}"
  billing_mode   = var.dynamodb_billing_mode
  hash_key       = "fleet_id"
  range_key      = "timestamp"
  
  attribute {
    name = "fleet_id"
    type = "S"
  }
  
  attribute {
    name = "timestamp"
    type = "N"
  }
  
  point_in_time_recovery {
    enabled = true
  }
  
  server_side_encryption {
    enabled = true
  }
}

# S3 Buckets
resource "aws_s3_bucket" "game_assets" {
  bucket = "tald-unia-assets-${var.environment}"
  
  versioning {
    enabled = var.s3_versioning
  }
  
  server_side_encryption_configuration {
    rule {
      apply_server_side_encryption_by_default {
        sse_algorithm = "AES256"
      }
    }
  }
}

# CloudFront Distribution
resource "aws_cloudfront_distribution" "assets" {
  enabled             = true
  is_ipv6_enabled     = true
  price_class         = var.cloudfront_price_class
  
  origin {
    domain_name = aws_s3_bucket.game_assets.bucket_regional_domain_name
    origin_id   = "S3-${aws_s3_bucket.game_assets.id}"
    
    s3_origin_config {
      origin_access_identity = aws_cloudfront_origin_access_identity.assets.cloudfront_access_identity_path
    }
  }
  
  default_cache_behavior {
    allowed_methods        = ["GET", "HEAD", "OPTIONS"]
    cached_methods         = ["GET", "HEAD"]
    target_origin_id       = "S3-${aws_s3_bucket.game_assets.id}"
    viewer_protocol_policy = "redirect-to-https"
    
    forwarded_values {
      query_string = false
      cookies {
        forward = "none"
      }
    }
  }
  
  restrictions {
    geo_restriction {
      restriction_type = "none"
    }
  }
  
  viewer_certificate {
    cloudfront_default_certificate = true
  }
}

# Cognito User Pool
resource "aws_cognito_user_pool" "main" {
  name = "tald-unia-${var.environment}"
  
  password_policy {
    minimum_length    = var.cognito_password_policy.minimum_length
    require_lowercase = var.cognito_password_policy.require_lowercase
    require_numbers   = var.cognito_password_policy.require_numbers
    require_symbols   = var.cognito_password_policy.require_symbols
    require_uppercase = var.cognito_password_policy.require_uppercase
  }
  
  mfa_configuration = "ON"
  
  account_recovery_setting {
    recovery_mechanism {
      name     = "verified_email"
      priority = 1
    }
  }
}

# AppSync API
resource "aws_appsync_graphql_api" "main" {
  name                = "tald-unia-${var.environment}"
  authentication_type = var.appsync_authentication_type
  
  user_pool_config {
    user_pool_id = aws_cognito_user_pool.main.id
    default_action = "ALLOW"
  }
  
  log_config {
    cloudwatch_logs_role_arn = aws_iam_role.appsync_logs.arn
    field_log_level         = "ERROR"
  }
}

# ECS Fargate Cluster
resource "aws_ecs_cluster" "main" {
  name = "tald-unia-${var.environment}"
  
  setting {
    name  = "containerInsights"
    value = "enabled"
  }
}

# Outputs
output "vpc_id" {
  description = "VPC ID"
  value       = module.vpc.vpc_id
}

output "eks_cluster_endpoint" {
  description = "EKS cluster endpoint"
  value       = module.eks.cluster_endpoint
  sensitive   = true
}