# Environment Configuration
variable "environment" {
  type        = string
  description = "Deployment environment (staging/production)"
  validation {
    condition     = contains(["staging", "production"], var.environment)
    error_message = "Environment must be either staging or production"
  }
}

# Region Configuration
variable "aws_region" {
  type        = string
  description = "AWS region for resource deployment"
  default     = "us-west-2"
}

# Networking Configuration
variable "vpc_cidr" {
  type        = string
  description = "CIDR block for VPC"
  default     = "10.0.0.0/16"
}

# EKS Cluster Configuration
variable "cluster_version" {
  type        = string
  description = "Kubernetes version for EKS cluster"
  default     = "1.27"
}

variable "node_instance_types" {
  type        = list(string)
  description = "Instance types for EKS node groups"
  default     = ["t3.xlarge", "t3.2xlarge"]
}

variable "node_desired_size" {
  type        = number
  description = "Desired number of nodes in EKS node groups"
  default     = 3
}

variable "node_min_size" {
  type        = number
  description = "Minimum number of nodes in EKS node groups"
  default     = 1
}

variable "node_max_size" {
  type        = number
  description = "Maximum number of nodes in EKS node groups"
  default     = 5
}

# ElastiCache Configuration
variable "redis_node_type" {
  type        = string
  description = "Instance type for ElastiCache Redis nodes"
  default     = "cache.t3.medium"
}

variable "redis_num_cache_nodes" {
  type        = number
  description = "Number of cache nodes in Redis cluster"
  default     = 3
}

# DynamoDB Configuration
variable "dynamodb_billing_mode" {
  type        = string
  description = "DynamoDB billing mode (PROVISIONED or PAY_PER_REQUEST)"
  default     = "PAY_PER_REQUEST"
  validation {
    condition     = contains(["PROVISIONED", "PAY_PER_REQUEST"], var.dynamodb_billing_mode)
    error_message = "DynamoDB billing mode must be either PROVISIONED or PAY_PER_REQUEST"
  }
}

# S3 Configuration
variable "s3_versioning" {
  type        = bool
  description = "Enable versioning for S3 buckets"
  default     = true
}

# CloudFront Configuration
variable "cloudfront_price_class" {
  type        = string
  description = "CloudFront distribution price class"
  default     = "PriceClass_100"
  validation {
    condition     = contains(["PriceClass_100", "PriceClass_200", "PriceClass_All"], var.cloudfront_price_class)
    error_message = "CloudFront price class must be one of: PriceClass_100, PriceClass_200, PriceClass_All"
  }
}

# Cognito Configuration
variable "cognito_password_policy" {
  type = object({
    minimum_length    = number
    require_lowercase = bool
    require_numbers   = bool
    require_symbols   = bool
    require_uppercase = bool
  })
  description = "Cognito user pool password policy settings"
  default = {
    minimum_length    = 12
    require_lowercase = true
    require_numbers   = true
    require_symbols   = true
    require_uppercase = true
  }
  validation {
    condition     = var.cognito_password_policy.minimum_length >= 8
    error_message = "Password minimum length must be at least 8 characters"
  }
}

# Tags Configuration
variable "tags" {
  type        = map(string)
  description = "Common tags to be applied to all resources"
  default = {
    Project     = "TALD-UNIA"
    ManagedBy   = "Terraform"
    Environment = "production"
  }
}

# AppSync Configuration
variable "appsync_authentication_type" {
  type        = string
  description = "Authentication type for AppSync API"
  default     = "AMAZON_COGNITO_USER_POOLS"
  validation {
    condition     = contains(["API_KEY", "AWS_IAM", "AMAZON_COGNITO_USER_POOLS", "OPENID_CONNECT"], var.appsync_authentication_type)
    error_message = "AppSync authentication type must be one of: API_KEY, AWS_IAM, AMAZON_COGNITO_USER_POOLS, OPENID_CONNECT"
  }
}

# ECS Fargate Configuration
variable "ecs_task_cpu" {
  type        = number
  description = "CPU units for ECS Fargate tasks (1 CPU = 1024 units)"
  default     = 1024
}

variable "ecs_task_memory" {
  type        = number
  description = "Memory (in MiB) for ECS Fargate tasks"
  default     = 2048
}