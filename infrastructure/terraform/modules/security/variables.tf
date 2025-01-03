# Environment variable for deployment stage
variable "environment" {
  type        = string
  description = "Deployment environment (staging/production)"
  
  validation {
    condition     = contains(["staging", "production"], var.environment)
    error_message = "Environment must be either staging or production"
  }
}

# AWS region for security infrastructure deployment
variable "aws_region" {
  type        = string
  description = "AWS region for deploying security infrastructure"
  default     = "us-west-2"
}

# Domain name for TLS certificate generation
variable "domain_name" {
  type        = string
  description = "Domain name for TLS certificate generation"
  
  validation {
    condition     = can(regex("^[a-z0-9.-]+$", var.domain_name))
    error_message = "Domain name must be a valid DNS name"
  }
}

# VPC ID for security group creation
variable "vpc_id" {
  type        = string
  description = "VPC ID for security group creation"
  
  validation {
    condition     = can(regex("^vpc-", var.vpc_id))
    error_message = "VPC ID must start with vpc-"
  }
}

# KMS key deletion window configuration
variable "key_deletion_window" {
  type        = number
  description = "KMS key deletion window in days"
  default     = 30
  
  validation {
    condition     = var.key_deletion_window >= 7 && var.key_deletion_window <= 30
    error_message = "Key deletion window must be between 7 and 30 days"
  }
}

# KMS key rotation configuration
variable "enable_key_rotation" {
  type        = bool
  description = "Enable automatic KMS key rotation"
  default     = true
}

# CIDR blocks for security group rules
variable "allowed_cidr_blocks" {
  type        = list(string)
  description = "List of CIDR blocks allowed to access HTTPS and WebSocket endpoints"
  default     = ["0.0.0.0/0"]
  
  validation {
    condition     = alltrue([for cidr in var.allowed_cidr_blocks : can(cidrhost(cidr, 0))])
    error_message = "All elements must be valid CIDR blocks"
  }
}