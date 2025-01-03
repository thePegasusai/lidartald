# AWS Provider configuration
terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 4.0"
    }
  }
}

# Local variables for Redis configuration
locals {
  redis_family            = "redis7.0"
  redis_port             = "6379"
  redis_maintenance_window = "sun:05:00-sun:06:00"
  redis_snapshot_window   = "03:00-04:00"
  private_subnet_cidrs    = ["10.0.0.0/16"]
}

# ElastiCache subnet group for Redis cluster
resource "aws_elasticache_subnet_group" "redis" {
  name        = "tald-unia-${var.environment}-redis-subnet"
  subnet_ids  = data.aws_subnet.database.*.id
  description = "Subnet group for TALD UNIA Redis cluster deployment"

  tags = {
    Name        = "tald-unia-${var.environment}-redis-subnet"
    Environment = var.environment
    Project     = "TALD-UNIA"
    ManagedBy   = "Terraform"
  }
}

# ElastiCache parameter group for Redis optimization
resource "aws_elasticache_parameter_group" "redis" {
  family      = local.redis_family
  name        = "tald-unia-${var.environment}-redis-params"
  description = "Redis parameter group for TALD UNIA optimized for fleet coordination"

  parameter {
    name  = "maxmemory-policy"
    value = "allkeys-lru"
  }

  parameter {
    name  = "timeout"
    value = "300"
  }

  parameter {
    name  = "tcp-keepalive"
    value = "300"
  }

  parameter {
    name  = "maxclients"
    value = "65000"
  }

  tags = {
    Name        = "tald-unia-${var.environment}-redis-params"
    Environment = var.environment
  }
}

# ElastiCache replication group for Redis cluster
resource "aws_elasticache_replication_group" "redis" {
  replication_group_id          = "tald-unia-${var.environment}-redis"
  description                   = "Redis cluster for TALD UNIA fleet coordination and session management"
  node_type                     = var.redis_node_type
  num_cache_clusters           = var.redis_num_cache_nodes
  port                         = local.redis_port
  parameter_group_name         = aws_elasticache_parameter_group.redis.name
  subnet_group_name            = aws_elasticache_subnet_group.redis.name
  automatic_failover_enabled   = true
  multi_az_enabled            = true
  engine                      = "redis"
  engine_version              = "7.0"
  maintenance_window          = local.redis_maintenance_window
  snapshot_window             = local.redis_snapshot_window
  snapshot_retention_limit    = 7
  at_rest_encryption_enabled  = true
  transit_encryption_enabled  = true
  auth_token                  = var.redis_auth_token
  auto_minor_version_upgrade  = true
  notification_topic_arn      = aws_sns_topic.redis_notifications.arn

  tags = {
    Name        = "tald-unia-${var.environment}-redis"
    Environment = var.environment
    Project     = "TALD-UNIA"
    ManagedBy   = "Terraform"
  }
}

# Security group for Redis cluster
resource "aws_security_group" "redis" {
  name        = "tald-unia-${var.environment}-redis-sg"
  description = "Security group for Redis cluster access control"
  vpc_id      = data.aws_vpc.main.id

  ingress {
    from_port   = local.redis_port
    to_port     = local.redis_port
    protocol    = "tcp"
    cidr_blocks = local.private_subnet_cidrs
    description = "Allow Redis access from private subnets"
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
    description = "Allow all outbound traffic"
  }

  tags = {
    Name        = "tald-unia-${var.environment}-redis-sg"
    Environment = var.environment
  }
}

# Output values for Redis endpoint and port
output "redis_endpoint" {
  value       = aws_elasticache_replication_group.redis.primary_endpoint_address
  description = "Primary endpoint address for Redis cluster"
}

output "redis_port" {
  value       = aws_elasticache_replication_group.redis.port
  description = "Port number for Redis cluster"
}