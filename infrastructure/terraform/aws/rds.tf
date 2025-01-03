# AWS RDS Configuration for TALD UNIA Platform
# Provider version: hashicorp/aws ~> 4.0

# Local variables for database configuration
locals {
  db_name                 = "tald_unia"
  db_port                 = "5432"
  db_engine              = "postgres"
  db_engine_version      = "14.7"
  db_family              = "postgres14"
  backup_retention_period = "7"
}

# RDS subnet group for database placement
resource "aws_db_subnet_group" "main" {
  name        = "${var.environment}-tald-unia-rds"
  subnet_ids  = module.vpc.private_subnets
  
  tags = {
    Environment = var.environment
    Project     = "TALD UNIA"
  }
}

# RDS parameter group for database configuration
resource "aws_db_parameter_group" "main" {
  family = local.db_family
  name   = "${var.environment}-tald-unia-pg"
  
  parameter {
    name  = "max_connections"
    value = "1000"
  }
  
  parameter {
    name  = "shared_buffers"
    value = "{DBInstanceClassMemory/4096}"
  }
  
  parameter {
    name  = "work_mem"
    value = "16384"
  }
  
  parameter {
    name  = "maintenance_work_mem"
    value = "2097152"
  }
  
  parameter {
    name  = "effective_cache_size"
    value = "{DBInstanceClassMemory/2}"
  }
  
  tags = {
    Environment = var.environment
    Project     = "TALD UNIA"
  }
}

# Main RDS instance
resource "aws_db_instance" "main" {
  identifier = "${var.environment}-tald-unia"
  
  # Engine configuration
  engine         = local.db_engine
  engine_version = local.db_engine_version
  
  # Instance configuration
  instance_class        = "db.t3.large"
  allocated_storage     = 100
  max_allocated_storage = 1000
  
  # Database configuration
  db_name  = local.db_name
  username = "tald_admin"
  port     = local.db_port
  
  # High availability configuration
  multi_az = true
  
  # Security configuration
  storage_encrypted = true
  
  # Backup configuration
  backup_retention_period = local.backup_retention_period
  backup_window          = "03:00-04:00"
  maintenance_window     = "Mon:04:00-Mon:05:00"
  
  # Network configuration
  db_subnet_group_name = aws_db_subnet_group.main.name
  parameter_group_name = aws_db_parameter_group.main.name
  
  # Snapshot configuration
  skip_final_snapshot       = false
  final_snapshot_identifier = "${var.environment}-tald-unia-final"
  
  # Monitoring configuration
  performance_insights_enabled = true
  monitoring_interval         = 60
  enabled_cloudwatch_logs_exports = [
    "postgresql",
    "upgrade"
  ]
  
  # Protection configuration
  deletion_protection = true
  
  # Auto minor version upgrade
  auto_minor_version_upgrade = true
  
  tags = {
    Environment = var.environment
    Project     = "TALD UNIA"
  }
}

# RDS enhanced monitoring role
resource "aws_iam_role" "rds_enhanced_monitoring" {
  name = "${var.environment}-tald-unia-rds-monitoring"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "monitoring.rds.amazonaws.com"
        }
      }
    ]
  })

  tags = {
    Environment = var.environment
    Project     = "TALD UNIA"
  }
}

resource "aws_iam_role_policy_attachment" "rds_enhanced_monitoring" {
  role       = aws_iam_role.rds_enhanced_monitoring.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonRDSEnhancedMonitoringRole"
}

# Outputs for other modules to consume
output "rds_endpoint" {
  description = "RDS instance endpoint"
  value       = aws_db_instance.main.endpoint
}

output "rds_arn" {
  description = "RDS instance ARN"
  value       = aws_db_instance.main.arn
}

output "rds_id" {
  description = "RDS instance ID"
  value       = aws_db_instance.main.id
}