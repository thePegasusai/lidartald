# Provider configuration with required version constraints
terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 4.0"
    }
    tls = {
      source  = "hashicorp/tls"
      version = "~> 4.0"
    }
  }
}

# AWS provider configuration with default tags
provider "aws" {
  region = var.aws_region
  
  default_tags {
    tags = {
      Environment = var.environment
      Project     = "TALD UNIA"
      ManagedBy   = "Terraform"
    }
  }
}

# ACM certificate for HTTPS endpoints
resource "aws_acm_certificate" "main" {
  domain_name               = var.domain_name
  validation_method         = "DNS"
  subject_alternative_names = ["*.${var.domain_name}"]

  tags = {
    Name         = "tald-unia-${var.environment}-cert"
    Environment  = var.environment
    ExpiryAlert  = "30days"
  }

  lifecycle {
    create_before_destroy = true
  }
}

# KMS key for data encryption
resource "aws_kms_key" "main" {
  description              = "TALD UNIA encryption key for ${var.environment}"
  deletion_window_in_days  = var.key_deletion_window
  enable_key_rotation      = var.enable_key_rotation
  policy                   = data.aws_iam_policy_document.kms_key_policy.json

  tags = {
    Name              = "tald-unia-${var.environment}-key"
    Environment       = var.environment
    RotationSchedule  = "365days"
  }
}

# KMS key alias for easier reference
resource "aws_kms_alias" "main" {
  name          = "alias/tald-unia-${var.environment}"
  target_key_id = aws_kms_key.main.key_id
}

# KMS key policy document
data "aws_iam_policy_document" "kms_key_policy" {
  statement {
    sid    = "Enable IAM User Permissions"
    effect = "Allow"
    principals {
      type        = "AWS"
      identifiers = ["arn:aws:iam::${data.aws_caller_identity.current.account_id}:root"]
    }
    actions   = ["kms:*"]
    resources = ["*"]
  }

  statement {
    sid    = "Allow CloudWatch Logs"
    effect = "Allow"
    principals {
      type        = "Service"
      identifiers = ["logs.${var.aws_region}.amazonaws.com"]
    }
    actions = [
      "kms:Encrypt*",
      "kms:Decrypt*",
      "kms:ReEncrypt*",
      "kms:GenerateDataKey*",
      "kms:Describe*"
    ]
    resources = ["*"]
  }
}

# Get current AWS account ID
data "aws_caller_identity" "current" {}

# Security group for TALD UNIA services
resource "aws_security_group" "main" {
  name        = "tald-unia-${var.environment}-sg"
  description = "Security group for TALD UNIA ${var.environment} services"
  vpc_id      = var.vpc_id

  # HTTPS ingress rule
  ingress {
    description      = "HTTPS"
    from_port        = 443
    to_port          = 443
    protocol         = "tcp"
    cidr_blocks      = var.allowed_cidr_blocks
    ipv6_cidr_blocks = ["::/0"]
  }

  # WebSocket ingress rule
  ingress {
    description      = "WebSocket"
    from_port        = 8443
    to_port          = 8443
    protocol         = "tcp"
    cidr_blocks      = var.allowed_cidr_blocks
    ipv6_cidr_blocks = ["::/0"]
  }

  # Allow all outbound traffic
  egress {
    description      = "All outbound traffic"
    from_port        = 0
    to_port          = 0
    protocol         = "-1"
    cidr_blocks      = ["0.0.0.0/0"]
    ipv6_cidr_blocks = ["::/0"]
  }

  tags = {
    Name         = "tald-unia-${var.environment}-sg"
    Environment  = var.environment
    LastReviewed = timestamp()
    ReviewCycle  = "90days"
  }

  lifecycle {
    create_before_destroy = true
  }
}

# CloudWatch log group for security monitoring
resource "aws_cloudwatch_log_group" "security_logs" {
  name              = "/tald-unia/${var.environment}/security"
  retention_in_days = 90
  kms_key_id        = aws_kms_key.main.arn

  tags = {
    Name        = "tald-unia-${var.environment}-security-logs"
    Environment = var.environment
  }
}

# CloudWatch metric alarm for certificate expiry
resource "aws_cloudwatch_metric_alarm" "certificate_expiry" {
  alarm_name          = "tald-unia-${var.environment}-cert-expiry"
  comparison_operator = "LessThanThreshold"
  evaluation_periods  = "1"
  metric_name         = "DaysToExpiry"
  namespace           = "AWS/ACM"
  period              = "86400"
  statistic           = "Minimum"
  threshold           = "30"
  alarm_description   = "Certificate expiry alert for TALD UNIA ${var.environment}"
  alarm_actions       = []  # Add SNS topic ARN for notifications

  dimensions = {
    CertificateArn = aws_acm_certificate.main.arn
  }

  tags = {
    Name        = "tald-unia-${var.environment}-cert-expiry-alarm"
    Environment = var.environment
  }
}