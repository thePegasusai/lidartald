# AWS Provider configuration with version constraint
terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 4.0"
    }
  }
}

# Local variables for common tags
locals {
  common_tags = {
    Project     = "TALD-UNIA"
    Environment = var.environment
    ManagedBy   = "Terraform"
  }
}

# Game content bucket for storing game assets and content
resource "aws_s3_bucket" "game_content" {
  bucket = "tald-unia-game-content-${var.environment}"
  tags   = local.common_tags
}

resource "aws_s3_bucket_versioning" "game_content" {
  bucket = aws_s3_bucket.game_content.id
  versioning_configuration {
    status = var.s3_versioning ? "Enabled" : "Disabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "game_content" {
  bucket = aws_s3_bucket.game_content.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_lifecycle_configuration" "game_content" {
  bucket = aws_s3_bucket.game_content.id
  rule {
    id     = "transition-to-ia"
    status = "Enabled"
    transition {
      days          = 30
      storage_class = "STANDARD_IA"
    }
  }
}

# User data bucket for storing profiles and achievements
resource "aws_s3_bucket" "user_data" {
  bucket = "tald-unia-user-data-${var.environment}"
  tags   = local.common_tags
}

resource "aws_s3_bucket_versioning" "user_data" {
  bucket = aws_s3_bucket.user_data.id
  versioning_configuration {
    status = var.s3_versioning ? "Enabled" : "Disabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "user_data" {
  bucket = aws_s3_bucket.user_data.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_lifecycle_configuration" "user_data" {
  bucket = aws_s3_bucket.user_data.id
  rule {
    id     = "expire-old-data"
    status = "Enabled"
    expiration {
      days = 90
    }
  }
}

# Environment data bucket for LiDAR scans and environment data
resource "aws_s3_bucket" "environment_data" {
  bucket = "tald-unia-environment-data-${var.environment}"
  tags   = local.common_tags
}

resource "aws_s3_bucket_versioning" "environment_data" {
  bucket = aws_s3_bucket.environment_data.id
  versioning_configuration {
    status = var.s3_versioning ? "Enabled" : "Disabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "environment_data" {
  bucket = aws_s3_bucket.environment_data.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_lifecycle_configuration" "environment_data" {
  bucket = aws_s3_bucket.environment_data.id
  rule {
    id     = "transition-and-expire"
    status = "Enabled"
    transition {
      days          = 7
      storage_class = "STANDARD_IA"
    }
    expiration {
      days = 30
    }
  }
}

# CloudFront access policy for game content bucket
resource "aws_s3_bucket_policy" "game_content" {
  bucket = aws_s3_bucket.game_content.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid       = "AllowCloudFrontAccess"
        Effect    = "Allow"
        Principal = {
          Service = "cloudfront.amazonaws.com"
        }
        Action   = "s3:GetObject"
        Resource = "${aws_s3_bucket.game_content.arn}/*"
        Condition = {
          StringEquals = {
            "AWS:SourceArn" = aws_cloudfront_distribution.main.arn
          }
        }
      }
    ]
  })
}

# Block public access for all buckets
resource "aws_s3_bucket_public_access_block" "game_content" {
  bucket                  = aws_s3_bucket.game_content.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_public_access_block" "user_data" {
  bucket                  = aws_s3_bucket.user_data.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_public_access_block" "environment_data" {
  bucket                  = aws_s3_bucket.environment_data.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# Output values for other modules
output "game_content_bucket_id" {
  description = "ID of the game content S3 bucket"
  value       = aws_s3_bucket.game_content.id
}

output "user_data_bucket_id" {
  description = "ID of the user data S3 bucket"
  value       = aws_s3_bucket.user_data.id
}

output "environment_data_bucket_id" {
  description = "ID of the environment data S3 bucket"
  value       = aws_s3_bucket.environment_data.id
}