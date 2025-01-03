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

# Origin Access Identity for CloudFront to access S3
resource "aws_cloudfront_origin_access_identity" "game_content_oai" {
  comment = "Origin Access Identity for TALD UNIA game content bucket"
}

# CloudFront distribution for game content delivery
resource "aws_cloudfront_distribution" "game_content_distribution" {
  enabled             = true
  is_ipv6_enabled    = true
  price_class        = var.cloudfront_price_class
  comment            = "TALD UNIA game content distribution - ${var.environment}"
  default_root_object = "index.html"

  origin {
    domain_name = aws_s3_bucket.game_content.bucket_regional_domain_name
    origin_id   = "S3-game-content"

    s3_origin_config {
      origin_access_identity = aws_cloudfront_origin_access_identity.game_content_oai.cloudfront_access_identity_path
    }
  }

  default_cache_behavior {
    allowed_methods  = ["GET", "HEAD", "OPTIONS"]
    cached_methods   = ["GET", "HEAD"]
    target_origin_id = "S3-game-content"

    forwarded_values {
      query_string = false
      cookies {
        forward = "none"
      }
    }

    viewer_protocol_policy = "redirect-to-https"
    min_ttl                = 0
    default_ttl            = 3600  # 1 hour
    max_ttl                = 86400 # 24 hours
    compress               = true

    # Enable caching for common file types
    cache_policy_id = aws_cloudfront_cache_policy.game_content.id
  }

  # Custom error response for SPA routing
  custom_error_response {
    error_code         = 404
    response_code      = 200
    response_page_path = "/index.html"
  }

  # Geo restriction - none as per requirements
  restrictions {
    geo_restriction {
      restriction_type = "none"
    }
  }

  # SSL Certificate configuration
  viewer_certificate {
    cloudfront_default_certificate = true
  }

  tags = local.common_tags
}

# Cache policy for game content
resource "aws_cloudfront_cache_policy" "game_content" {
  name        = "tald-unia-game-content-cache-policy-${var.environment}"
  comment     = "Cache policy for TALD UNIA game content"
  default_ttl = 3600
  max_ttl     = 86400
  min_ttl     = 0

  parameters_in_cache_key_and_forwarded_to_origin {
    cookies_config {
      cookie_behavior = "none"
    }
    headers_config {
      header_behavior = "none"
    }
    query_strings_config {
      query_string_behavior = "none"
    }
    enable_accept_encoding_brotli = true
    enable_accept_encoding_gzip   = true
  }
}

# S3 bucket policy for CloudFront access
resource "aws_s3_bucket_policy" "cloudfront_access" {
  bucket = aws_s3_bucket.game_content.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid       = "AllowCloudFrontOAIAccess"
        Effect    = "Allow"
        Principal = {
          AWS = aws_cloudfront_origin_access_identity.game_content_oai.iam_arn
        }
        Action   = "s3:GetObject"
        Resource = "${aws_s3_bucket.game_content.arn}/*"
      }
    ]
  })
}

# Output values for other modules
output "cloudfront_distribution_id" {
  description = "ID of the CloudFront distribution"
  value       = aws_cloudfront_distribution.game_content_distribution.id
}

output "cloudfront_domain_name" {
  description = "Domain name of the CloudFront distribution"
  value       = aws_cloudfront_distribution.game_content_distribution.domain_name
}

output "cloudfront_oai_iam_arn" {
  description = "IAM ARN of the CloudFront Origin Access Identity"
  value       = aws_cloudfront_origin_access_identity.game_content_oai.iam_arn
}