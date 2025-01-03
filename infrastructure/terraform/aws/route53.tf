# Provider configuration with required version constraints
terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 4.0"
    }
  }
}

# Variable definition for domain name with validation
variable "domain_name" {
  type        = string
  description = "Primary domain name for TALD UNIA platform"
  validation {
    condition     = can(regex("^[a-z0-9][a-z0-9-]{1,61}[a-z0-9]\\.[a-z]{2,}$", var.domain_name))
    error_message = "Domain name must be a valid DNS name"
  }
}

# Primary Route53 hosted zone for the domain
resource "aws_route53_zone" "main" {
  name    = var.domain_name
  comment = "Managed by Terraform - TALD UNIA ${var.environment} environment"

  tags = {
    Environment = var.environment
    Project     = "TALD UNIA"
    ManagedBy   = "Terraform"
  }
}

# A record for CloudFront distribution
resource "aws_route53_record" "cloudfront_a" {
  zone_id = aws_route53_zone.main.zone_id
  name    = var.domain_name
  type    = "A"

  alias {
    name                   = aws_cloudfront_distribution.main.domain_name
    zone_id               = aws_cloudfront_distribution.main.hosted_zone_id
    evaluate_target_health = false
  }
}

# AAAA record for CloudFront distribution (IPv6 support)
resource "aws_route53_record" "cloudfront_aaaa" {
  zone_id = aws_route53_zone.main.zone_id
  name    = var.domain_name
  type    = "AAAA"

  alias {
    name                   = aws_cloudfront_distribution.main.domain_name
    zone_id               = aws_cloudfront_distribution.main.hosted_zone_id
    evaluate_target_health = false
  }
}

# Output the hosted zone ID for reference by other modules
output "route53_zone_id" {
  description = "ID of the Route53 hosted zone"
  value       = aws_route53_zone.main.zone_id
}

# Output the name servers for DNS delegation
output "route53_name_servers" {
  description = "List of name servers for the Route53 zone"
  value       = aws_route53_zone.main.name_servers
}