# VPC Outputs
output "vpc_id" {
  description = "ID of the created VPC"
  value       = module.vpc.vpc_id
  sensitive   = false
}

output "private_subnets" {
  description = "List of private subnet IDs"
  value       = module.vpc.private_subnets
  sensitive   = false
}

output "public_subnets" {
  description = "List of public subnet IDs"
  value       = module.vpc.public_subnets
  sensitive   = false
}

# EKS Cluster Outputs
output "eks_cluster_endpoint" {
  description = "Endpoint for EKS cluster"
  value       = module.eks.cluster_endpoint
  sensitive   = false
}

output "eks_cluster_name" {
  description = "Name of the EKS cluster"
  value       = module.eks.cluster_name
  sensitive   = false
}

output "eks_cluster_ca_certificate" {
  description = "Certificate authority data for EKS cluster"
  value       = module.eks.cluster_ca_certificate
  sensitive   = true
}

# ElastiCache Redis Outputs
output "redis_endpoint" {
  description = "ElastiCache Redis cluster endpoint"
  value       = module.elasticache.endpoint
  sensitive   = false
}

output "redis_port" {
  description = "ElastiCache Redis port number"
  value       = module.elasticache.port
  sensitive   = false
}

# DynamoDB Outputs
output "dynamodb_table_names" {
  description = "Map of DynamoDB table names by purpose"
  value       = module.dynamodb.table_names
  sensitive   = false
}

output "dynamodb_table_arns" {
  description = "Map of DynamoDB table ARNs"
  value       = module.dynamodb.table_arns
  sensitive   = true
}

# S3 and CloudFront Outputs
output "s3_bucket_name" {
  description = "Name of the main S3 storage bucket"
  value       = module.s3.bucket_name
  sensitive   = false
}

output "s3_bucket_arn" {
  description = "ARN of the main S3 storage bucket"
  value       = module.s3.bucket_arn
  sensitive   = true
}

output "cloudfront_domain" {
  description = "CloudFront distribution domain name"
  value       = module.cloudfront.domain_name
  sensitive   = false
}

# Cognito Outputs
output "cognito_user_pool_id" {
  description = "ID of the Cognito user pool"
  value       = module.cognito.user_pool_id
  sensitive   = true
}

output "cognito_user_pool_client_id" {
  description = "ID of the Cognito user pool client"
  value       = module.cognito.user_pool_client_id
  sensitive   = true
}