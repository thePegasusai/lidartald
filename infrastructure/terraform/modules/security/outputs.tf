# TLS certificate ARN output
output "certificate_arn" {
  description = "ARN of the TLS certificate for HTTPS endpoints"
  value       = aws_acm_certificate.main.arn
}

# KMS key ARN output - marked as sensitive since it's used for encryption
output "kms_key_arn" {
  description = "ARN of the KMS key for data encryption"
  value       = aws_kms_key.main.arn
  sensitive   = true
}

# Security group ID output
output "security_group_id" {
  description = "ID of the security group for service access control"
  value       = aws_security_group.main.id
}