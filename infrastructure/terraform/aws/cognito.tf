# Configure AWS provider with version constraint
terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 4.0"
    }
  }
}

# Data sources for AWS account and region information
data "aws_caller_identity" "current" {}
data "aws_region" "current" {}

# Main Cognito User Pool with enhanced security features
resource "aws_cognito_user_pool" "main" {
  name = "tald-unia-${var.environment}"

  # User attributes and verification
  username_attributes      = ["email"]
  auto_verified_attributes = ["email"]
  
  # Multi-factor authentication configuration
  mfa_configuration = "OPTIONAL"
  
  # Software token MFA configuration
  software_token_mfa_configuration {
    enabled = true
  }

  # Password policy configuration
  password_policy {
    minimum_length    = var.cognito_password_policy.minimum_length
    require_lowercase = var.cognito_password_policy.require_lowercase
    require_numbers   = var.cognito_password_policy.require_numbers
    require_symbols   = var.cognito_password_policy.require_symbols
    require_uppercase = var.cognito_password_policy.require_uppercase
    temporary_password_validity_days = 7
  }

  # User pool add-ons for advanced security
  user_pool_add_ons {
    advanced_security_mode = "ENFORCED"
  }

  # Account recovery settings
  account_recovery_setting {
    recovery_mechanism {
      name     = "verified_email"
      priority = 1
    }
  }

  # Admin user creation settings
  admin_create_user_config {
    allow_admin_create_user_only = true
    invite_message_template {
      email_subject = "Welcome to TALD UNIA Platform"
      email_message = "Your username is {username} and temporary password is {####}"
      sms_message   = "Your username is {username} and temporary password is {####}"
    }
  }

  # Email configuration
  email_configuration {
    email_sending_account = "COGNITO_DEFAULT"
  }

  # Device tracking
  device_configuration {
    challenge_required_on_new_device      = true
    device_only_remembered_on_user_prompt = true
  }

  # User attribute schema
  schema {
    name                     = "role"
    attribute_data_type      = "String"
    developer_only_attribute = false
    mutable                 = true
    required                = false
    string_attribute_constraints {
      min_length = 1
      max_length = 256
    }
  }

  # Data encryption configuration
  encryption_configuration {
    kms_key_id = data.aws_kms_key.main.arn
  }

  # Lambda triggers for custom authentication flows
  lambda_config {
    pre_authentication         = aws_lambda_function.pre_auth.arn
    post_authentication        = aws_lambda_function.post_auth.arn
    pre_token_generation      = aws_lambda_function.pre_token.arn
    user_migration            = aws_lambda_function.user_migration.arn
    define_auth_challenge     = aws_lambda_function.define_auth_challenge.arn
    create_auth_challenge     = aws_lambda_function.create_auth_challenge.arn
    verify_auth_challenge     = aws_lambda_function.verify_auth_challenge.arn
  }

  tags = {
    Environment = var.environment
    Service     = "authentication"
    ManagedBy   = "terraform"
  }
}

# Cognito User Pool Client for web application
resource "aws_cognito_user_pool_client" "web_client" {
  name                = "tald-unia-web-${var.environment}"
  user_pool_id        = aws_cognito_user_pool.main.id
  
  # OAuth configuration
  allowed_oauth_flows  = ["code"]
  allowed_oauth_scopes = ["email", "openid", "profile", "aws.cognito.signin.user.admin"]
  callback_urls        = ["https://app.tald-unia.com/callback"]
  logout_urls         = ["https://app.tald-unia.com/logout"]
  
  # Security features
  generate_secret                      = true
  prevent_user_existence_errors        = "ENABLED"
  enable_token_revocation             = true
  enable_propagate_additional_user_context_data = true
  
  # Token validity configuration
  refresh_token_validity        = 30
  access_token_validity        = 1
  id_token_validity           = 1
  
  token_validity_units {
    refresh_token = "days"
    access_token  = "hours"
    id_token     = "hours"
  }

  # Authentication flows
  explicit_auth_flows = [
    "ALLOW_REFRESH_TOKEN_AUTH",
    "ALLOW_USER_SRP_AUTH"
  ]
}

# Cognito Identity Pool for AWS service access
resource "aws_cognito_identity_pool" "main" {
  identity_pool_name = "tald_unia_${var.environment}_identity_pool"
  
  allow_unauthenticated_identities = false
  allow_classic_flow              = false

  cognito_identity_providers {
    client_id               = aws_cognito_user_pool_client.web_client.id
    provider_name           = aws_cognito_user_pool.main.endpoint
    server_side_token_check = true
  }

  tags = {
    Environment = var.environment
    Service     = "identity"
    ManagedBy   = "terraform"
  }
}

# IAM roles for Cognito Identity Pool
resource "aws_iam_role" "authenticated" {
  name = "tald-unia-${var.environment}-cognito-authenticated"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Principal = {
          Federated = "cognito-identity.amazonaws.com"
        }
        Action = "sts:AssumeRoleWithWebIdentity"
        Condition = {
          StringEquals = {
            "cognito-identity.amazonaws.com:aud" = aws_cognito_identity_pool.main.id
          }
          "ForAnyValue:StringLike" = {
            "cognito-identity.amazonaws.com:amr" = "authenticated"
          }
        }
      }
    ]
  })
}

# Output values for other modules
output "cognito_user_pool_id" {
  value       = aws_cognito_user_pool.main.id
  description = "The ID of the Cognito User Pool"
}

output "cognito_client_id" {
  value       = aws_cognito_user_pool_client.web_client.id
  description = "The ID of the Cognito User Pool Client"
}

output "cognito_client_secret" {
  value       = aws_cognito_user_pool_client.web_client.client_secret
  description = "The secret of the Cognito User Pool Client"
  sensitive   = true
}

output "identity_pool_id" {
  value       = aws_cognito_identity_pool.main.id
  description = "The ID of the Cognito Identity Pool"
}