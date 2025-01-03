# AWS Provider configuration with version constraint
terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 4.0"
    }
  }
}

# EKS Node IAM Role
resource "aws_iam_role" "eks_node_role" {
  name = "tald-unia-eks-node-${var.environment}"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Principal = {
          Service = "ec2.amazonaws.com"
        }
        Action = "sts:AssumeRole"
        Condition = {
          StringEquals = {
            "aws:RequestedRegion" = var.aws_region
          }
        }
      }
    ]
  })

  managed_policy_arns = [
    "arn:aws:iam::aws:policy/AmazonEKSWorkerNodePolicy",
    "arn:aws:iam::aws:policy/AmazonEKS_CNI_Policy",
    "arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly"
  ]

  tags = {
    Environment = var.environment
    Service     = "TALD-UNIA-EKS"
  }
}

# Fleet Service IAM Role
resource "aws_iam_role" "fleet_service_role" {
  name = "tald-unia-fleet-service-${var.environment}"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Principal = {
          Service = "ecs-tasks.amazonaws.com"
        }
        Action = "sts:AssumeRole"
        Condition = {
          StringEquals = {
            "aws:RequestedRegion" = var.aws_region
          }
        }
      }
    ]
  })

  inline_policy {
    name = "fleet-service-permissions"
    policy = jsonencode({
      Version = "2012-10-17"
      Statement = [
        {
          Effect = "Allow"
          Action = [
            "dynamodb:GetItem",
            "dynamodb:PutItem",
            "dynamodb:UpdateItem",
            "dynamodb:Query",
            "dynamodb:BatchWriteItem"
          ]
          Resource = [
            "arn:aws:dynamodb:${var.aws_region}:*:table/tald-unia-fleet-${var.environment}",
            "arn:aws:dynamodb:${var.aws_region}:*:table/tald-unia-fleet-${var.environment}/index/*"
          ]
        },
        {
          Effect = "Allow"
          Action = [
            "s3:GetObject",
            "s3:PutObject"
          ]
          Resource = "arn:aws:s3:::tald-unia-fleet-${var.environment}/*"
        },
        {
          Effect = "Allow"
          Action = [
            "elasticache:DescribeCacheClusters",
            "elasticache:ListTagsForResource"
          ]
          Resource = "*"
        }
      ]
    })
  }

  tags = {
    Environment = var.environment
    Service     = "TALD-UNIA-Fleet"
  }
}

# LiDAR Processing IAM Role
resource "aws_iam_role" "lidar_processing_role" {
  name = "tald-unia-lidar-processing-${var.environment}"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Principal = {
          Service = "lambda.amazonaws.com"
        }
        Action = "sts:AssumeRole"
        Condition = {
          StringEquals = {
            "aws:RequestedRegion" = var.aws_region
          }
        }
      }
    ]
  })

  inline_policy {
    name = "lidar-processing-permissions"
    policy = jsonencode({
      Version = "2012-10-17"
      Statement = [
        {
          Effect = "Allow"
          Action = [
            "s3:GetObject",
            "s3:PutObject"
          ]
          Resource = "arn:aws:s3:::tald-unia-lidar-${var.environment}/*"
        },
        {
          Effect = "Allow"
          Action = [
            "dynamodb:PutItem",
            "dynamodb:UpdateItem"
          ]
          Resource = "arn:aws:dynamodb:${var.aws_region}:*:table/tald-unia-scans-${var.environment}"
        }
      ]
    })
  }

  tags = {
    Environment = var.environment
    Service     = "TALD-UNIA-LiDAR"
  }
}

# Game State IAM Role
resource "aws_iam_role" "game_state_role" {
  name = "tald-unia-game-state-${var.environment}"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Principal = {
          Service = "appsync.amazonaws.com"
        }
        Action = "sts:AssumeRole"
        Condition = {
          StringEquals = {
            "aws:RequestedRegion" = var.aws_region
          }
        }
      }
    ]
  })

  inline_policy {
    name = "game-state-permissions"
    policy = jsonencode({
      Version = "2012-10-17"
      Statement = [
        {
          Effect = "Allow"
          Action = [
            "dynamodb:GetItem",
            "dynamodb:PutItem",
            "dynamodb:UpdateItem",
            "dynamodb:DeleteItem",
            "dynamodb:Query",
            "dynamodb:BatchWriteItem"
          ]
          Resource = [
            "arn:aws:dynamodb:${var.aws_region}:*:table/tald-unia-games-${var.environment}",
            "arn:aws:dynamodb:${var.aws_region}:*:table/tald-unia-games-${var.environment}/index/*"
          ]
        },
        {
          Effect = "Allow"
          Action = [
            "elasticache:DescribeCacheClusters",
            "elasticache:ListTagsForResource"
          ]
          Resource = "*"
        }
      ]
    })
  }

  tags = {
    Environment = var.environment
    Service     = "TALD-UNIA-GameState"
  }
}

# Output the EKS node role ARN
output "eks_node_role_arn" {
  value       = aws_iam_role.eks_node_role.arn
  description = "ARN of the EKS node IAM role"
}

# Output the Fleet service role ARN
output "fleet_service_role_arn" {
  value       = aws_iam_role.fleet_service_role.arn
  description = "ARN of the Fleet service IAM role"
}