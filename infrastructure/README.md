# TALD UNIA Infrastructure Documentation

<!-- markdownlint-disable MD013 -->

## Overview

TALD UNIA's infrastructure is built on a hybrid architecture combining local device capabilities, edge network 
functionality, and cloud services. This document provides comprehensive details about the infrastructure setup, 
deployment procedures, and maintenance guidelines.

### Project Structure

```
infrastructure/
├── terraform/
│   └── aws/                 # AWS infrastructure as code
├── helm/
│   └── tald-unia/          # Kubernetes deployment charts
├── scripts/                 # Automation scripts
└── monitoring/             # Monitoring configurations
```

### Technology Stack

- **Cloud Platform**: AWS (2023-Q4)
- **Container Orchestration**: Kubernetes 1.27+
- **Infrastructure as Code**: Terraform 1.0.0+
- **Package Management**: Helm 3.12+
- **Monitoring**: Prometheus 2.44.0, Grafana 10.0.3
- **Logging**: ELK Stack 8.9.0

### Environment Configuration

- Development: `dev.env`
- Staging: `staging.env`
- Production: `prod.env`

## Prerequisites

Before deploying the infrastructure, ensure the following tools are installed:

- AWS CLI v2.13+
- Terraform v1.0.0+
- kubectl v1.27+
- Helm v3.12+
- Docker v24.0+

Required credentials:
- AWS access keys with appropriate IAM permissions
- Kubernetes cluster access credentials
- Container registry credentials

## Infrastructure Components

### AWS Resources

#### ECS Fargate
- **Purpose**: Container orchestration
- **Configuration**: 
  - Task definitions for API, WebSocket, and Analytics services
  - Auto-scaling based on CPU/Memory metrics
  - Spot instance configuration for cost optimization

#### DynamoDB
- **Purpose**: NoSQL database for user profiles and game states
- **Configuration**:
  - On-demand capacity mode
  - Point-in-time recovery enabled
  - Global tables for multi-region deployment

#### S3
- **Purpose**: Game content and asset storage
- **Configuration**:
  - Versioning enabled
  - CloudFront distribution
  - Lifecycle policies for cost optimization

#### ElastiCache
- **Purpose**: Session management and real-time data
- **Configuration**:
  - Redis cluster mode
  - Multi-AZ deployment
  - Encryption at rest

#### AppSync
- **Purpose**: GraphQL API and real-time updates
- **Configuration**:
  - WebSocket subscriptions
  - Schema directives for authorization
  - Resolver mapping templates

#### Cognito
- **Purpose**: User authentication and authorization
- **Configuration**:
  - OAuth 2.0 flows
  - MFA enforcement
  - Custom authentication triggers

### Kubernetes Resources

#### Core Services
- API Service (3 replicas)
- WebSocket Service (5 replicas)
- Analytics Service (2 replicas)

#### Monitoring Stack
- Prometheus server
- Grafana dashboards
- Alert manager

#### Security Components
- Network policies
- RBAC configurations
- Service accounts

## Deployment Guide

### Terraform Deployment

1. Initialize Terraform:
```bash
terraform init -backend-config=environments/prod/backend.hcl
```

2. Plan deployment:
```bash
terraform plan -var-file=environments/prod/terraform.tfvars
```

3. Apply configuration:
```bash
terraform apply -auto-approve
```

### Kubernetes Setup

1. Initialize cluster:
```bash
./scripts/init-cluster.sh --env production
```

2. Deploy Helm charts:
```bash
helm upgrade --install tald-unia ./helm/tald-unia -f values-prod.yaml
```

3. Verify deployment:
```bash
kubectl get pods -n tald-unia
```

## Monitoring

### Prometheus Configuration
- Metrics collection interval: 15s
- Retention period: 15 days
- Storage: 100GB PVC

### Grafana Dashboards
- Infrastructure Overview
- Service Performance
- Resource Utilization
- Security Metrics

### ELK Stack
- Filebeat for log collection
- Logstash for processing
- Elasticsearch for storage
- Kibana for visualization

## Security

### TLS Configuration
- DTLS 1.3 for mesh network
- Certificate rotation every 90 days
- Let's Encrypt integration

### Network Policies
- Zero-trust architecture
- Pod-to-pod communication rules
- External access controls

### Access Control
- RBAC implementation
- Service account management
- Secret rotation

## Maintenance

### Backup Procedures
- Daily automated backups
- Cross-region replication
- 30-day retention policy

### Update Strategy
- Rolling updates for zero-downtime
- Canary deployments
- Automated rollback capability

### Scaling Guidelines
- Horizontal pod autoscaling
- Vertical pod autoscaling
- Cluster autoscaling

## Troubleshooting

### Common Issues

#### Deployment Failures
- Check pipeline logs
- Verify resource quotas
- Validate configurations

#### Network Issues
- Check DNS resolution
- Verify security groups
- Validate network policies

#### Service Failures
- Check container logs
- Verify resource limits
- Validate dependencies

### Support Contacts

- Infrastructure Team: infra@tald-unia.com
- Security Team: security@tald-unia.com
- DevOps Team: devops@tald-unia.com

---

For detailed implementation references, see:
- [Terraform Configuration](./terraform/aws/main.tf)
- [Helm Charts](./helm/tald-unia/Chart.yaml)
- [Cluster Scripts](./scripts/init-cluster.sh)