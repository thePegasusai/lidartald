# TALD UNIA Backend Services

Enterprise-grade backend services for the TALD UNIA LiDAR-enabled gaming platform, providing real-time point cloud processing, mesh networking, game state management, and social features.

## System Requirements

### Hardware Requirements
- CPU: 8+ cores, AVX2/FMA support
- GPU: NVIDIA RTX 20xx series or better (Compute Capability 7.5+)
- RAM: 16GB minimum, 32GB recommended
- Storage: NVMe SSD with 500GB+ available space
- Network: Gigabit Ethernet

### Software Requirements
- Ubuntu 22.04 LTS or newer
- CUDA Toolkit 12.0
- Vulkan SDK 1.3
- Docker 24.0+
- Docker Compose 2.20+
- NVIDIA Container Toolkit

## Quick Start

1. Clone the repository and set up environment:
```bash
git clone <repository-url>
cd tald-unia/backend
cp .env.example .env
# Configure environment variables in .env
```

2. Build and start services:
```bash
docker compose build
docker compose up -d
```

3. Verify deployment:
```bash
docker compose ps
curl http://localhost:3000/health
```

## Core Components

### LiDAR Core (C++)
High-performance point cloud processing pipeline
- 30Hz scan rate
- 0.01cm resolution
- CUDA-accelerated processing
- Real-time feature detection

### Fleet Manager (Rust)
Distributed mesh networking coordinator
- WebRTC P2P connections
- 32 device mesh support
- <50ms network latency
- Automatic failover

### Game Engine (C++/Vulkan)
Reality-based game runtime
- Vulkan graphics pipeline
- Physics simulation
- Environment mapping
- 60 FPS performance

### Social Engine (Node.js)
User interaction and matchmaking service
- Real-time user discovery
- Fleet formation
- Profile management
- Session coordination

## Development Setup

### Prerequisites Installation
```bash
# Install system dependencies
sudo apt update && sudo apt install -y \
    build-essential \
    cmake \
    ninja-build \
    pkg-config \
    libvulkan-dev \
    python3-pip

# Install CUDA Toolkit
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt update
sudo apt install -y cuda-toolkit-12-0

# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env

# Install Node.js
curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
sudo apt install -y nodejs
```

### Environment Configuration
1. Configure environment variables:
```bash
# Security credentials
JWT_SECRET=<generate-secure-random-string>
ENCRYPTION_KEY=<generate-32-byte-key>

# Database configuration
POSTGRES_USER=tald
POSTGRES_PASSWORD=<secure-password>
POSTGRES_DB=tald_unia
```

2. Configure hardware settings:
```bash
# NVIDIA GPU settings
nvidia-smi -pm 1
nvidia-smi --auto-boost-default=0
nvidia-smi -ac 5001,1590

# System limits
sudo sysctl -w vm.max_map_count=262144
sudo sysctl -w net.core.somaxconn=65535
```

## Build Process

### Development Build
```bash
# Build all services
docker compose -f docker-compose.yml -f docker-compose.dev.yml build

# Start development environment
docker compose -f docker-compose.yml -f docker-compose.dev.yml up -d
```

### Production Build
```bash
# Build optimized images
docker compose build --no-cache

# Deploy services
docker compose up -d --force-recreate
```

## Testing

### Unit Tests
```bash
# C++ components
cd build && ctest --output-on-failure

# Rust components
cargo test --all-features

# Node.js components
npm test
```

### Integration Tests
```bash
# Start test environment
docker compose -f docker-compose.test.yml up -d

# Run integration test suite
npm run test:integration
```

## Monitoring

### Health Checks
- LiDAR Core: http://localhost:9090/health
- Fleet Manager: http://localhost:8080/health
- Game Engine: http://localhost:9091/health
- Social Engine: http://localhost:3000/health

### Metrics
- Prometheus endpoints exposed on port 9090
- Grafana dashboards available at port 3000
- Custom metrics for:
  - Point cloud processing performance
  - Mesh network latency
  - Game engine frame times
  - User interaction patterns

## Security

### Authentication
- OAuth 2.0 + JWT for API authentication
- WebRTC secure signaling
- TLS 1.3 for all connections
- Hardware-backed key storage

### Encryption
- AES-256-GCM for data at rest
- TLS 1.3 for data in transit
- End-to-end encryption for P2P communication

### Access Control
- RBAC for API endpoints
- Fleet-level access controls
- Resource isolation per container

## Documentation

### API Documentation
- OpenAPI/Swagger UI: http://localhost:8080/docs
- WebSocket Protocol: http://localhost:8080/docs/ws
- Fleet Protocol: http://localhost:8080/docs/fleet

### Architecture Documentation
- Component diagrams in /docs/architecture
- Sequence diagrams in /docs/flows
- Performance benchmarks in /docs/performance

## Support

### Troubleshooting
- Check service logs: `docker compose logs -f [service]`
- Verify GPU access: `nvidia-smi`
- Monitor resources: `docker stats`

### Common Issues
1. GPU not detected:
   - Verify NVIDIA drivers
   - Check NVIDIA Container Toolkit
   
2. Performance issues:
   - Monitor GPU utilization
   - Check network latency
   - Verify CPU frequency scaling

## License
Proprietary - All rights reserved