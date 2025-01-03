# Stage 1: C++ Builder
FROM gcc:12.2-alpine AS cpp-builder

# Install build dependencies
RUN apk add --no-cache \
    cmake=3.26.4-r0 \
    ninja=1.11.1-r3 \
    vulkan-dev=1.3.250-r0 \
    cuda-dev=12.0.0-r0 \
    eigen-dev=3.4.0-r1 \
    pcl-dev=1.12.1-r0 \
    bullet-dev=3.24-r0 \
    glm-dev=0.9.9.8-r0 \
    spdlog-dev=1.11.0-r0 \
    openmp-dev=15.0.7-r0

# Copy C++ source files
COPY lidar_core/ /build/lidar_core/
COPY game_engine/ /build/game_engine/
COPY CMakeLists.txt /build/

# Build C++ components with optimizations
WORKDIR /build
RUN cmake -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_FLAGS="-O3 -DNDEBUG -march=native -flto -mavx2 -mfma" \
    -DCMAKE_CUDA_FLAGS="--use_fast_math -O3" \
    -DBUILD_TESTING=OFF \
    . && \
    ninja

# Stage 2: Rust Builder
FROM rust:1.70-alpine AS rust-builder

# Install build dependencies
RUN apk add --no-cache \
    musl-dev=1.2.4-r1 \
    openssl-dev=3.1.2-r0 \
    cmake=3.26.4-r0

# Copy Rust source files and manifest
COPY fleet_manager/ /build/fleet_manager/
COPY Cargo.toml /build/

# Build Rust components with optimizations
WORKDIR /build
RUN cargo build --release --features production

# Stage 3: Node.js Builder
FROM node:18-alpine AS node-builder

# Install build dependencies
RUN apk add --no-cache \
    python3=3.11.5-r0 \
    make=4.4.1-r1 \
    g++=12.2.1_git20220924-r10

WORKDIR /build

# Copy package files
COPY package.json package-lock.json ./
COPY social_engine/ ./social_engine/

# Install dependencies and build
RUN npm ci --production && \
    npm run build && \
    npm prune --production

# Stage 4: Final Runtime
FROM node:18-alpine

# Install runtime dependencies
RUN apk add --no-cache \
    libstdc++=12.2.1_git20220924-r10 \
    vulkan-loader=1.3.250-r0 \
    cuda-runtime=12.0.0-r0 \
    redis=7.0.12-r0 \
    tini=0.19.0-r1

# Create non-root user
RUN addgroup -S tald && \
    adduser -S -G tald tald

# Set working directory
WORKDIR /app

# Copy built artifacts from previous stages
COPY --from=cpp-builder /build/lib/ /usr/local/lib/
COPY --from=rust-builder /build/target/release/ /usr/local/bin/
COPY --from=node-builder /build/node_modules/ ./node_modules/
COPY --from=node-builder /build/dist/ ./dist/

# Create required directories with proper permissions
RUN mkdir -p /data /cache && \
    chown -R tald:tald /data /cache /app

# Set environment variables
ENV NODE_ENV=production \
    RUST_LOG=info \
    LIDAR_SCAN_RATE=30 \
    FLEET_MAX_DEVICES=32

# Configure security limits
RUN ulimit -n 65535 && \
    echo "tald soft nofile 65535" >> /etc/security/limits.conf && \
    echo "tald hard nofile 65535" >> /etc/security/limits.conf

# Expose ports
EXPOSE 3000 3001 9090

# Set up volumes
VOLUME ["/data", "/cache"]

# Switch to non-root user
USER tald

# Health check configuration
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:3000/health || exit 1

# Use tini as init system
ENTRYPOINT ["/sbin/tini", "--"]

# Start the application
CMD ["node", "dist/server.js"]