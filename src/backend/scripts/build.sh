#!/usr/bin/env bash

# TALD UNIA Backend Build Script
# Version: 1.0.0
# Description: Advanced build orchestration script for compiling and packaging 
# all TALD UNIA backend components with optimized performance settings

set -euo pipefail
IFS=$'\n\t'

# Build configuration
readonly BUILD_DIR="./build"
readonly INSTALL_DIR="./dist"
readonly CMAKE_BUILD_TYPE="Release"
readonly CARGO_PROFILE="release"
readonly NODE_ENV="production"
readonly PARALLEL_JOBS="$(nproc)"
readonly CCACHE_DIR="${BUILD_DIR}/.ccache"
readonly CARGO_TARGET_DIR="${BUILD_DIR}/target"
readonly NODE_OPTIONS="--max-old-space-size=8192"

# Required tool versions
readonly REQUIRED_CMAKE_VERSION="3.26"
readonly REQUIRED_CARGO_VERSION="1.70"
readonly REQUIRED_NODE_VERSION="18.0"

# Color codes for output
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly NC='\033[0m'

# Logging functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Version comparison function
version_gt() {
    test "$(printf '%s\n' "$@" | sort -V | head -n 1)" != "$1"
}

# Check dependencies and their versions
check_dependencies() {
    local status=0

    # Check CMake
    if ! command -v cmake &> /dev/null; then
        log_error "CMake not found. Please install CMake >= ${REQUIRED_CMAKE_VERSION}"
        status=1
    else
        local cmake_version=$(cmake --version | head -n1 | cut -d' ' -f3)
        if version_gt "${REQUIRED_CMAKE_VERSION}" "${cmake_version}"; then
            log_error "CMake version ${cmake_version} is too old. Required >= ${REQUIRED_CMAKE_VERSION}"
            status=1
        fi
    fi

    # Check Cargo
    if ! command -v cargo &> /dev/null; then
        log_error "Cargo not found. Please install Rust >= ${REQUIRED_CARGO_VERSION}"
        status=1
    else
        local cargo_version=$(cargo --version | cut -d' ' -f2)
        if version_gt "${REQUIRED_CARGO_VERSION}" "${cargo_version}"; then
            log_error "Cargo version ${cargo_version} is too old. Required >= ${REQUIRED_CARGO_VERSION}"
            status=1
        fi
    fi

    # Check Node.js
    if ! command -v node &> /dev/null; then
        log_error "Node.js not found. Please install Node.js >= ${REQUIRED_NODE_VERSION}"
        status=1
    else
        local node_version=$(node --version | cut -d'v' -f2)
        if version_gt "${REQUIRED_NODE_VERSION}" "${node_version}"; then
            log_error "Node.js version ${node_version} is too old. Required >= ${REQUIRED_NODE_VERSION}"
            status=1
        fi
    fi

    # Check CUDA toolkit
    if ! command -v nvcc &> /dev/null; then
        log_error "CUDA toolkit not found. Please install CUDA >= 12.0"
        status=1
    fi

    # Check Vulkan SDK
    if [ -z "${VULKAN_SDK:-}" ]; then
        log_error "Vulkan SDK not found. Please install Vulkan SDK >= 1.3"
        status=1
    fi

    return $status
}

# Setup build environment
setup_build_environment() {
    # Create build directories
    mkdir -p "${BUILD_DIR}"
    mkdir -p "${INSTALL_DIR}"
    mkdir -p "${CCACHE_DIR}"
    mkdir -p "${CARGO_TARGET_DIR}"

    # Configure ccache
    export CCACHE_DIR
    export CCACHE_MAXSIZE="10G"
    export CCACHE_COMPRESS=1

    # Configure Cargo
    export CARGO_TARGET_DIR
    export CARGO_HOME="${BUILD_DIR}/.cargo"
    export RUSTFLAGS="-C target-cpu=native -C lto=thin"

    # Configure Node.js
    export NODE_ENV
    export NODE_OPTIONS
    export npm_config_cache="${BUILD_DIR}/.npm"

    log_info "Build environment setup complete"
}

# Build C++ components
build_cpp_components() {
    log_info "Building C++ components..."
    
    local cmake_build_dir="${BUILD_DIR}/cpp"
    mkdir -p "${cmake_build_dir}"
    cd "${cmake_build_dir}"

    # Configure CMake with optimizations
    cmake ../.. \
        -GNinja \
        -DCMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE}" \
        -DCMAKE_C_COMPILER_LAUNCHER=ccache \
        -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
        -DCMAKE_INSTALL_PREFIX="${INSTALL_DIR}" \
        -DENABLE_PGO=ON \
        -DUSE_SIMD=ON \
        -DENABLE_CUDA=ON \
        -DENABLE_VULKAN=ON

    # Build components in parallel
    cmake --build . --parallel "${PARALLEL_JOBS}"

    # Run tests if enabled
    if [ "${BUILD_TESTING:-ON}" = "ON" ]; then
        ctest --output-on-failure --parallel "${PARALLEL_JOBS}"
    fi

    # Install
    cmake --install .
    
    cd - > /dev/null
    log_info "C++ components built successfully"
}

# Build Rust components
build_rust_components() {
    log_info "Building Rust components..."

    # Build with optimizations
    RUSTFLAGS="-C target-cpu=native -C lto=thin" \
    cargo build \
        --profile="${CARGO_PROFILE}" \
        --workspace \
        --features="production" \
        --jobs="${PARALLEL_JOBS}"

    # Run tests
    if [ "${BUILD_TESTING:-ON}" = "ON" ]; then
        cargo test --workspace --profile="${CARGO_PROFILE}"
    fi

    # Copy artifacts
    mkdir -p "${INSTALL_DIR}/bin"
    find "${CARGO_TARGET_DIR}/${CARGO_PROFILE}" -maxdepth 1 -type f -executable \
        -exec cp {} "${INSTALL_DIR}/bin/" \;

    log_info "Rust components built successfully"
}

# Build Node.js components
build_node_components() {
    log_info "Building Node.js components..."

    # Install production dependencies
    npm ci --production --no-audit

    # Build TypeScript
    npm run build

    # Run tests
    if [ "${BUILD_TESTING:-ON}" = "ON" ]; then
        npm test
    fi

    # Copy dist files
    cp -r dist/* "${INSTALL_DIR}/"
    
    log_info "Node.js components built successfully"
}

# Verify build artifacts
verify_build() {
    local status=0

    # Check required binaries
    local required_binaries=(
        "${INSTALL_DIR}/bin/lidar_core"
        "${INSTALL_DIR}/bin/game_engine"
        "${INSTALL_DIR}/bin/fleet_manager"
        "${INSTALL_DIR}/bin/social_engine"
    )

    for binary in "${required_binaries[@]}"; do
        if [ ! -f "${binary}" ]; then
            log_error "Missing required binary: ${binary}"
            status=1
        fi
    done

    # Check libraries
    local required_libs=(
        "${INSTALL_DIR}/lib/liblidar_core.so"
        "${INSTALL_DIR}/lib/libgame_engine.so"
    )

    for lib in "${required_libs[@]}"; do
        if [ ! -f "${lib}" ]; then
            log_error "Missing required library: ${lib}"
            status=1
        fi
    done

    return $status
}

# Main build pipeline
main() {
    log_info "Starting TALD UNIA backend build..."

    # Check dependencies
    if ! check_dependencies; then
        log_error "Dependency check failed"
        exit 1
    fi

    # Setup build environment
    setup_build_environment

    # Clean previous build
    rm -rf "${BUILD_DIR}" "${INSTALL_DIR}"

    # Build all components
    build_cpp_components
    build_rust_components
    build_node_components

    # Verify build
    if ! verify_build; then
        log_error "Build verification failed"
        exit 1
    fi

    log_info "TALD UNIA backend build completed successfully"
}

# Execute main function
main "$@"