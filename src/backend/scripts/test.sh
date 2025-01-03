#!/bin/bash

# TALD UNIA Backend Test Suite Runner
# Version: 1.0.0
# Description: Comprehensive test orchestration for all backend components

# Exit on any error
set -e

# Global constants
readonly TEST_TIMEOUT=300 # 5 minutes timeout for full test suite
readonly BUILD_DIR="./build"
readonly TEST_OUTPUT_DIR="./test-results"
readonly MAX_RETRIES=3 # Maximum number of test retries for flaky tests
readonly PARALLEL_JOBS=$(nproc) # Number of parallel test jobs
readonly MIN_GPU_MEMORY=4096 # Minimum required GPU memory in MB

# Color codes for output
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly NC='\033[0m' # No Color

# Verify required tools
check_requirements() {
    echo "Checking system requirements..."
    
    # Check cargo version (1.70)
    if ! cargo --version | grep -q "1.70"; then
        echo -e "${RED}Error: Required Cargo version 1.70 not found${NC}"
        exit 1
    }

    # Check cmake version (3.26)
    if ! cmake --version | grep -q "3.26"; then
        echo -e "${RED}Error: Required CMake version 3.26 not found${NC}"
        exit 1
    }

    # Check Node.js version for Jest (29.5)
    if ! node --version | grep -q "v18"; then
        echo -e "${RED}Error: Required Node.js v18.x not found${NC}"
        exit 1
    }
}

# Verify GPU requirements
check_gpu_requirements() {
    echo "Validating GPU requirements..."
    
    # Check CUDA availability
    if ! command -v nvidia-smi &> /dev/null; then
        echo -e "${RED}Error: NVIDIA GPU driver not found${NC}"
        return 1
    }

    # Check available GPU memory
    local available_memory=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -n1)
    if [ "$available_memory" -lt "$MIN_GPU_MEMORY" ]; then
        echo -e "${RED}Error: Insufficient GPU memory. Required: ${MIN_GPU_MEMORY}MB, Available: ${available_memory}MB${NC}"
        return 1
    }

    # Verify Vulkan support
    if ! command -v vulkaninfo &> /dev/null; then
        echo -e "${RED}Error: Vulkan runtime not found${NC}"
        return 1
    }

    return 0
}

# Run Rust-based fleet manager tests
run_rust_tests() {
    echo "Running Fleet Manager tests..."
    
    cd src/backend/fleet_manager
    
    # Clean previous test artifacts
    cargo clean
    
    # Run tests with retry mechanism
    local retry_count=0
    while [ $retry_count -lt $MAX_RETRIES ]; do
        if cargo test --all-features --release --jobs ${PARALLEL_JOBS} -- --test-threads ${PARALLEL_JOBS}; then
            echo -e "${GREEN}Fleet Manager tests passed${NC}"
            return 0
        fi
        
        retry_count=$((retry_count + 1))
        echo -e "${YELLOW}Retry attempt $retry_count for Fleet Manager tests${NC}"
    done
    
    echo -e "${RED}Fleet Manager tests failed after ${MAX_RETRIES} attempts${NC}"
    return 1
}

# Run C++ tests for LiDAR core and game engine
run_cpp_tests() {
    echo "Running C++ component tests..."
    
    # Create build directory
    mkdir -p $BUILD_DIR
    cd $BUILD_DIR
    
    # Configure CMake with GPU support
    cmake .. -DCMAKE_BUILD_TYPE=Release \
            -DBUILD_TESTING=ON \
            -DUSE_GPU=ON \
            -DCMAKE_CUDA_ARCHITECTURES=native
    
    # Build and run tests
    cmake --build . --target test --parallel ${PARALLEL_JOBS}
    
    # Run tests with CTest
    local retry_count=0
    while [ $retry_count -lt $MAX_RETRIES ]; do
        if ctest --output-on-failure --parallel ${PARALLEL_JOBS}; then
            echo -e "${GREEN}C++ tests passed${NC}"
            return 0
        fi
        
        retry_count=$((retry_count + 1))
        echo -e "${YELLOW}Retry attempt $retry_count for C++ tests${NC}"
    done
    
    echo -e "${RED}C++ tests failed after ${MAX_RETRIES} attempts${NC}"
    return 1
}

# Run Node.js social engine tests
run_node_tests() {
    echo "Running Social Engine tests..."
    
    cd src/backend/social_engine
    
    # Install dependencies
    npm ci
    
    # Run tests with coverage
    local retry_count=0
    while [ $retry_count -lt $MAX_RETRIES ]; do
        if npm test -- --coverage --detectOpenHandles --maxWorkers=${PARALLEL_JOBS}; then
            echo -e "${GREEN}Social Engine tests passed${NC}"
            return 0
        fi
        
        retry_count=$((retry_count + 1))
        echo -e "${YELLOW}Retry attempt $retry_count for Social Engine tests${NC}"
    done
    
    echo -e "${RED}Social Engine tests failed after ${MAX_RETRIES} attempts${NC}"
    return 1
}

# Generate comprehensive test report
generate_test_report() {
    echo "Generating test report..."
    
    mkdir -p $TEST_OUTPUT_DIR
    
    # Aggregate test results
    echo "Test Summary Report" > $TEST_OUTPUT_DIR/test_report.txt
    echo "==================" >> $TEST_OUTPUT_DIR/test_report.txt
    echo "Date: $(date)" >> $TEST_OUTPUT_DIR/test_report.txt
    echo "" >> $TEST_OUTPUT_DIR/test_report.txt
    
    # Collect coverage data
    echo "Coverage Summary:" >> $TEST_OUTPUT_DIR/test_report.txt
    if [ -f "src/backend/fleet_manager/target/coverage/index.html" ]; then
        echo "Fleet Manager Coverage: $(grep -oP 'Total.*?(\d+\.\d+%)' src/backend/fleet_manager/target/coverage/index.html)" >> $TEST_OUTPUT_DIR/test_report.txt
    fi
    
    if [ -f "$BUILD_DIR/coverage.info" ]; then
        echo "C++ Components Coverage: $(lcov --summary $BUILD_DIR/coverage.info | grep 'lines' | awk '{print $2}')" >> $TEST_OUTPUT_DIR/test_report.txt
    fi
    
    if [ -f "src/backend/social_engine/coverage/coverage-summary.json" ]; then
        echo "Social Engine Coverage: $(jq '.total.lines.pct' src/backend/social_engine/coverage/coverage-summary.json)%" >> $TEST_OUTPUT_DIR/test_report.txt
    fi
}

# Main execution
main() {
    echo "Starting TALD UNIA backend test suite..."
    
    # Create test output directory
    mkdir -p $TEST_OUTPUT_DIR
    
    # Check requirements
    check_requirements || exit 1
    check_gpu_requirements || exit 1
    
    # Start time measurement
    local start_time=$(date +%s)
    
    # Run component tests
    run_rust_tests || exit 1
    run_cpp_tests || exit 1
    run_node_tests || exit 1
    
    # Generate report
    generate_test_report
    
    # Calculate total execution time
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    echo -e "${GREEN}All tests completed successfully in ${duration} seconds${NC}"
    echo "Test report available at: $TEST_OUTPUT_DIR/test_report.txt"
}

# Execute with timeout
timeout $TEST_TIMEOUT bash -c main || {
    echo -e "${RED}Test suite execution timed out after ${TEST_TIMEOUT} seconds${NC}"
    exit 1
}