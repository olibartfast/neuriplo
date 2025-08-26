#!/bin/bash

# Script to run TensorRT tests in Docker container
# This script builds the TensorRT Docker image and runs the tests

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if Docker is running
check_docker() {
    if ! docker info > /dev/null 2>&1; then
        print_error "Docker is not running or not accessible"
        exit 1
    fi
    print_success "Docker is running"
}

# Function to build the TensorRT Docker image
build_image() {
    print_status "Building TensorRT Docker image..."
    
    # Check if Dockerfile exists
    if [ ! -f "docker/Dockerfile.tensorrt" ]; then
        print_error "Dockerfile.tensorrt not found in docker/ directory"
        exit 1
    fi
    
    # Build the image
    docker build -f docker/Dockerfile.tensorrt -t neuriplo-tensorrt .
    
    if [ $? -eq 0 ]; then
        print_success "TensorRT Docker image built successfully"
    else
        print_error "Failed to build TensorRT Docker image"
        exit 1
    fi
}

# Function to run tests in the container
run_tests() {
    print_status "Running TensorRT tests in Docker container..."
    
    # Run the container with tests
    docker run --rm \
        -v "$(pwd)/test_results:/app/test_results" \
        --gpus all \
        neuriplo-tensorrt \
        bash -c "
            echo 'Running TensorRT backend tests...'
            cd /app/test
            
            # Run the test executable
            if [ -f './TensorRTInferTest' ]; then
                echo 'Found test executable, running tests...'
                ./TensorRTInferTest
                TEST_EXIT_CODE=\$?
                
                if [ \$TEST_EXIT_CODE -eq 0 ]; then
                    echo 'All TensorRT tests passed!'
                    exit 0
                else
                    echo 'Some TensorRT tests failed!'
                    exit \$TEST_EXIT_CODE
                fi
            else
                echo 'Test executable not found. Checking test directory...'
                ls -la /app/test/
                echo 'Checking if tests were built...'
                find /app -name '*test*' -type f -executable
                exit 1
            fi
        "
    
    TEST_EXIT_CODE=$?
    
    if [ $TEST_EXIT_CODE -eq 0 ]; then
        print_success "All TensorRT tests passed!"
    else
        print_error "TensorRT tests failed with exit code $TEST_EXIT_CODE"
        exit $TEST_EXIT_CODE
    fi
}

# Function to run tests with verbose output
run_tests_verbose() {
    print_status "Running TensorRT tests with verbose output..."
    
    # Run the container with verbose tests
    docker run --rm \
        -v "$(pwd)/test_results:/app/test_results" \
        --gpus all \
        neuriplo-tensorrt \
        bash -c "
            echo 'Running TensorRT backend tests with verbose output...'
            cd /app/test
            
            # Run the test executable with verbose output
            if [ -f './TensorRTInferTest' ]; then
                echo 'Found test executable, running tests with verbose output...'
                ./TensorRTInferTest --verbose
                TEST_EXIT_CODE=\$?
                
                if [ \$TEST_EXIT_CODE -eq 0 ]; then
                    echo 'All TensorRT tests passed!'
                    exit 0
                else
                    echo 'Some TensorRT tests failed!'
                    exit \$TEST_EXIT_CODE
                fi
            else
                echo 'Test executable not found. Checking test directory...'
                ls -la /app/test/
                echo 'Checking if tests were built...'
                find /app -name '*test*' -type f -executable
                exit 1
            fi
        "
    
    TEST_EXIT_CODE=$?
    
    if [ $TEST_EXIT_CODE -eq 0 ]; then
        print_success "All TensorRT tests passed!"
    else
        print_error "TensorRT tests failed with exit code $TEST_EXIT_CODE"
        exit $TEST_EXIT_CODE
    fi
}

# Function to run tests with GPU profiling
run_tests_profiling() {
    print_status "Running TensorRT tests with GPU profiling..."
    
    # Run the container with profiling tests
    docker run --rm \
        -v "$(pwd)/test_results:/app/test_results" \
        --gpus all \
        neuriplo-tensorrt \
        bash -c "
            echo 'Running TensorRT backend tests with GPU profiling...'
            cd /app/test
            
            # Check if nvidia-smi is available
            if command -v nvidia-smi > /dev/null 2>&1; then
                echo 'GPU Information:'
                nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
            fi
            
            # Run the test executable with profiling
            if [ -f './TensorRTInferTest' ]; then
                echo 'Found test executable, running tests with profiling...'
                ./TensorRTInferTest --profiling
                TEST_EXIT_CODE=\$?
                
                if [ \$TEST_EXIT_CODE -eq 0 ]; then
                    echo 'All TensorRT tests passed!'
                    exit 0
                else
                    echo 'Some TensorRT tests failed!'
                    exit \$TEST_EXIT_CODE
                fi
            else
                echo 'Test executable not found. Checking test directory...'
                ls -la /app/test/
                echo 'Checking if tests were built...'
                find /app -name '*test*' -type f -executable
                exit 1
            fi
        "
    
    TEST_EXIT_CODE=$?
    
    if [ $TEST_EXIT_CODE -eq 0 ]; then
        print_success "All TensorRT tests passed!"
    else
        print_error "TensorRT tests failed with exit code $TEST_EXIT_CODE"
        exit $TEST_EXIT_CODE
    fi
}

# Function to run interactive shell in container
run_shell() {
    print_status "Starting interactive shell in TensorRT container..."
    
    docker run --rm -it \
        -v "$(pwd)/test_results:/app/test_results" \
        --gpus all \
        neuriplo-tensorrt \
        bash
}

# Function to check GPU availability
check_gpu() {
    print_status "Checking GPU availability..."
    
    if command -v nvidia-smi > /dev/null 2>&1; then
        print_success "NVIDIA GPU detected:"
        nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
    else
        print_warning "NVIDIA GPU not detected or nvidia-smi not available"
        print_warning "TensorRT tests may run on CPU only"
    fi
}

# Function to check if test executable exists
check_test_executable() {
    print_status "Checking if TensorRT test executable exists..."
    
    docker run --rm \
        --gpus all \
        neuriplo-tensorrt \
        bash -c "
            echo 'Checking for TensorRT test executable...'
            if [ -f '/app/test/TensorRTInferTest' ]; then
                echo 'Test executable found at /app/test/TensorRTInferTest'
                ls -la /app/test/TensorRTInferTest
                exit 0
            else
                echo 'Test executable not found at /app/test/TensorRTInferTest'
                echo 'Checking /app/test directory:'
                ls -la /app/test/ || echo 'Directory does not exist'
                echo 'Searching for test executables:'
                find /app -name '*test*' -type f -executable 2>/dev/null || echo 'No test executables found'
                exit 1
            fi
        "
    
    if [ $? -eq 0 ]; then
        print_success "TensorRT test executable found"
        return 0
    else
        print_error "TensorRT test executable not found"
        return 1
    fi
}

# Function to clean up Docker images
cleanup() {
    print_status "Cleaning up Docker images..."
    
    # Remove the TensorRT image
    docker rmi neuriplo-tensorrt 2>/dev/null || true
    
    print_success "Cleanup completed"
}

# Function to show usage
show_usage() {
    echo "Usage: $0 [OPTION]"
    echo ""
    echo "Options:"
    echo "  build       Build the TensorRT Docker image"
    echo "  rebuild     Rebuild the TensorRT Docker image (removes existing)"
    echo "  test        Run TensorRT tests (default)"
    echo "  test-verbose Run TensorRT tests with verbose output"
    echo "  test-profiling Run TensorRT tests with GPU profiling"
    echo "  test-debug  Run TensorRT tests with debugging information"
    echo "  check-test  Check if test executable exists"
    echo "  shell       Start interactive shell in container"
    echo "  check-gpu   Check GPU availability"
    echo "  cleanup     Remove Docker images"
    echo "  help        Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                    # Build and run tests"
    echo "  $0 build              # Only build the image"
    echo "  $0 rebuild             # Rebuild the image (removes existing)"
    echo "  $0 test               # Only run tests (requires built image)"
    echo "  $0 test-verbose       # Run tests with verbose output"
    echo "  $0 test-profiling     # Run tests with GPU profiling"
    echo "  $0 test-debug         # Run tests with debugging information"
    echo "  $0 check-test         # Check if test executable exists"
    echo "  $0 shell              # Start interactive shell"
    echo "  $0 check-gpu          # Check GPU availability"
    echo "  $0 cleanup            # Clean up Docker images"
    echo ""
    echo "Note: TensorRT tests require NVIDIA GPU and Docker with GPU support"
}

# Function to run tests with debugging information
run_tests_debug() {
    print_status "Running TensorRT tests with debugging information..."
    
    # Run the container with debug tests
    docker run --rm \
        -v "$(pwd)/test_results:/app/test_results" \
        --gpus all \
        neuriplo-tensorrt \
        bash -c "
            echo 'Running TensorRT backend tests with debugging information...'
            cd /app/test
            
            # Show environment information
            echo 'Environment Information:'
            echo 'LD_LIBRARY_PATH: '\$LD_LIBRARY_PATH
            echo 'TENSORRT_DIR: '\$TENSORRT_DIR
            echo 'Current directory: '\$(pwd)
            echo 'Contents of test directory:'
            ls -la /app/test/
            
            # Check if nvidia-smi is available
            if command -v nvidia-smi > /dev/null 2>&1; then
                echo 'GPU Information:'
                nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
            fi
            
            # Run the test executable with debugging
            if [ -f './TensorRTInferTest' ]; then
                echo 'Found test executable, running tests with debugging...'
                ./TensorRTInferTest --gtest_list_tests
                echo 'Running actual tests...'
                ./TensorRTInferTest --gtest_output=xml:test_results.xml
                TEST_EXIT_CODE=\$?
                
                if [ \$TEST_EXIT_CODE -eq 0 ]; then
                    echo 'All TensorRT tests passed!'
                    exit 0
                else
                    echo 'Some TensorRT tests failed!'
                    echo 'Test results:'
                    cat test_results.xml 2>/dev/null || echo 'No test results file found'
                    exit \$TEST_EXIT_CODE
                fi
            else
                echo 'Test executable not found. Checking test directory...'
                ls -la /app/test/
                echo 'Checking if tests were built...'
                find /app -name '*test*' -type f -executable
                exit 1
            fi
        "
    
    TEST_EXIT_CODE=$?
    
    if [ $TEST_EXIT_CODE -eq 0 ]; then
        print_success "All TensorRT tests passed!"
    else
        print_error "TensorRT tests failed with exit code $TEST_EXIT_CODE"
        exit $TEST_EXIT_CODE
    fi
}

# Function to rebuild the TensorRT Docker image
rebuild_image() {
    print_status "Rebuilding TensorRT Docker image..."
    
    # Remove existing image if it exists
    docker rmi neuriplo-tensorrt 2>/dev/null || true
    
    # Build the image
    build_image
}

# Main script logic
main() {
    # Check if we're in the right directory
    if [ ! -f "CMakeLists.txt" ]; then
        print_error "This script must be run from the project root directory"
        exit 1
    fi
    
    # Create test results directory
    mkdir -p test_results
    
    # Check Docker
    check_docker
    
    # Parse command line arguments
    case "${1:-test}" in
        "build")
            build_image
            ;;
        "rebuild")
            rebuild_image
            ;;
        "test")
            # Check if image exists, build if not
            if ! docker image inspect neuriplo-tensorrt >/dev/null 2>&1; then
                print_warning "TensorRT Docker image not found, building first..."
                build_image
            fi
            # Check if test executable exists
            if check_test_executable; then
                run_tests
            else
                print_error "Test executable not found. Please check the build process."
                exit 1
            fi
            ;;
        "test-verbose")
            # Check if image exists, build if not
            if ! docker image inspect neuriplo-tensorrt >/dev/null 2>&1; then
                print_warning "TensorRT Docker image not found, building first..."
                build_image
            fi
            # Check if test executable exists
            if check_test_executable; then
                run_tests_verbose
            else
                print_error "Test executable not found. Please check the build process."
                exit 1
            fi
            ;;
        "test-profiling")
            # Check if image exists, build if not
            if ! docker image inspect neuriplo-tensorrt >/dev/null 2>&1; then
                print_warning "TensorRT Docker image not found, building first..."
                build_image
            fi
            # Check if test executable exists
            if check_test_executable; then
                run_tests_profiling
            else
                print_error "Test executable not found. Please check the build process."
                exit 1
            fi
            ;;
        "test-debug")
            # Check if image exists, build if not
            if ! docker image inspect neuriplo-tensorrt >/dev/null 2>&1; then
                print_warning "TensorRT Docker image not found, building first..."
                build_image
            fi
            # Check if test executable exists
            if check_test_executable; then
                run_tests_debug
            else
                print_error "Test executable not found. Please check the build process."
                exit 1
            fi
            ;;
        "check-test")
            # Check if image exists, build if not
            if ! docker image inspect neuriplo-tensorrt >/dev/null 2>&1; then
                print_warning "TensorRT Docker image not found, building first..."
                build_image
            fi
            check_test_executable
            ;;
        "shell")
            # Check if image exists, build if not
            if ! docker image inspect neuriplo-tensorrt >/dev/null 2>&1; then
                print_warning "TensorRT Docker image not found, building first..."
                build_image
            fi
            run_shell
            ;;
        "check-gpu")
            check_gpu
            ;;
        "cleanup")
            cleanup
            ;;
        "help"|"-h"|"--help")
            show_usage
            ;;
        *)
            print_error "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
}

# Run main function with all arguments
main "$@" 