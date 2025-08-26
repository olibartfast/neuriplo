#!/bin/bash

# Script to run LibTorch tests in Docker container
# This script builds the LibTorch Docker image and runs the tests

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

# Function to build the LibTorch Docker image
build_image() {
    print_status "Building LibTorch Docker image..."
    
    # Check if Dockerfile exists
    if [ ! -f "docker/Dockerfile.libtorch" ]; then
        print_error "Dockerfile.libtorch not found in docker/ directory"
        exit 1
    fi
    
    # Build the image
    docker build -f docker/Dockerfile.libtorch -t neuriplo-libtorch .
    
    if [ $? -eq 0 ]; then
        print_success "LibTorch Docker image built successfully"
    else
        print_error "Failed to build LibTorch Docker image"
        exit 1
    fi
}

# Function to run tests in the container
run_tests() {
    print_status "Running LibTorch tests in Docker container..."
    
    # Run the container with tests
    docker run --rm \
        -v "$(pwd)/test_results:/app/test_results" \
        neuriplo-libtorch \
        bash -c "
            echo 'Running LibTorch backend tests...'
            cd /app/build/backends/libtorch/test
            
            # Run the test executable
            if [ -f './LibtorchInferTest' ]; then
                echo 'Found test executable, running tests...'
                ./LibtorchInferTest
                TEST_EXIT_CODE=\$?
                
                if [ \$TEST_EXIT_CODE -eq 0 ]; then
                    echo 'All LibTorch tests passed!'
                    exit 0
                else
                    echo 'Some LibTorch tests failed!'
                    exit \$TEST_EXIT_CODE
                fi
            else
                echo 'Test executable not found. Checking build directory...'
                ls -la /app/build/backends/libtorch/test/
                echo 'Checking if tests were built...'
                find /app/build -name '*test*' -type f -executable
                exit 1
            fi
        "
    
    TEST_EXIT_CODE=$?
    
    if [ $TEST_EXIT_CODE -eq 0 ]; then
        print_success "All LibTorch tests passed!"
    else
        print_error "LibTorch tests failed with exit code $TEST_EXIT_CODE"
        exit $TEST_EXIT_CODE
    fi
}

# Function to run tests with verbose output
run_tests_verbose() {
    print_status "Running LibTorch tests with verbose output..."
    
    # Run the container with verbose tests
    docker run --rm \
        -v "$(pwd)/test_results:/app/test_results" \
        neuriplo-libtorch \
        bash -c "
            echo 'Running LibTorch backend tests with verbose output...'
            cd /app/build/backends/libtorch/test
            
            # Run the test executable with verbose output
            if [ -f './LibtorchInferTest' ]; then
                echo 'Found test executable, running tests with verbose output...'
                ./LibtorchInferTest --verbose
                TEST_EXIT_CODE=\$?
                
                if [ \$TEST_EXIT_CODE -eq 0 ]; then
                    echo 'All LibTorch tests passed!'
                    exit 0
                else
                    echo 'Some LibTorch tests failed!'
                    exit \$TEST_EXIT_CODE
                fi
            else
                echo 'Test executable not found. Checking build directory...'
                ls -la /app/build/backends/libtorch/test/
                echo 'Checking if tests were built...'
                find /app/build -name '*test*' -type f -executable
                exit 1
            fi
        "
    
    TEST_EXIT_CODE=$?
    
    if [ $TEST_EXIT_CODE -eq 0 ]; then
        print_success "All LibTorch tests passed!"
    else
        print_error "LibTorch tests failed with exit code $TEST_EXIT_CODE"
        exit $TEST_EXIT_CODE
    fi
}

# Function to run interactive shell in container
run_shell() {
    print_status "Starting interactive shell in LibTorch container..."
    
    docker run --rm -it \
        -v "$(pwd)/test_results:/app/test_results" \
        neuriplo-libtorch \
        bash
}

# Function to clean up Docker images
cleanup() {
    print_status "Cleaning up Docker images..."
    
    # Remove the LibTorch image
    docker rmi neuriplo-libtorch 2>/dev/null || true
    
    print_success "Cleanup completed"
}

# Function to show usage
show_usage() {
    echo "Usage: $0 [OPTION]"
    echo ""
    echo "Options:"
    echo "  build       Build the LibTorch Docker image"
    echo "  test        Run LibTorch tests (default)"
    echo "  test-verbose Run LibTorch tests with verbose output"
    echo "  shell       Start interactive shell in container"
    echo "  cleanup     Remove Docker images"
    echo "  help        Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                    # Build and run tests"
    echo "  $0 build              # Only build the image"
    echo "  $0 test               # Only run tests (requires built image)"
    echo "  $0 test-verbose       # Run tests with verbose output"
    echo "  $0 shell              # Start interactive shell"
    echo "  $0 cleanup            # Clean up Docker images"
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
        "test")
            # Check if image exists, build if not
            if ! docker image inspect neuriplo-libtorch >/dev/null 2>&1; then
                print_warning "LibTorch Docker image not found, building first..."
                build_image
            fi
            run_tests
            ;;
        "test-verbose")
            # Check if image exists, build if not
            if ! docker image inspect neuriplo-libtorch >/dev/null 2>&1; then
                print_warning "LibTorch Docker image not found, building first..."
                build_image
            fi
            run_tests_verbose
            ;;
        "shell")
            # Check if image exists, build if not
            if ! docker image inspect neuriplo-libtorch >/dev/null 2>&1; then
                print_warning "LibTorch Docker image not found, building first..."
                build_image
            fi
            run_shell
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