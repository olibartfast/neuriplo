#!/bin/bash

# Script to build and run OpenVINO backend unit tests in Docker
# Usage: ./run_openvino_tests.sh [--build-only] [--run-only] [--clean]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
IMAGE_NAME="inference-engines:openvino"
CONTAINER_NAME="openvino-test-runner"

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

# Function to show usage
show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --build-only    Only build the Docker image"
    echo "  --run-only      Only run tests (assumes image exists)"
    echo "  --clean         Clean up containers and images"
    echo "  --help          Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                    # Build and run tests"
    echo "  $0 --build-only       # Only build the image"
    echo "  $0 --run-only         # Only run tests"
    echo "  $0 --clean            # Clean up Docker resources"
}

# Function to clean up Docker resources
cleanup() {
    print_status "Cleaning up Docker resources..."
    
    # Stop and remove container if running
    if docker ps -q -f name="$CONTAINER_NAME" | grep -q .; then
        docker stop "$CONTAINER_NAME" 2>/dev/null || true
        docker rm "$CONTAINER_NAME" 2>/dev/null || true
    fi
    
    # Remove container if exists
    if docker ps -aq -f name="$CONTAINER_NAME" | grep -q .; then
        docker rm "$CONTAINER_NAME" 2>/dev/null || true
    fi
    
    # Remove image if exists
    if docker images -q "$IMAGE_NAME" | grep -q .; then
        docker rmi "$IMAGE_NAME" 2>/dev/null || true
    fi
    
    print_success "Cleanup completed"
}

# Function to build Docker image
build_image() {
    print_status "Building OpenVINO Docker image..."
    
    cd "$PROJECT_ROOT"
    
    # Build the Docker image
    docker build --rm -t "$IMAGE_NAME" -f docker/Dockerfile.openvino .
    
    if [ $? -eq 0 ]; then
        print_success "Docker image built successfully: $IMAGE_NAME"
    else
        print_error "Failed to build Docker image"
        exit 1
    fi
}

# Function to run tests
run_tests() {
    print_status "Running OpenVINO backend unit tests..."
    
    # Check if image exists
    if ! docker images -q "$IMAGE_NAME" | grep -q .; then
        print_error "Docker image $IMAGE_NAME not found. Please build it first with --build-only"
        exit 1
    fi
    
    # Create test results directory
    mkdir -p "$PROJECT_ROOT/test_results"
    
    # Run the container with test execution
    docker run --rm \
        --name "$CONTAINER_NAME" \
        -v "$PROJECT_ROOT/test_results:/app/test_results" \
        "$IMAGE_NAME" \
        /app/build/backends/openvino/test/OpenVINOInferTest
    
    if [ $? -eq 0 ]; then
        print_success "OpenVINO tests completed successfully"
        print_status "Test results saved to: $PROJECT_ROOT/test_results"
    else
        print_error "OpenVINO tests failed"
        exit 1
    fi
}

# Function to run tests with verbose output
run_tests_verbose() {
    print_status "Running OpenVINO backend unit tests with verbose output..."
    
    # Check if image exists
    if ! docker images -q "$IMAGE_NAME" | grep -q .; then
        print_error "Docker image $IMAGE_NAME not found. Please build it first with --build-only"
        exit 1
    fi
    
    # Create test results directory
    mkdir -p "$PROJECT_ROOT/test_results"
    
    # Run the container with verbose test execution
    docker run --rm \
        --name "$CONTAINER_NAME" \
        -v "$PROJECT_ROOT/test_results:/app/test_results" \
        -e GTEST_COLOR=1 \
        "$IMAGE_NAME" \
        /app/build/backends/openvino/test/OpenVINOInferTest --gtest_color=yes --gtest_verbose
}

# Parse command line arguments
BUILD_ONLY=false
RUN_ONLY=false
CLEAN_ONLY=false
VERBOSE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --build-only)
            BUILD_ONLY=true
            shift
            ;;
        --run-only)
            RUN_ONLY=true
            shift
            ;;
        --clean)
            CLEAN_ONLY=true
            shift
            ;;
        --verbose)
            VERBOSE=true
            shift
            ;;
        --help)
            show_usage
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Main execution logic
if [ "$CLEAN_ONLY" = true ]; then
    cleanup
    exit 0
fi

if [ "$BUILD_ONLY" = true ]; then
    build_image
    exit 0
fi

if [ "$RUN_ONLY" = true ]; then
    if [ "$VERBOSE" = true ]; then
        run_tests_verbose
    else
        run_tests
    fi
    exit 0
fi

# Default: build and run
print_status "Building and running OpenVINO backend unit tests..."
build_image

if [ "$VERBOSE" = true ]; then
    run_tests_verbose
else
    run_tests
fi

print_success "OpenVINO backend testing completed successfully!" 