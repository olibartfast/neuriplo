#!/bin/bash

# Build and run Cactus backend unit tests in Docker.
# Usage: ./run_cactus_tests.sh [--build-only] [--run-only] [--clean] [--verbose]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
IMAGE_NAME="neuriplo:cactus"
CONTAINER_NAME="cactus-test-runner"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_status()  { echo -e "${BLUE}[INFO]${NC} $1"; }
print_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
print_error()   { echo -e "${RED}[ERROR]${NC} $1"; }

show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --build-only   Only build the Docker image"
    echo "  --run-only     Only run tests (assumes image exists)"
    echo "  --clean        Clean up containers and images"
    echo "  --verbose      Run tests with verbose output"
    echo "  --help         Show this help message"
}

cleanup() {
    print_status "Cleaning up Docker resources..."
    docker ps -q  -f name="$CONTAINER_NAME" | grep -q . && docker stop "$CONTAINER_NAME" 2>/dev/null || true
    docker ps -aq -f name="$CONTAINER_NAME" | grep -q . && docker rm  "$CONTAINER_NAME" 2>/dev/null || true
    docker images -q "$IMAGE_NAME"           | grep -q . && docker rmi "$IMAGE_NAME"    2>/dev/null || true
    print_success "Cleanup completed"
}

build_image() {
    print_status "Building Cactus Docker image..."
    cd "$PROJECT_ROOT"
    docker build --rm -t "$IMAGE_NAME" -f docker/Dockerfile.cactus .
    print_success "Docker image built: $IMAGE_NAME"
}

run_tests() {
    local extra_args=("$@")
    print_status "Running Cactus backend unit tests..."

    if ! docker images -q "$IMAGE_NAME" | grep -q .; then
        print_error "Docker image $IMAGE_NAME not found. Build it first with --build-only"
        exit 1
    fi

    mkdir -p "$PROJECT_ROOT/test_results"

    docker run --rm \
        --name "$CONTAINER_NAME" \
        -v "$PROJECT_ROOT/test_results:/app/test_results" \
        "${extra_args[@]}" \
        "$IMAGE_NAME" \
        /app/build/backends/cactus/test/CactusInferTest

    print_success "Cactus tests completed"
}

BUILD_ONLY=false
RUN_ONLY=false
CLEAN_ONLY=false
VERBOSE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --build-only) BUILD_ONLY=true ;;
        --run-only)   RUN_ONLY=true   ;;
        --clean)      CLEAN_ONLY=true ;;
        --verbose)    VERBOSE=true    ;;
        --help)       show_usage; exit 0 ;;
        *) print_error "Unknown option: $1"; show_usage; exit 1 ;;
    esac
    shift
done

if [ "$CLEAN_ONLY" = true ]; then cleanup; exit 0; fi
if [ "$BUILD_ONLY" = true ]; then build_image; exit 0; fi

EXTRA_RUN_ARGS=()
if [ "$VERBOSE" = true ]; then
    EXTRA_RUN_ARGS+=(-e GTEST_COLOR=1)
fi

if [ "$RUN_ONLY" = true ]; then
    run_tests "${EXTRA_RUN_ARGS[@]}"
    exit 0
fi

build_image
run_tests "${EXTRA_RUN_ARGS[@]}"
print_success "Cactus backend testing completed!"
