#!/bin/bash

# Backend Testing Script for InferenceEngines
# This script tests each backend individually, builds them, and runs unit tests
# Author: Generated for InferenceEngines project
# Date: $(date)

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Project root directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BUILD_DIR="$PROJECT_ROOT/build"
TEST_RESULTS_DIR="$PROJECT_ROOT/test_results"

# Supported backends
BACKENDS=("OPENCV_DNN" "ONNX_RUNTIME" "LIBTORCH" "LIBTENSORFLOW" "TENSORRT" "OPENVINO")

# Log functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if a backend is available
check_backend_availability() {
    local backend=$1
    log_info "Checking availability of backend: $backend"
    
    case $backend in
        "OPENCV_DNN")
            if pkg-config --exists opencv4; then
                local opencv_version=$(pkg-config --modversion opencv4)
                if [[ "$(printf '%s\n' "$OPENCV_MIN_VERSION" "$opencv_version" | sort -V | head -n1)" = "$OPENCV_MIN_VERSION" ]]; then
                    log_success "OpenCV $opencv_version found (meets minimum $OPENCV_MIN_VERSION)"
                    return 0
                else
                    log_warning "OpenCV $opencv_version found but version $OPENCV_MIN_VERSION or higher is required"
                    return 1
                fi
            else
                log_warning "OpenCV not found"
                return 1
            fi
            ;;
        "ONNX_RUNTIME")
            local onnx_runtime_dir="${HOME}/onnxruntime"
            local alt_onnx_runtime_dir="/usr/local/lib/onnxruntime"
            local expected_version="$ONNX_RUNTIME_VERSION"
            
            if [ -d "$onnx_runtime_dir" ] || [ -d "$alt_onnx_runtime_dir" ]; then
                # Try to verify version if onnxruntime is in path
                if command -v onnxruntime_INFO > /dev/null 2>&1; then
                    local installed_version=$(onnxruntime_INFO | grep "ONNX Runtime version" | awk '{print $4}')
                    if [ "$installed_version" = "$expected_version" ]; then
                        log_success "ONNX Runtime $installed_version found (matches expected $expected_version)"
                    else
                        log_warning "ONNX Runtime $installed_version found but version $expected_version is expected"
                    fi
                else
                    log_success "ONNX Runtime found, but couldn't verify version"
                fi
                return 0
            else
                log_warning "ONNX Runtime not found"
                return 1
            fi
            ;;
        "LIBTORCH")
            local libtorch_dir="${HOME}/libtorch"
            local alt_libtorch_dir="/usr/local/lib/libtorch"
            local expected_version="$LIBTORCH_VERSION"
            
            if [ -d "$libtorch_dir" ] || [ -d "$alt_libtorch_dir" ]; then
                # Try to verify version from version.txt if it exists
                local version_file="$libtorch_dir/share/cmake/Torch/TorchConfigVersion.cmake"
                if [ -f "$version_file" ]; then
                    local installed_version=$(grep "set(PACKAGE_VERSION" "$version_file" | cut -d'"' -f2)
                    if [ "$installed_version" = "$expected_version" ]; then
                        log_success "LibTorch $installed_version found (matches expected $expected_version)"
                    else
                        log_warning "LibTorch $installed_version found but version $expected_version is expected"
                    fi
                else
                    log_success "LibTorch found, but couldn't verify version"
                fi
                return 0
            else
                log_warning "LibTorch not found"
                return 1
            fi
            ;;
        "LIBTENSORFLOW")
            local expected_version="$TENSORFLOW_VERSION"
            
            if pkg-config --exists tensorflow; then
                # Check version if possible
                local installed_version=$(pkg-config --modversion tensorflow 2>/dev/null || echo "unknown")
                if [ "$installed_version" != "unknown" ]; then
                    if [ "$installed_version" = "$expected_version" ]; then
                        log_success "TensorFlow $installed_version found (matches expected $expected_version)"
                    else
                        log_warning "TensorFlow $installed_version found but version $expected_version is expected"
                    fi
                else
                    log_success "TensorFlow found, but couldn't verify version"
                fi
                return 0
            else
                log_warning "TensorFlow not found"
                return 1
            fi
            ;;
        "TENSORRT")
            local tensorrt_dir_pattern="${HOME}/TensorRT-*"
            local alt_tensorrt_dir="/usr/local/TensorRT"
            local expected_version="$TENSORRT_VERSION"
            
            if [ -d "${HOME}/TensorRT-${expected_version}" ]; then
                log_success "TensorRT ${expected_version} found (exact match)"
                return 0
            elif [ -d "$alt_tensorrt_dir" ]; then
                # Check version if possible from version.json if it exists
                local version_file="$alt_tensorrt_dir/version.json"
                if [ -f "$version_file" ]; then
                    local installed_version=$(grep '"version"' "$version_file" | cut -d'"' -f4)
                    if [ "$installed_version" = "$expected_version" ]; then
                        log_success "TensorRT $installed_version found (matches expected $expected_version)"
                    else
                        log_warning "TensorRT $installed_version found but version $expected_version is expected"
                    fi
                else
                    log_success "TensorRT found, but couldn't verify version"
                fi
                return 0
            elif compgen -G "$tensorrt_dir_pattern" > /dev/null; then
                log_warning "TensorRT found but version may not match $expected_version"
                return 0
            else
                log_warning "TensorRT not found"
                return 1
            fi
            ;;
        "OPENVINO")
            local openvino_dir_pattern="${HOME}/intel/openvino*"
            local alt_openvino_dir="/opt/intel/openvino"
            local expected_version="$OPENVINO_VERSION"
            
            if [ -d "${HOME}/intel/openvino_${expected_version}" ]; then
                log_success "OpenVINO ${expected_version} found (exact match)"
                return 0
            elif [ -d "$alt_openvino_dir" ]; then
                # Check version if possible from version.txt if it exists
                if command -v openVINO_INFO > /dev/null 2>&1; then
                    local installed_version=$(openVINO_INFO 2>/dev/null | grep "OpenVINO version" | awk '{print $3}')
                    if [ "$installed_version" = "$expected_version" ]; then
                        log_success "OpenVINO $installed_version found (matches expected $expected_version)"
                    else
                        log_warning "OpenVINO $installed_version found but version $expected_version is expected"
                    fi
                else
                    log_success "OpenVINO found, but couldn't verify version"
                fi
                return 0
            elif compgen -G "$openvino_dir_pattern" > /dev/null; then
                log_warning "OpenVINO found but version may not match $expected_version"
                return 0
            else
                log_warning "OpenVINO not found"
                return 1
            fi
            ;;
        *)
            log_error "Unknown backend: $backend"
            return 1
            ;;
    esac
}

# Function to get backend directory name
get_backend_dir() {
    local backend=$1
    case $backend in
        "OPENCV_DNN") echo "opencv-dnn" ;;
        "ONNX_RUNTIME") echo "onnx-runtime" ;;
        "LIBTORCH") echo "libtorch" ;;
        "LIBTENSORFLOW") echo "libtensorflow" ;;
        "TENSORRT") echo "tensorrt" ;;
        "OPENVINO") echo "openvino" ;;
        *) echo "${backend,,}" ;;
    esac
}

# Function to build a specific backend
build_backend() {
    local backend=$1
    local backend_dir=$(get_backend_dir "$backend")
    local build_dir="$BUILD_DIR/$backend_dir"
    
    log_info "Building backend: $backend"
    log_info "Dependency versions will be validated by CMake during configuration..."
    
    # Create build directory
    mkdir -p "$build_dir"
    cd "$build_dir"
    
    # Configure with CMake
    log_info "Configuring CMake for $backend..."
    if ! cmake -DDEFAULT_BACKEND="$backend" \
               -DBUILD_INFERENCE_ENGINE_TESTS=ON \
               -DCMAKE_BUILD_TYPE=Release \
               "$PROJECT_ROOT"; then
        log_error "CMake configuration failed for $backend"
        log_warning "This may be due to missing or incompatible dependencies"
        log_warning "Check that all dependencies meet version requirements in cmake/versions.cmake"
        return 1
    fi
    
    # Build
    log_info "Building $backend..."
    if ! cmake --build . --parallel $(nproc); then
        log_error "Build failed for $backend"
        return 1
    fi
    
    log_success "Successfully built $backend"
    return 0
}

# Function to create dummy model for backend
create_dummy_model() {
    local backend=$1
    local test_dir="$2"
    
    log_info "Creating dummy model for $backend backend..."
    cd "$test_dir"
    
    case $backend in
        "OPENCV_DNN"|"ONNX_RUNTIME"|"OPENVINO")
            # Create a minimal ONNX model with proper structure
            python3 << 'EOF'
import struct
import os

# Create minimal ONNX file with ResNet-18 structure
def create_minimal_onnx(filename="resnet18.onnx"):
    # ONNX file magic number and basic structure
    content = bytearray()
    
    # ONNX magic header
    content.extend(b'\x08\x07')  # ONNX version
    
    # Model name
    model_name = b'ResNet18'
    content.extend(struct.pack('<I', len(model_name)))
    content.extend(model_name)
    
    # Input tensor info (name: "input", shape: [1,3,224,224])
    input_name = b'input'
    content.extend(struct.pack('<I', len(input_name)))
    content.extend(input_name)
    content.extend(struct.pack('<IIII', 1, 3, 224, 224))  # NCHW
    
    # Output tensor info (name: "output", shape: [1,1000])
    output_name = b'output'
    content.extend(struct.pack('<I', len(output_name)))
    content.extend(output_name)
    content.extend(struct.pack('<II', 1, 1000))  # Classification output
    
    # Pad to reasonable size
    content.extend(b'\x00' * (1024 - len(content)))
    
    with open(filename, 'wb') as f:
        f.write(content)
    
    print(f"Created dummy ONNX: {filename}")
    return True

create_minimal_onnx()
EOF
            echo "resnet18.onnx" > model_path.txt
            ;;
        "LIBTORCH")
            # Create dummy TorchScript model
            python3 << 'EOF'
try:
    import torch
    import torch.nn as nn
    
    class DummyResNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 64, 7, stride=2, padding=3)
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(64, 1000)
            
        def forward(self, x):
            x = self.conv(x)
            x = self.pool(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            return x
    
    model = DummyResNet()
    model.eval()
    
    # Trace the model
    dummy_input = torch.randn(1, 3, 224, 224)
    traced_model = torch.jit.trace(model, dummy_input)
    traced_model.save("resnet18.pt")
    print("Created dummy TorchScript model")
    
except ImportError:
    # Fallback: create dummy file
    with open("resnet18.pt", "wb") as f:
        f.write(b"dummy_torchscript_model")
    print("Created dummy TorchScript file")
EOF
            echo "resnet18.pt" > model_path.txt
            ;;
        "LIBTENSORFLOW")
            # Create dummy SavedModel
            python3 << 'EOF'
try:
    import tensorflow as tf
    import os
    
    # Create a simple model
    inputs = tf.keras.Input(shape=(224, 224, 3), name='input')
    x = tf.keras.layers.Conv2D(64, 7, strides=2, padding='same')(inputs)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    outputs = tf.keras.layers.Dense(1000, name='output')(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    # Save as SavedModel
    if os.path.exists("saved_model"):
        import shutil
        shutil.rmtree("saved_model")
    
    tf.saved_model.save(model, "saved_model")
    print("Created dummy TensorFlow SavedModel")
    
except ImportError:
    # Fallback: create dummy directory structure
    os.makedirs("saved_model", exist_ok=True)
    with open("saved_model/saved_model.pb", "wb") as f:
        f.write(b"dummy_tensorflow_model")
    print("Created dummy SavedModel structure")
EOF
            echo "saved_model" > model_path.txt
            ;;
        "TENSORRT")
            # For TensorRT, create dummy engine file
            echo "dummy_tensorrt_engine" > resnet18.engine
            echo "resnet18.engine" > model_path.txt
            log_warning "Created dummy TensorRT engine (tests may fail without real engine)"
            ;;
    esac
    
    log_success "Dummy model created for $backend"
    return 0
}

# Function to setup model for backend
setup_backend_model() {
    local backend=$1
    local test_dir="$2"
    
    log_info "Setting up model for $backend backend..."
    
    # Strategy 1: Try dynamic model downloader first
    local model_downloader="$PROJECT_ROOT/scripts/model_downloader.py"
    
    if [ -f "$model_downloader" ]; then
        log_info "Using model downloader to set up model for $backend"
        cd "$test_dir"
        
        # Download/generate model for this backend directly to test directory
        if python3 "$model_downloader" "$backend" --output-dir "$test_dir" --keep-temp; then
            # Create model_path.txt for compatibility with existing tests
            if [ -f "resnet18.onnx" ]; then
                echo "resnet18.onnx" > model_path.txt
            elif [ -f "resnet18.pt" ]; then
                echo "resnet18.pt" > model_path.txt
            elif [ -f "resnet18.engine" ]; then
                echo "resnet18.engine" > model_path.txt
            elif [ -f "resnet18.xml" ]; then
                echo "resnet18.xml" > model_path.txt
            elif [ -d "saved_model" ]; then
                echo "saved_model" > model_path.txt
            fi
            
            log_success "Model setup completed for $backend using downloader"
            return 0
        else
            log_warning "Model downloader failed, creating dummy model..."
        fi
    else
        log_warning "Model downloader not found, creating dummy model..."
    fi
    
    # Strategy 2: Create dummy model as fallback
    if create_dummy_model "$backend" "$test_dir"; then
        log_success "Dummy model created for $backend"
        return 0
    else
        log_error "Failed to create dummy model for $backend"
        return 1
    fi
}

# Function to cleanup backend models
cleanup_backend_model() {
    local test_dir="$1"
    
    if [ -d "$test_dir" ]; then
        cd "$test_dir"
        
        # Remove downloaded models but keep test scripts
        rm -f resnet18.onnx resnet18.pt resnet18.engine resnet18.xml resnet18.bin
        rm -rf saved_model
        rm -f model_path.txt
        
        log_info "Cleaned up temporary models"
    fi
}

# Function to run tests for a specific backend
run_backend_tests() {
    local backend=$1
    local backend_dir=$(get_backend_dir "$backend")
    local build_dir="$BUILD_DIR/$backend_dir"
    local results_file="$TEST_RESULTS_DIR/${backend_dir}_results.xml"
    
    log_info "Running tests for backend: $backend"
    
    cd "$build_dir"
    
    # Create test results directory
    mkdir -p "$TEST_RESULTS_DIR"
    
    # Find and run the test executable
    local test_executable=""
    case $backend in
        "OPENCV_DNN")
            test_executable="backends/opencv-dnn/test/OCVDNNInferTest"
            ;;
        "LIBTORCH")
            test_executable="backends/libtorch/test/LibtorchInferTest"
            ;;
        "ONNX_RUNTIME")
            test_executable="backends/onnx-runtime/test/ONNXRuntimeInferTest"
            ;;
        "LIBTENSORFLOW")
            test_executable="backends/libtensorflow/test/TensorFlowInferTest"
            ;;
        "TENSORRT")
            test_executable="backends/tensorrt/test/TensorRTInferTest"
            ;;
        "OPENVINO")
            test_executable="backends/openvino/test/OpenVINOInferTest"
            ;;
    esac
    
    if [ -f "$test_executable" ]; then
        log_info "Running test executable: $test_executable"
        if ./"$test_executable" --gtest_output=xml:"$results_file"; then
            log_success "Tests passed for $backend"
            return 0
        else
            log_error "Tests failed for $backend"
            return 1
        fi
    else
        log_warning "Test executable not found for $backend: $test_executable"
        log_info "Checking if tests are built with CTest..."
        
        # Try running with CTest
        if ctest --output-on-failure -R "${backend}Test" --verbose; then
            log_success "CTest passed for $backend"
            return 0
        else
            log_warning "No tests found or tests failed for $backend"
            return 1
        fi
    fi
}

# Function to generate test summary
generate_test_summary() {
    local results_file="$TEST_RESULTS_DIR/summary.txt"
    
    log_info "Generating test summary..."
    
    echo "InferenceEngines Backend Test Summary" > "$results_file"
    echo "Generated on: $(date)" >> "$results_file"
    echo "=========================================" >> "$results_file"
    echo "" >> "$results_file"
    
    for backend in "${BACKENDS[@]}"; do
        local backend_dir=$(get_backend_dir "$backend")
        local result_file="$TEST_RESULTS_DIR/${backend_dir}_results.xml"
        local status="NOT_TESTED"
        
        if [ -f "$result_file" ]; then
            if grep -q 'failures="0"' "$result_file" && grep -q 'errors="0"' "$result_file"; then
                status="PASSED"
            else
                status="FAILED"
            fi
        fi
        
        printf "%-15s: %s\n" "$backend" "$status" >> "$results_file"
    done
    
    echo "" >> "$results_file"
    echo "Detailed results available in $TEST_RESULTS_DIR" >> "$results_file"
    
    cat "$results_file"
    log_success "Test summary generated: $results_file"
}

# Function to clean previous builds
clean_builds() {
    log_info "Cleaning previous builds..."
    if [ -d "$BUILD_DIR" ]; then
        rm -rf "$BUILD_DIR"
    fi
    if [ -d "$TEST_RESULTS_DIR" ]; then
        rm -rf "$TEST_RESULTS_DIR"
    fi
    mkdir -p "$BUILD_DIR"
    mkdir -p "$TEST_RESULTS_DIR"
    log_success "Clean completed"
}

# Function to check dependency versions against centralized versions
check_dependency_versions() {
    log_info "Verifying dependency versions from centralized management..."
    
    # Parse versions from cmake/versions.cmake
    if [ -f "$PROJECT_ROOT/cmake/versions.cmake" ]; then
        log_info "Reading version information from cmake/versions.cmake"
        
        # Extract versions using grep and sed (strip quotes and whitespace)
        ONNX_RUNTIME_VERSION=$(grep "ONNX_RUNTIME_VERSION" "$PROJECT_ROOT/cmake/versions.cmake" | grep -o '"[^"]*"' | sed 's/"//g')
        TENSORRT_VERSION=$(grep "TENSORRT_VERSION" "$PROJECT_ROOT/cmake/versions.cmake" | grep -o '"[^"]*"' | sed 's/"//g')
        LIBTORCH_VERSION=$(grep "LIBTORCH_VERSION" "$PROJECT_ROOT/cmake/versions.cmake" | grep -o '"[^"]*"' | sed 's/"//g')
        OPENVINO_VERSION=$(grep "OPENVINO_VERSION" "$PROJECT_ROOT/cmake/versions.cmake" | grep -o '"[^"]*"' | sed 's/"//g')
        TENSORFLOW_VERSION=$(grep "TENSORFLOW_VERSION" "$PROJECT_ROOT/cmake/versions.cmake" | grep -o '"[^"]*"' | sed 's/"//g')
        OPENCV_MIN_VERSION=$(grep "OPENCV_MIN_VERSION" "$PROJECT_ROOT/cmake/versions.cmake" | grep -o '"[^"]*"' | sed 's/"//g')
        CUDA_VERSION=$(grep "CUDA_VERSION" "$PROJECT_ROOT/cmake/versions.cmake" | grep -o '"[^"]*"' | sed 's/"//g')
        
        log_info "Expected versions from centralized management:"
        log_info "  OpenCV: >= $OPENCV_MIN_VERSION"
        log_info "  ONNX Runtime: $ONNX_RUNTIME_VERSION"
        log_info "  TensorRT: $TENSORRT_VERSION"
        log_info "  LibTorch: $LIBTORCH_VERSION"
        log_info "  OpenVINO: $OPENVINO_VERSION"
        log_info "  TensorFlow: $TENSORFLOW_VERSION"
        log_info "  CUDA: $CUDA_VERSION"
        
        # Check installed versions
        log_info "Checking installed versions..."
        
        # OpenCV
        if pkg-config --exists opencv4; then
            OPENCV_VERSION=$(pkg-config --modversion opencv4)
            if [[ "$(printf '%s\n' "$OPENCV_MIN_VERSION" "$OPENCV_VERSION" | sort -V | head -n1)" = "$OPENCV_MIN_VERSION" ]]; then
                log_success "OpenCV version $OPENCV_VERSION satisfies minimum requirement $OPENCV_MIN_VERSION"
            else
                log_warning "OpenCV version $OPENCV_VERSION is older than required $OPENCV_MIN_VERSION"
            fi
        else
            log_warning "OpenCV not found, skipping version check"
        fi
        
        # CUDA
        if command -v nvcc >/dev/null; then
            INSTALLED_CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $6}' | cut -c2-)
            if [[ "$INSTALLED_CUDA_VERSION" == "$CUDA_VERSION"* ]]; then
                log_success "CUDA version $INSTALLED_CUDA_VERSION matches required $CUDA_VERSION"
            else
                log_warning "CUDA version mismatch: Found $INSTALLED_CUDA_VERSION, expected $CUDA_VERSION"
            fi
        else
            log_warning "CUDA not found, skipping version check"
        fi
        
        # Return success
        return 0
    else
        log_error "Cannot find centralized version management file: cmake/versions.cmake"
        return 1
    fi
}

# Function to install missing test dependencies
install_test_dependencies() {
    log_info "Checking and installing missing test dependencies..."
    
    # Install basic dependencies for testing
    if command -v apt-get >/dev/null 2>&1; then
        # Debian/Ubuntu
        log_info "Debian/Ubuntu system detected"
        log_info "Installing test dependencies via apt..."
        
        # Check for sudo access
        if command -v sudo >/dev/null 2>&1; then
            sudo apt-get update
            sudo apt-get install -y cmake build-essential libgtest-dev libgmock-dev libgoogle-glog-dev libopencv-dev
            
            # Build and install GTest if needed
            if [ ! -f "/usr/lib/libgtest.a" ] && [ -d "/usr/src/googletest" ]; then
                log_info "Building GTest from source..."
                cd /usr/src/googletest
                sudo cmake .
                sudo make
                sudo cp lib/libgtest*.a /usr/lib/
            fi
        else
            log_warning "Cannot install system dependencies: sudo not available"
            log_info "Proceeding with mock testing only"
        fi
    elif command -v yum >/dev/null 2>&1; then
        # CentOS/RHEL/Fedora
        log_info "CentOS/RHEL/Fedora system detected"
        log_info "Installing test dependencies via yum..."
        
        if command -v sudo >/dev/null 2>&1; then
            sudo yum install -y cmake gcc-c++ gtest-devel gmock-devel glog-devel opencv-devel
        else
            log_warning "Cannot install system dependencies: sudo not available"
            log_info "Proceeding with mock testing only"
        fi
    else
        log_warning "Unknown package manager, cannot install system dependencies"
        log_info "Proceeding with mock testing only"
    fi
    
    # Verify basic test dependencies
    local missing_deps=0
    
    if ! command -v cmake >/dev/null 2>&1; then
        log_error "CMake not found - required for building tests"
        missing_deps=$((missing_deps + 1))
    fi
    
    if ! pkg-config --exists opencv4; then
        log_warning "OpenCV not found - will use mock for OpenCV tests"
    fi
    
    if [ $missing_deps -gt 0 ]; then
        log_warning "$missing_deps dependencies missing for proper testing"
        log_info "Tests will still run using mock approach where possible"
    else
        log_success "Basic test dependencies are available"
    fi
}

# Main function
main() {
    log_info "Starting InferenceEngines Backend Testing"
    log_info "Project root: $PROJECT_ROOT"
    
    # Parse command line arguments
    local specific_backend=""
    local clean_first=false
    local skip_build=false
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --backend)
                specific_backend="$2"
                shift 2
                ;;
            --clean)
                clean_first=true
                shift
                ;;
            --skip-build)
                skip_build=true
                shift
                ;;
            --help)
                echo "Usage: $0 [OPTIONS]"
                echo "Options:"
                echo "  --backend BACKEND   Test only specific backend"
                echo "  --clean            Clean builds before testing"
                echo "  --skip-build       Skip build step, only run tests"
                echo "  --help             Show this help message"
                echo ""
                echo "Available backends: ${BACKENDS[*]}"
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                exit 1
                ;;
        esac
    done
    
    # Check dependency versions against centralized management
    check_dependency_versions
    
    # Install missing test dependencies if needed
    if [ "$specific_backend" = "" ] || [ "$specific_backend" = "ALL" ]; then
        install_test_dependencies
    fi
    
    # Clean if requested
    if [ "$clean_first" = true ]; then
        clean_builds
    fi
    
    # Check dependency versions
    if ! check_dependency_versions; then
        log_error "Dependency version check failed"
        exit 1
    fi
    
    # Determine which backends to test
    local backends_to_test=()
    if [ -n "$specific_backend" ]; then
        backends_to_test=("$specific_backend")
    else
        backends_to_test=("${BACKENDS[@]}")
    fi
    
    # Test each backend
    local failed_backends=()
    local successful_backends=()
    
    for backend in "${backends_to_test[@]}"; do
        log_info "========================================="
        log_info "Testing backend: $backend"
        log_info "========================================="
        
        # Check availability
        if ! check_backend_availability "$backend"; then
            log_warning "Skipping $backend (dependencies not available)"
            continue
        fi
        
        # Build backend (unless skipping)
        if [ "$skip_build" = false ]; then
            if ! build_backend "$backend"; then
                failed_backends+=("$backend")
                continue
            fi
        fi
        
        # Setup model for this backend
        local backend_dir=$(get_backend_dir "$backend")
        local test_dir="$PROJECT_ROOT/backends/$backend_dir/test"
        setup_backend_model "$backend" "$test_dir"
        
        # Run tests
        if run_backend_tests "$backend"; then
            successful_backends+=("$backend")
        else
            failed_backends+=("$backend")
        fi
        
        # Cleanup temporary models
        cleanup_backend_model "$test_dir"
        
        echo ""
    done
    
    # Generate summary
    generate_test_summary
    
    # Print final results
    log_info "========================================="
    log_info "FINAL RESULTS"
    log_info "========================================="
    
    if [ ${#successful_backends[@]} -gt 0 ]; then
        log_success "Successful backends: ${successful_backends[*]}"
    fi
    
    if [ ${#failed_backends[@]} -gt 0 ]; then
        log_error "Failed backends: ${failed_backends[*]}"
        exit 1
    else
        log_success "All tested backends passed!"
        exit 0
    fi
}

# Run main function with all arguments
main "$@"
