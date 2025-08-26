#!/bin/bash

# Enhanced Backend Testing Script for neuriplo
# This script tests each backend individually, builds them, and runs unit tests
# Author: Generated for neuriplo project
# Date: $(date)

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Project root directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BUILD_DIR="$PROJECT_ROOT/build"
TEST_RESULTS_DIR="$PROJECT_ROOT/test_results"

# Supported backends
BACKENDS=("OPENCV_DNN" "ONNX_RUNTIME" "LIBTORCH" "LIBTENSORFLOW" "TENSORRT" "OPENVINO" "GGML")

# Test configuration
PARALLEL_JOBS=4
PERFORMANCE_BENCHMARK_ITERATIONS=1000
MEMORY_LEAK_ITERATIONS=5000
STRESS_TEST_THREADS=8
STRESS_TEST_ITERATIONS=500

# Load versions from versions.env
if [ -f "$PROJECT_ROOT/versions.env" ]; then
    source "$PROJECT_ROOT/versions.env"
    # Map versions to expected variable names
    OPENCV_MIN_VERSION="$OPENCV_VERSION"
    LIBTORCH_VERSION="$PYTORCH_VERSION"
else
    echo "Error: versions.env file not found"
    exit 1
fi

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

log_performance() {
    echo -e "${PURPLE}[PERFORMANCE]${NC} $1"
}

log_memory() {
    echo -e "${CYAN}[MEMORY]${NC} $1"
}

# Function to check if a backend is available
check_backend_availability() {
    local backend=$1
    log_info "Checking availability of backend: $backend"
    
    case $backend in
        "OPENCV_DNN")
            if pkg-config --exists opencv4; then
                local opencv_version=$(pkg-config --modversion opencv4)
                echo "DEBUG: opencv_version='$opencv_version' OPENCV_MIN_VERSION='$OPENCV_MIN_VERSION'"
                if [[ "$opencv_version" == "$OPENCV_MIN_VERSION" ]] || [[ "$(printf '%s\n' "$OPENCV_MIN_VERSION" "$opencv_version" | sort -V | head -n1 | tr -d '\n')" = "$OPENCV_MIN_VERSION" ]]; then
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
            local deps_onnx_runtime_dir="${HOME}/dependencies/onnxruntime-linux-x64-gpu-$ONNX_RUNTIME_VERSION"
            local expected_version="$ONNX_RUNTIME_VERSION"
            
            if [ -d "$onnx_runtime_dir" ] || [ -d "$alt_onnx_runtime_dir" ] || [ -d "$deps_onnx_runtime_dir" ]; then
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
            local deps_libtorch_dir="${HOME}/dependencies/libtorch-2.0.1"
            local expected_version="$LIBTORCH_VERSION"
            
            if [ -d "$libtorch_dir" ] || [ -d "$alt_libtorch_dir" ] || [ -d "$deps_libtorch_dir" ]; then
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
            local tensorflow_dir="${HOME}/dependencies/tensorflow"
            
            # Check for custom TensorFlow installation
            if [ -d "$tensorflow_dir" ]; then
                log_success "TensorFlow found in dependencies directory"
                return 0
            elif pkg-config --exists tensorflow; then
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
            local deps_tensorrt_dir="${HOME}/dependencies/TensorRT-$TENSORRT_VERSION"
            local expected_version="$TENSORRT_VERSION"
            
            if [ -d "${HOME}/TensorRT-${expected_version}" ]; then
                log_success "TensorRT ${expected_version} found (exact match)"
                return 0
            elif [ -d "$deps_tensorrt_dir" ]; then
                log_success "TensorRT found in dependencies directory"
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
        "GGML")
            local ggml_dir="${HOME}/dependencies/ggml"
            local alt_ggml_dir="/usr/local/ggml"
            local legacy_ggml_dir="${HOME}/.local/ggml"
            
            if [ -d "$ggml_dir" ] && [ -f "$ggml_dir/lib/libggml.so" ] && [ -f "$ggml_dir/include/ggml.h" ]; then
                log_success "GGML found in dependencies directory"
                return 0
            elif [ -d "$alt_ggml_dir" ] && [ -f "$alt_ggml_dir/lib/libggml.so" ]; then
                log_success "GGML found in system directory"
                return 0
            elif [ -d "$legacy_ggml_dir" ] && [ -f "$legacy_ggml_dir/lib/libggml.so" ]; then
                log_success "GGML found in legacy user directory"
                return 0
            else
                log_warning "GGML not found"
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
        "GGML") echo "ggml" ;;
        *) echo "unknown" ;;
    esac
}

# Function to get test executable name
get_test_executable_name() {
    local backend=$1
    case $backend in
        "OPENCV_DNN") echo "OCVDNNInferTest" ;;
        "ONNX_RUNTIME") echo "ONNXRuntimeInferTest" ;;
        "LIBTORCH") echo "LibtorchInferTest" ;;
        "LIBTENSORFLOW") echo "TensorFlowInferTest" ;;
        "TENSORRT") echo "TensorRTInferTest" ;;
        "OPENVINO") echo "OpenVINOInferTest" ;;
        "GGML") echo "GGMLInferTest" ;;
        *) echo "UnknownInferTest" ;;
    esac
}

# Function to run performance benchmark
run_performance_benchmark() {
    local backend=$1
    local backend_dir=$(get_backend_dir $backend)
    local test_executable_name=$(get_test_executable_name $backend)
    local test_executable="${BUILD_DIR}/backends/${backend_dir}/test/${test_executable_name}"
    
    if [ -f "$test_executable" ]; then
        log_performance "Running performance benchmark for $backend..."
        
        # Run benchmark with performance metrics
        local start_time=$(date +%s.%N)
        "$test_executable" --gtest_filter="*Performance*" --benchmark_iterations=$PERFORMANCE_BENCHMARK_ITERATIONS > "${TEST_RESULTS_DIR}/${backend_dir}_performance.log" 2>&1
        local end_time=$(date +%s.%N)
        
        local benchmark_time=$(echo "$end_time - $start_time" | bc -l)
        log_performance "Performance benchmark completed in ${benchmark_time}s"
        
        # Extract performance metrics
        if [ -f "${TEST_RESULTS_DIR}/${backend_dir}_performance.log" ]; then
            local avg_time=$(grep "Average inference time" "${TEST_RESULTS_DIR}/${backend_dir}_performance.log" | awk '{print $NF}' | head -1)
            local throughput=$(grep "Throughput" "${TEST_RESULTS_DIR}/${backend_dir}_performance.log" | awk '{print $NF}' | head -1)
            local memory_usage=$(grep "Memory usage" "${TEST_RESULTS_DIR}/${backend_dir}_performance.log" | awk '{print $NF}' | head -1)
            
            if [ ! -z "$avg_time" ]; then
                log_performance "$backend - Avg Time: ${avg_time}ms, Throughput: ${throughput}fps, Memory: ${memory_usage}MB"
            fi
        fi
    else
        log_warning "Performance benchmark executable not found for $backend"
    fi
}

# Function to run memory leak detection
run_memory_leak_detection() {
    local backend=$1
    local backend_dir=$(get_backend_dir $backend)
    local test_executable_name=$(get_test_executable_name $backend)
    local test_executable="${BUILD_DIR}/backends/${backend_dir}/test/${test_executable_name}"
    
    if [ -f "$test_executable" ]; then
        log_memory "Running memory leak detection for $backend..."
        
        # Run memory leak test
        "$test_executable" --gtest_filter="*MemoryLeak*" --memory_leak_iterations=$MEMORY_LEAK_ITERATIONS > "${TEST_RESULTS_DIR}/${backend_dir}_memory.log" 2>&1
        
        # Check for memory leaks
        if grep -q "memory leak detected" "${TEST_RESULTS_DIR}/${backend_dir}_memory.log"; then
            log_error "Memory leak detected in $backend"
            return 1
        else
            log_success "No memory leaks detected in $backend"
            return 0
        fi
    else
        log_warning "Memory leak detection executable not found for $backend"
        return 0
    fi
}

# Function to run stress test
run_stress_test() {
    local backend=$1
    local backend_dir=$(get_backend_dir $backend)
    local test_executable_name=$(get_test_executable_name $backend)
    local test_executable="${BUILD_DIR}/backends/${backend_dir}/test/${test_executable_name}"
    
    if [ -f "$test_executable" ]; then
        log_info "Running stress test for $backend..."
        
        # Run stress test
        "$test_executable" --gtest_filter="*Stress*" --stress_threads=$STRESS_TEST_THREADS --stress_iterations=$STRESS_TEST_ITERATIONS > "${TEST_RESULTS_DIR}/${backend_dir}_stress.log" 2>&1
        
        # Check for stress test failures
        if grep -q "FAILED" "${TEST_RESULTS_DIR}/${backend_dir}_stress.log"; then
            log_error "Stress test failed for $backend"
            return 1
        else
            log_success "Stress test passed for $backend"
            return 0
        fi
    else
        log_warning "Stress test executable not found for $backend"
        return 0
    fi
}

# Function to test a single backend
test_backend() {
    local backend=$1
    local backend_dir=$(get_backend_dir $backend)
    
    log_info "========================================="
    log_info "Testing backend: $backend"
    log_info "========================================="
    
    # Check backend availability
    if ! check_backend_availability "$backend"; then
        log_warning "Skipping $backend (dependencies not available)"
        return 1
    fi
    
    # Build the backend if not skipping build
    if [ "$SKIP_BUILD" != "true" ]; then
        log_info "Building $backend..."
        cd "$BUILD_DIR"
        
        # Configure CMake with specific backend
        if [ "$backend" = "ONNX_RUNTIME" ]; then
            cmake -DDEFAULT_BACKEND="$backend" -DBUILD_INFERENCE_ENGINE_TESTS=ON -DONNX_RUNTIME_DIR="$HOME/dependencies/onnxruntime-linux-x64-gpu-$ONNX_RUNTIME_VERSION" .. > "${TEST_RESULTS_DIR}/${backend_dir}_build.log" 2>&1
        elif [ "$backend" = "LIBTORCH" ]; then
            cmake -DDEFAULT_BACKEND="$backend" -DBUILD_INFERENCE_ENGINE_TESTS=ON -DTorch_DIR="$HOME/dependencies/libtorch-2.0.1/share/cmake/Torch" .. > "${TEST_RESULTS_DIR}/${backend_dir}_build.log" 2>&1
        elif [ "$backend" = "TENSORRT" ]; then
            cmake -DDEFAULT_BACKEND="$backend" -DBUILD_INFERENCE_ENGINE_TESTS=ON -DTENSORRT_DIR="$HOME/dependencies/TensorRT-$TENSORRT_VERSION" .. > "${TEST_RESULTS_DIR}/${backend_dir}_build.log" 2>&1
        elif [ "$backend" = "GGML" ]; then
            cmake -DDEFAULT_BACKEND="$backend" -DBUILD_INFERENCE_ENGINE_TESTS=ON -DGGML_DIR="$HOME/dependencies/ggml" .. > "${TEST_RESULTS_DIR}/${backend_dir}_build.log" 2>&1
        else
            cmake -DDEFAULT_BACKEND="$backend" -DBUILD_INFERENCE_ENGINE_TESTS=ON .. > "${TEST_RESULTS_DIR}/${backend_dir}_build.log" 2>&1
        fi
        
        # Build the project using available build system
        if command -v ninja >/dev/null 2>&1 && [ -f "build.ninja" ]; then
            ninja >> "${TEST_RESULTS_DIR}/${backend_dir}_build.log" 2>&1
        else
            make -j$PARALLEL_JOBS >> "${TEST_RESULTS_DIR}/${backend_dir}_build.log" 2>&1
        fi
        
        if [ $? -ne 0 ]; then
            log_error "Build failed for $backend"
            return 1
        fi
        
        log_success "Build completed for $backend"
    fi
    
    # Setup TensorFlow model if needed
    if [ "$backend" = "LIBTENSORFLOW" ]; then
        log_info "Setting up TensorFlow model for testing..."
        cd "$BUILD_DIR"
        
        # Check if model already exists
        if [ ! -d "saved_model" ]; then
            # Create temporary Python environment
            python3 -m venv /tmp/tf_test_env
            source /tmp/tf_test_env/bin/activate
            
            # Install TensorFlow
            pip install --upgrade pip > /dev/null 2>&1
            pip install tensorflow tensorflow-hub > /dev/null 2>&1
            
            # Generate SavedModel
            cat > generate_saved_model.py << 'EOF'
#!/usr/bin/env python3
import tensorflow as tf
from tensorflow import keras
import os

def generate_saved_model():
    """Generate a TensorFlow SavedModel using ResNet50 from Keras Applications."""
    print("Loading ResNet50 model from Keras Applications...")
    model = keras.applications.ResNet50(weights='imagenet')
    saved_model_path = 'saved_model'
    print(f"Exporting model to {saved_model_path}...")
    model.export(saved_model_path)
    print(f"SavedModel successfully created at: {saved_model_path}")

if __name__ == "__main__":
    generate_saved_model()
EOF
            
            python3 generate_saved_model.py > "${TEST_RESULTS_DIR}/${backend_dir}_model_generation.log" 2>&1
            
            # Cleanup
            rm -f generate_saved_model.py
            deactivate
            rm -rf /tmp/tf_test_env
            
            if [ ! -d "saved_model" ]; then
                log_error "Failed to generate TensorFlow model"
                return 1
            fi
        fi
        
        # Create model_path.txt for the test
        echo "$BUILD_DIR/saved_model" > "$BUILD_DIR/model_path.txt"
        log_success "TensorFlow model setup completed"
    fi
    
    # Setup GGML model if needed
    if [ "$backend" = "GGML" ]; then
        log_info "Setting up GGML model for testing..."
        cd "$BUILD_DIR"
        
        # Check if model already exists
        if [ ! -f "resnet18.ggml" ]; then
            # Check if conversion script exists
            local conversion_script="$PROJECT_ROOT/scripts/convert_to_ggml.sh"
            if [ -f "$conversion_script" ]; then
                log_info "Converting ResNet18 to GGML format..."
                
                # Create temporary Python environment for conversion
                python3 -m venv /tmp/ggml_test_env
                source /tmp/ggml_test_env/bin/activate
                
                # Install PyTorch dependencies
                pip install --upgrade pip > /dev/null 2>&1
                pip install torch torchvision numpy > /dev/null 2>&1
                
                # Run conversion
                python3 "$PROJECT_ROOT/scripts/convert_resnet18_to_ggml.py" --output "resnet18.ggml" --test-dir "." > "${TEST_RESULTS_DIR}/${backend_dir}_model_generation.log" 2>&1
                
                # Cleanup
                deactivate
                rm -rf /tmp/ggml_test_env
                
                if [ ! -f "resnet18.ggml" ]; then
                    log_error "Failed to generate GGML model"
                    return 1
                fi
            else
                log_warning "GGML conversion script not found, creating placeholder model"
                # Create a simple placeholder model file
                echo "GGML" > "resnet18.ggml"
                echo "1" >> "resnet18.ggml"
                echo "1000" >> "resnet18.ggml"
                echo "150528" >> "resnet18.ggml"
            fi
        fi
        
        # Create model_path.txt for the test
        echo "$BUILD_DIR/resnet18.ggml" > "$BUILD_DIR/model_path.txt"
        log_success "GGML model setup completed"
    fi
    
    # Run tests
    local test_executable_name=$(get_test_executable_name $backend)
    local test_executable="${BUILD_DIR}/backends/${backend_dir}/test/${test_executable_name}"
    
    if [ -f "$test_executable" ]; then
        log_info "Running tests for $backend..."
        
        # Run basic tests
        "$test_executable" --gtest_output=xml:"${TEST_RESULTS_DIR}/${backend_dir}_results.xml" > "${TEST_RESULTS_DIR}/${backend_dir}_test.log" 2>&1
        local test_result=$?
        
        if [ $test_result -eq 0 ]; then
            log_success "Tests passed for $backend"
            
            # Run additional test types if available
            run_performance_benchmark "$backend"
            run_memory_leak_detection "$backend"
            run_stress_test "$backend"
            
            return 0
        else
            log_error "Tests failed for $backend"
            return 1
        fi
    else
        log_error "Test executable not found for $backend"
        return 1
    fi
}

# Function to generate test summary
generate_test_summary() {
    local results_file="$TEST_RESULTS_DIR/summary.txt"
    
    log_info "Generating test summary..."
    
    echo "neuriplo Backend Test Summary" > "$results_file"
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

# Function to run tests in parallel
run_parallel_tests() {
    local backends_to_test=("$@")
    local pids=()
    local results=()
    
    log_info "Running tests in parallel with $PARALLEL_JOBS jobs..."
    
    # Start tests in parallel
    for backend in "${backends_to_test[@]}"; do
        # Wait if we've reached the parallel limit
        while [ ${#pids[@]} -ge $PARALLEL_JOBS ]; do
            for i in "${!pids[@]}"; do
                if ! kill -0 "${pids[$i]}" 2>/dev/null; then
                    wait "${pids[$i]}"
                    results[$i]=$?
                    unset pids[$i]
                    break
                fi
            done
            sleep 0.1
        done
        
        # Start new test
        test_backend "$backend" &
        pids+=($!)
        log_info "Started test for $backend (PID: $!)"
    done
    
    # Wait for all remaining tests
    for i in "${!pids[@]}"; do
        wait "${pids[$i]}"
        results[$i]=$?
    done
    
    # Check results
    local all_passed=true
    for i in "${!results[@]}"; do
        if [ ${results[$i]} -ne 0 ]; then
            all_passed=false
            break
        fi
    done
    
    return $([ "$all_passed" = true ] && echo 0 || echo 1)
}

# Main execution
main() {
    log_info "Starting Enhanced neuriplo Backend Testing"
    log_info "Project root: $PROJECT_ROOT"
    
    # Parse command line arguments
    local specific_backend=""
    local clean_build=false
    local skip_build=false
    local parallel_mode=false
    local performance_only=false
    local memory_only=false
    local stress_only=false
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --backend)
                specific_backend="$2"
                shift 2
                ;;
            --clean)
                clean_build=true
                shift
                ;;
            --skip-build)
                skip_build=true
                shift
                ;;
            --parallel)
                parallel_mode=true
                shift
                ;;
            --performance-only)
                performance_only=true
                shift
                ;;
            --memory-only)
                memory_only=true
                shift
                ;;
            --stress-only)
                stress_only=true
                shift
                ;;
            --help)
                show_help
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done
    
    # Create test results directory
    mkdir -p "$TEST_RESULTS_DIR"
    
    # Clean build if requested
    if [ "$clean_build" = true ]; then
        log_info "Cleaning build directory..."
        rm -rf "$BUILD_DIR"
    fi
    
    # Create build directory if it doesn't exist
    mkdir -p "$BUILD_DIR"
    
    # Determine which backends to test
    local backends_to_test=()
    if [ -n "$specific_backend" ]; then
        if [[ " ${BACKENDS[@]} " =~ " ${specific_backend} " ]]; then
            backends_to_test=("$specific_backend")
        else
            log_error "Unknown backend: $specific_backend"
            log_info "Available backends: ${BACKENDS[*]}"
            exit 1
        fi
    else
        backends_to_test=("${BACKENDS[@]}")
    fi
    
    # Run tests
    local test_result=0
    if [ "$parallel_mode" = true ] && [ ${#backends_to_test[@]} -gt 1 ]; then
        run_parallel_tests "${backends_to_test[@]}"
        test_result=$?
    else
        for backend in "${backends_to_test[@]}"; do
            if ! test_backend "$backend"; then
                test_result=1
            fi
        done
    fi
    
    # Generate test summary
    generate_test_summary
    
    # Final results
    log_info "========================================="
    log_info "FINAL RESULTS"
    log_info "========================================="
    
    if [ $test_result -eq 0 ]; then
        log_success "All tested backends passed!"
    else
        log_error "Some backends failed tests"
    fi
    
    exit $test_result
}

# Function to show help
show_help() {
    echo "Usage: $0 [OPTIONS]"
    echo "Options:"
    echo "  --backend BACKEND   Test only specific backend"
    echo "  --clean            Clean builds before testing"
    echo "  --skip-build       Skip build step, only run tests"
    echo "  --parallel         Run tests in parallel"
    echo "  --performance-only Run only performance tests"
    echo "  --memory-only      Run only memory leak tests"
    echo "  --stress-only      Run only stress tests"
    echo "  --help             Show this help message"
    echo ""
    echo "Available backends: ${BACKENDS[*]}"
}

# Run main function
main "$@"
