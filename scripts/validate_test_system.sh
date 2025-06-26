#!/bin/bash

# Quick Test Validation Script
# This script runs a quick validation of the testing system

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Project root directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

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

print_header() {
    echo ""
    echo "=============================================="
    echo "  InferenceEngines Test Validation"
    echo "=============================================="
    echo ""
}

validate_scripts() {
    log_info "Validating test scripts..."
    
    local scripts=(
        "test_backends.sh"
        "setup_test_models.sh"
        "run_complete_tests.sh"
        "model_downloader.py"
    )
    
    local all_valid=true
    
    for script in "${scripts[@]}"; do
        local script_path="$PROJECT_ROOT/scripts/$script"
        if [ -f "$script_path" ]; then
            if [ -x "$script_path" ] || [[ "$script" == *.py ]]; then
                log_success "✓ $script exists and is executable"
            else
                log_warning "⚠ $script exists but is not executable"
                chmod +x "$script_path"
                log_info "  Made $script executable"
            fi
        else
            log_error "✗ $script not found"
            all_valid=false
        fi
    done
    
    return $all_valid
}

validate_test_files() {
    log_info "Validating test files..."
    
    local backends=("opencv-dnn" "libtorch" "onnx-runtime" "libtensorflow" "tensorrt" "openvino")
    local all_valid=true
    
    for backend in "${backends[@]}"; do
        local test_dir="$PROJECT_ROOT/backends/$backend/test"
        
        if [ -d "$test_dir" ]; then
            # Check for C++ test file
            local cpp_files=($(find "$test_dir" -name "*Test.cpp" 2>/dev/null))
            if [ ${#cpp_files[@]} -gt 0 ]; then
                log_success "✓ $backend: Test file found"
            else
                log_warning "⚠ $backend: No C++ test file found"
            fi
            
            # Check for CMakeLists.txt
            if [ -f "$test_dir/CMakeLists.txt" ]; then
                log_success "✓ $backend: CMakeLists.txt found"
            else
                log_warning "⚠ $backend: CMakeLists.txt missing"
            fi
        else
            log_error "✗ $backend: Test directory not found"
            all_valid=false
        fi
    done
    
    return $all_valid
}

test_model_downloader() {
    log_info "Testing model downloader..."
    
    local temp_dir=$(mktemp -d)
    local downloader="$PROJECT_ROOT/scripts/model_downloader.py"
    
    if python3 "$downloader" OPENCV_DNN --output-dir "$temp_dir" >/dev/null 2>&1; then
        if [ -f "$temp_dir/resnet18.onnx" ]; then
            log_success "✓ Model downloader created model file"
        else
            log_warning "⚠ Model downloader ran but no model file found"
        fi
    else
        log_error "✗ Model downloader failed"
        return 1
    fi
    
    # Cleanup
    rm -rf "$temp_dir"
    return 0
}

test_backend_availability() {
    log_info "Checking backend availability..."
    
    # Check OpenCV (most likely to be available)
    if pkg-config --exists opencv4 >/dev/null 2>&1; then
        log_success "✓ OpenCV found (opencv4)"
    elif pkg-config --exists opencv >/dev/null 2>&1; then
        log_success "✓ OpenCV found (opencv)"
    else
        log_warning "⚠ OpenCV not found via pkg-config"
    fi
    
    # Check Python packages
    local python_packages=("torch" "tensorflow" "onnxruntime")
    for package in "${python_packages[@]}"; do
        if python3 -c "import $package" >/dev/null 2>&1; then
            log_success "✓ Python package: $package"
        else
            log_warning "⚠ Python package not found: $package"
        fi
    done
    
    # Check system tools
    local tools=("cmake" "make" "g++")
    for tool in "${tools[@]}"; do
        if command -v "$tool" >/dev/null 2>&1; then
            log_success "✓ System tool: $tool"
        else
            log_warning "⚠ System tool not found: $tool"
        fi
    done
}

test_build_system() {
    log_info "Testing build system..."
    
    local test_dir=$(mktemp -d)
    cd "$test_dir"
    
    # Try a minimal CMake configuration
    if cmake "$PROJECT_ROOT" -DDEFAULT_BACKEND=OPENCV_DNN >/dev/null 2>&1; then
        log_success "✓ CMake configuration successful"
        return 0
    else
        log_error "✗ CMake configuration failed"
        return 1
    fi
    
    # Cleanup
    cd "$PROJECT_ROOT"
    rm -rf "$test_dir"
}

run_quick_test() {
    log_info "Running quick backend test..."
    
    # Try to run the test script with a quick check
    local test_script="$PROJECT_ROOT/scripts/test_backends.sh"
    
    if [ -x "$test_script" ]; then
        # Run with help to verify it works
        if "$test_script" --help >/dev/null 2>&1; then
            log_success "✓ Backend test script runs correctly"
        else
            log_error "✗ Backend test script has issues"
            return 1
        fi
    else
        log_error "✗ Backend test script not executable"
        return 1
    fi
    
    return 0
}

generate_validation_report() {
    local report_file="$PROJECT_ROOT/test_validation_report.txt"
    
    {
        echo "InferenceEngines Test Validation Report"
        echo "Generated on: $(date)"
        echo "========================================"
        echo ""
        
        echo "System Information:"
        echo "  OS: $(uname -s)"
        echo "  Architecture: $(uname -m)"
        echo "  Kernel: $(uname -r)"
        echo ""
        
        echo "Python Environment:"
        echo "  Python version: $(python3 --version 2>&1 || echo 'Not found')"
        echo "  Available packages:"
        for pkg in torch tensorflow onnxruntime opencv-python numpy; do
            if python3 -c "import $pkg; print(f'    $pkg: {$pkg.__version__}')" 2>/dev/null; then
                :
            else
                echo "    $pkg: Not available"
            fi
        done
        echo ""
        
        echo "Build Tools:"
        for tool in cmake make g++ ninja; do
            if command -v "$tool" >/dev/null 2>&1; then
                echo "  $tool: $(command -v $tool)"
            else
                echo "  $tool: Not found"
            fi
        done
        echo ""
        
        echo "Test Files Status:"
        for backend in opencv-dnn libtorch onnx-runtime libtensorflow tensorrt openvino; do
            test_dir="$PROJECT_ROOT/backends/$backend/test"
            if [ -d "$test_dir" ]; then
                echo "  $backend: Test directory exists"
                if ls "$test_dir"/*Test.cpp >/dev/null 2>&1; then
                    echo "    - C++ test file: ✓"
                else
                    echo "    - C++ test file: ✗"
                fi
                if [ -f "$test_dir/CMakeLists.txt" ]; then
                    echo "    - CMakeLists.txt: ✓"
                else
                    echo "    - CMakeLists.txt: ✗"
                fi
            else
                echo "  $backend: ✗ Test directory missing"
            fi
        done
        echo ""
        
        echo "Recommendations:"
        echo "  1. Install missing Python packages: pip install torch tensorflow onnxruntime"
        echo "  2. Install OpenCV development headers if missing"
        echo "  3. Ensure CMake version >= 3.10"
        echo "  4. Run './scripts/setup_test_models.sh' to prepare models"
        echo "  5. Use './scripts/test_backends.sh --backend OPENCV_DNN' for quick testing"
        echo ""
        
    } > "$report_file"
    
    log_success "Validation report generated: $report_file"
}

main() {
    print_header
    
    log_info "Starting InferenceEngines test system validation..."
    
    cd "$PROJECT_ROOT"
    
    local all_passed=true
    
    # Run validation steps
    if ! validate_scripts; then
        all_passed=false
    fi
    
    if ! validate_test_files; then
        all_passed=false
    fi
    
    if ! test_model_downloader; then
        all_passed=false
    fi
    
    test_backend_availability  # This is informational, not a failure
    
    if ! test_build_system; then
        all_passed=false
    fi
    
    if ! run_quick_test; then
        all_passed=false
    fi
    
    # Generate report
    generate_validation_report
    
    echo ""
    echo "=============================================="
    if [ "$all_passed" = true ]; then
        log_success "✓ All validation checks passed!"
        log_info "The test system is ready to use."
        echo ""
        log_info "Next steps:"
        log_info "  1. Run: ./scripts/setup_test_models.sh"
        log_info "  2. Run: ./scripts/test_backends.sh --backend OPENCV_DNN"
        log_info "  3. Run: ./scripts/run_complete_tests.sh"
    else
        log_warning "⚠ Some validation checks failed."
        log_info "Check the issues above and install missing dependencies."
        log_info "The system may still work for available backends."
    fi
    echo "=============================================="
    
    return $all_passed
}

main "$@"
