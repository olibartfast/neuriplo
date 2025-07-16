#!/bin/bash

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Validate scripts
validate_scripts() {
    local scripts=("test_backends.sh" "setup_test_models.sh" "run_complete_tests.sh" "model_downloader.py")
    local all_valid=true

    for script in "${scripts[@]}"; do
        local script_path="$PROJECT_ROOT/scripts/$script"
        if [[ -f "$script_path" ]]; then
            [[ -x "$script_path" || "$script" == *.py ]] || chmod +x "$script_path"
        else
            echo "Error: $script not found"
            all_valid=false
        fi
    done
    return $all_valid
}

# Validate test files
validate_test_files() {
    local backends=("opencv-dnn" "libtorch" "onnx-runtime" "libtensorflow" "tensorrt" "openvino")
    local all_valid=true

    for backend in "${backends[@]}"; do
        local test_dir="$PROJECT_ROOT/backends/$backend/test"
        if [[ -d "$test_dir" ]]; then
            [[ -n $(find "$test_dir" -name "*Test.cpp") ]] || echo "Warning: $backend: No C++ test file"
            [[ -f "$test_dir/CMakeLists.txt" ]] || echo "Warning: $backend: CMakeLists.txt missing"
        else
            echo "Error: $backend: Test directory not found"
            all_valid=false
        fi
    done
    return $all_valid
}

# Test model downloader
test_model_downloader() {
    local temp_dir=$(mktemp -d)
    if python3 "$PROJECT_ROOT/scripts/model_downloader.py" OPENCV_DNN --output-dir "$temp_dir" >/dev/null 2>&1 && [[ -f "$temp_dir/resnet18.onnx" ]]; then
        rm -rf "$temp_dir"
        return 0
    else
        echo "Error: Model downloader failed"
        rm -rf "$temp_dir"
        return 1
    fi
}

# Test backend availability
test_backend_availability() {
    pkg-config --exists opencv4 || pkg-config --exists opencv || echo "Warning: OpenCV not found"
    for pkg in torch tensorflow onnxruntime; do
        python3 -c "import $pkg" >/dev/null 2>&1 || echo "Warning: Python package $pkg not found"
    done
    for tool in cmake make g++; do
        command -v "$tool" >/dev/null 2>&1 || echo "Warning: $tool not found"
    done
}

# Test build system
test_build_system() {
    local test_dir=$(mktemp -d)
    cd "$test_dir"
    cmake "$PROJECT_ROOT" -DDEFAULT_BACKEND=OPENCV_DNN >/dev/null 2>&1 && { cd "$PROJECT_ROOT"; rm -rf "$test_dir"; return 0; }
    echo "Error: CMake configuration failed"
    cd "$PROJECT_ROOT"
    rm -rf "$test_dir"
    return 1
}

# Run quick test
run_quick_test() {
    local test_script="$PROJECT_ROOT/scripts/test_backends.sh"
    [[ -x "$test_script" && $("$test_script" --help >/dev/null 2>&1) ]] && return 0
    echo "Error: Backend test script failed"
    return 1
}

# Generate report
generate_validation_report() {
    local report_file="$PROJECT_ROOT/test_validation_report.txt"
    {
        echo "Test Validation Report - $(date)"
        echo ""
        echo "Test Files Status:"
        for backend in opencv-dnn libtorch onnx-runtime libtensorflow tensorrt openvino; do
            local test_dir="$PROJECT_ROOT/backends/$backend/test"
            if [[ -d "$test_dir" ]]; then
                echo "  $backend: Test directory exists"
                ls "$test_dir"/*Test.cpp >/dev/null 2>&1 && echo "    - C++ test file: ✓" || echo "    - C++ test file: ✗"
                [[ -f "$test_dir/CMakeLists.txt" ]] && echo "    - CMakeLists.txt: ✓" || echo "    - CMakeLists.txt: ✗"
            else
                echo "  $backend: Test directory missing"
            fi
        done
    } > "$report_file"
    echo "Report generated: $report_file"
}

# Main
cd "$PROJECT_ROOT"
echo "Starting test system validation..."

all_passed=true
validate_scripts || all_passed=false
validate_test_files || all_passed=false
test_model_downloader || all_passed=false
test_backend_availability
test_build_system || all_passed=false
run_quick_test || all_passed=false
generate_validation_report

if [[ "$all_passed" == "true" ]]; then
    echo "✓ All validation checks passed!"
else
    echo "⚠ Some validation checks failed. Check report: $report_file"
fi