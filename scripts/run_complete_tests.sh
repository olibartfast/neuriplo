#!/bin/bash

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Define backends directly (was in backends.conf)
BACKENDS=("OPENCV_DNN" "ONNX_RUNTIME" "LIBTORCH" "LIBTENSORFLOW" "TENSORRT" "OPENVINO" "GGML" "TVM")

# Backend directory mapping
declare -A BACKEND_DIRS=(
    ["OPENCV_DNN"]="opencv-dnn"
    ["ONNX_RUNTIME"]="onnx-runtime" 
    ["LIBTORCH"]="libtorch"
    ["LIBTENSORFLOW"]="libtensorflow"
    ["TENSORRT"]="tensorrt"
    ["OPENVINO"]="openvino"
    ["GGML"]="ggml"
    ["TVM"]="tvm"
)

# Helper functions
get_backend_dir() {
    local backend=$1
    echo "${BACKEND_DIRS[$backend]:-unknown}"
}

get_backends_list() {
    echo "${BACKENDS[@]}"
}

# Parse arguments
backend=""
skip_models=false
clean_flag=false
quick_test=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --backend) backend="$2"; shift 2 ;;
        --skip-models) skip_models=true; shift ;;
        --clean) clean_flag=true; shift ;;
        --quick) quick_test=true; skip_models=true; shift ;;
        --help)
            echo "Usage: $0 [--backend BACKEND] [--skip-models] [--clean] [--quick] [--help]"
            echo "Backends: $(get_backends_list)"
            exit 0 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# Model setup
run_model_setup() {
    "$SCRIPT_DIR/setup_test_models.sh" && return 0 || return 1
}

# Backend tests
run_backend_tests() {
    local cmd="$SCRIPT_DIR/test_backends.sh"
    [[ -n "$1" ]] && cmd="$cmd --backend $1"
    [[ "$2" == "true" ]] && cmd="$cmd --clean"
    eval "$cmd" && return 0 || return 1
}

# Generate report
generate_final_report() {
    local results_dir="$PROJECT_ROOT/test_results"
    local report_file="$results_dir/final_report.txt"
    mkdir -p "$results_dir"

    {
        echo "neuriplo Test Report"
        echo "Generated: $(date)"
        echo ""
        echo "Test Results:"
        for backend in "${BACKENDS[@]}"; do
            local backend_dir=$(get_backend_dir "$backend")
            local result_file="$results_dir/${backend_dir}_results.xml"
            if [[ -f "$result_file" ]]; then
                local tests=$(grep -o 'tests="[0-9]*"' "$result_file" | cut -d'"' -f2)
                local failures=$(grep -o 'failures="[0-9]*"' "$result_file" | cut -d'"' -f2)
                printf "  %-15s: %s tests, %s failures\n" "$backend" "$tests" "$failures"
            else
                printf "  %-15s: No results\n" "$backend"
            fi
        done
        echo ""
        echo "Next Steps:"
        echo "  - Review logs in test_results/"
        echo "  - Install missing dependencies for failed backends"
    } > "$report_file"

    cat "$report_file"
    echo "Report generated: $report_file"
}

# Main workflow
cd "$PROJECT_ROOT"
echo "Starting neuriplo testing..."

if [[ "$quick_test" == "true" ]]; then
    echo "Running quick test"
fi

if [[ "$skip_models" == "false" ]] && ! run_model_setup; then
    echo "Model setup failed, continuing..."
fi

if ! run_backend_tests "$backend" "$clean_flag"; then
    echo "Some backend tests failed"
fi

generate_final_report
echo "Testing complete. Results in test_results/final_report.txt"