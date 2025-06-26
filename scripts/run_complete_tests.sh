#!/bin/bash

# Complete Backend Testing Workflow
# This script runs the complete testing workflow for InferenceEngines

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Project root directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

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

print_header() {
    echo ""
    echo "=============================================="
    echo "  InferenceEngines Complete Testing Suite"
    echo "=============================================="
    echo ""
}

print_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "This script runs the complete testing workflow:"
    echo "1. Sets up test models"
    echo "2. Tests all available backends"
    echo "3. Generates comprehensive report"
    echo ""
    echo "Options:"
    echo "  --backend BACKEND    Test only specific backend"
    echo "  --skip-models       Skip model setup"
    echo "  --clean             Clean builds before testing"
    echo "  --quick             Quick test (skip model generation)"
    echo "  --help              Show this help message"
    echo ""
    echo "Available backends: OPENCV_DNN ONNX_RUNTIME LIBTORCH LIBTENSORFLOW TENSORRT OPENVINO"
    echo ""
    echo "Examples:"
    echo "  $0                           # Full test suite"
    echo "  $0 --backend OPENCV_DNN      # Test only OpenCV DNN"
    echo "  $0 --quick                   # Quick test without model generation"
    echo "  $0 --clean                   # Clean build and full test"
    echo ""
}

run_model_setup() {
    log_info "Setting up test models..."
    if "$SCRIPT_DIR/setup_test_models.sh"; then
        log_success "Model setup completed"
        return 0
    else
        log_error "Model setup failed"
        return 1
    fi
}

run_backend_tests() {
    local backend="$1"
    local clean_flag="$2"
    
    log_info "Running backend tests..."
    
    local cmd="$SCRIPT_DIR/test_backends.sh"
    if [ -n "$backend" ]; then
        cmd="$cmd --backend $backend"
    fi
    if [ "$clean_flag" = "true" ]; then
        cmd="$cmd --clean"
    fi
    
    if eval "$cmd"; then
        log_success "Backend tests completed"
        return 0
    else
        log_error "Backend tests failed"
        return 1
    fi
}

generate_final_report() {
    local results_dir="$PROJECT_ROOT/test_results"
    local report_file="$results_dir/final_report.txt"
    
    log_info "Generating final test report..."
    
    mkdir -p "$results_dir"
    
    {
        echo "InferenceEngines Complete Test Report"
        echo "Generated on: $(date)"
        echo "======================================"
        echo ""
        
        echo "Project Information:"
        echo "  Project Root: $PROJECT_ROOT"
        echo "  Test Date: $(date)"
        echo "  System: $(uname -a)"
        echo ""
        
        echo "Test Results Summary:"
        if [ -f "$results_dir/summary.txt" ]; then
            cat "$results_dir/summary.txt"
        else
            echo "  No summary file found"
        fi
        echo ""
        
        echo "Individual Backend Results:"
        echo "----------------------------"
        for backend in OPENCV_DNN ONNX_RUNTIME LIBTORCH LIBTENSORFLOW TENSORRT OPENVINO; do
            local result_file="$results_dir/${backend,,}_results.xml"
            if [ -f "$result_file" ]; then
                local tests=$(grep -o 'tests="[0-9]*"' "$result_file" | cut -d'"' -f2)
                local failures=$(grep -o 'failures="[0-9]*"' "$result_file" | cut -d'"' -f2)
                local errors=$(grep -o 'errors="[0-9]*"' "$result_file" | cut -d'"' -f2)
                
                printf "  %-15s: %s tests, %s failures, %s errors\n" \
                    "$backend" "$tests" "$failures" "$errors"
            else
                printf "  %-15s: No results available\n" "$backend"
            fi
        done
        echo ""
        
        echo "Build Information:"
        echo "------------------"
        if [ -d "$PROJECT_ROOT/build" ]; then
            echo "  Build directory exists: Yes"
            echo "  Build size: $(du -sh $PROJECT_ROOT/build 2>/dev/null | cut -f1)"
        else
            echo "  Build directory exists: No"
        fi
        echo ""
        
        echo "Model Information:"
        echo "------------------"
        for backend_dir in "$PROJECT_ROOT"/backends/*/test/; do
            if [ -d "$backend_dir" ]; then
                backend_name=$(basename "$(dirname "$backend_dir")")
                cd "$backend_dir"
                
                local model_count=0
                for ext in onnx pt engine xml bin; do
                    if ls *.${ext} 1> /dev/null 2>&1; then
                        model_count=$((model_count + 1))
                    fi
                done
                
                printf "  %-15s: %d model file(s)\n" "$backend_name" "$model_count"
            fi
        done
        echo ""
        
        echo "Recommendations:"
        echo "----------------"
        if [ -f "$results_dir/summary.txt" ]; then
            local failed_count=$(grep -c "FAILED" "$results_dir/summary.txt" 2>/dev/null || echo "0")
            local passed_count=$(grep -c "PASSED" "$results_dir/summary.txt" 2>/dev/null || echo "0")
            
            if [ "$failed_count" -eq 0 ]; then
                echo "  ✓ All tested backends passed successfully"
                echo "  ✓ InferenceEngines is ready for deployment"
            elif [ "$passed_count" -gt 0 ]; then
                echo "  ⚠ Some backends failed - check individual logs"
                echo "  ⚠ Consider using only passing backends for deployment"
            else
                echo "  ✗ No backends passed testing"
                echo "  ✗ Check dependencies and configuration"
            fi
        fi
        echo ""
        
        echo "Next Steps:"
        echo "-----------"
        echo "  1. Review individual test logs in test_results/"
        echo "  2. Install missing dependencies for failed backends"
        echo "  3. Re-run tests for specific backends if needed"
        echo "  4. Use passing backends for your applications"
        echo ""
        
    } > "$report_file"
    
    # Display the report
    cat "$report_file"
    
    log_success "Final report generated: $report_file"
}

main() {
    local backend=""
    local skip_models=false
    local clean_flag=false
    local quick_test=false
    
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --backend)
                backend="$2"
                shift 2
                ;;
            --skip-models)
                skip_models=true
                shift
                ;;
            --clean)
                clean_flag=true
                shift
                ;;
            --quick)
                quick_test=true
                skip_models=true
                shift
                ;;
            --help)
                print_usage
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                print_usage
                exit 1
                ;;
        esac
    done
    
    print_header
    
    log_info "Starting complete InferenceEngines testing workflow"
    log_info "Project root: $PROJECT_ROOT"
    
    if [ "$quick_test" = true ]; then
        log_info "Running in quick test mode"
    fi
    
    cd "$PROJECT_ROOT"
    
    # Step 1: Model setup (unless skipped)
    if [ "$skip_models" = false ]; then
        if ! run_model_setup; then
            log_warning "Model setup failed, continuing with existing models..."
        fi
    else
        log_info "Skipping model setup as requested"
    fi
    
    # Step 2: Backend tests
    if ! run_backend_tests "$backend" "$clean_flag"; then
        log_warning "Some backend tests failed, generating report anyway..."
    fi
    
    # Step 3: Final report
    generate_final_report
    
    log_success "Complete testing workflow finished!"
    log_info "Check test_results/final_report.txt for detailed results"
}

# Run main function with all arguments
main "$@"
