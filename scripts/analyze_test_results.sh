#!/bin/bash

# Enhanced Test Results Analyzer for neuriplo
# Analyzes test results and generates comprehensive reports

set -e

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
TEST_RESULTS_DIR="$PROJECT_ROOT/test_results"
REPORTS_DIR="$PROJECT_ROOT/reports"

# Source centralized backend configuration
source "$SCRIPT_DIR/backends.conf"

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

# Note: get_backend_dir() function is now defined in backends.conf and sourced above

# Function to analyze XML test results
analyze_xml_results() {
    local backend=$1
    local backend_dir=$(get_backend_dir $backend)
    local xml_file="${TEST_RESULTS_DIR}/${backend_dir}_results.xml"
    
    if [ ! -f "$xml_file" ]; then
        echo "NOT_TESTED"
        return
    fi
    
    # Parse XML using grep and awk
    local total_tests=$(grep 'tests=' "$xml_file" | head -1 | grep -o 'tests="[^"]*"' | cut -d'"' -f2)
    local failures=$(grep 'failures=' "$xml_file" | head -1 | grep -o 'failures="[^"]*"' | cut -d'"' -f2)
    local errors=$(grep 'errors=' "$xml_file" | head -1 | grep -o 'errors="[^"]*"' | cut -d'"' -f2)
    local skipped=$(grep 'skipped=' "$xml_file" | head -1 | grep -o 'skipped="[^"]*"' | cut -d'"' -f2)
    
    if [ -z "$total_tests" ]; then
        echo "PARSE_ERROR"
        return
    fi
    
    if [ "$failures" = "0" ] && [ "$errors" = "0" ]; then
        echo "PASSED"
    else
        echo "FAILED"
    fi
}

# Function to analyze performance results
analyze_performance_results() {
    local backend=$1
    local backend_dir=$(get_backend_dir $backend)
    local perf_file="${TEST_RESULTS_DIR}/${backend_dir}_performance.log"
    
    if [ ! -f "$perf_file" ]; then
        echo "NO_DATA"
        return
    fi
    
    # Extract performance metrics
    local avg_time=$(grep "Average inference time" "$perf_file" | awk '{print $NF}' | head -1)
    local throughput=$(grep "Throughput" "$perf_file" | awk '{print $NF}' | head -1)
    local memory_usage=$(grep "Memory usage" "$perf_file" | awk '{print $NF}' | head -1)
    
    if [ -z "$avg_time" ]; then
        echo "NO_DATA"
        return
    fi
    
    # Determine performance rating
    local rating="UNKNOWN"
    if [ ! -z "$avg_time" ] && [ ! -z "$throughput" ]; then
        local avg_time_num=$(echo "$avg_time" | sed 's/[^0-9.]//g')
        local throughput_num=$(echo "$throughput" | sed 's/[^0-9.]//g')
        
        if (( $(echo "$avg_time_num < 10" | bc -l) )) && (( $(echo "$throughput_num > 50" | bc -l) )); then
            rating="EXCELLENT"
        elif (( $(echo "$avg_time_num < 50" | bc -l) )) && (( $(echo "$throughput_num > 10" | bc -l) )); then
            rating="GOOD"
        elif (( $(echo "$avg_time_num < 200" | bc -l) )) && (( $(echo "$throughput_num > 5" | bc -l) )); then
            rating="ACCEPTABLE"
        else
            rating="POOR"
        fi
    fi
    
    echo "$rating|$avg_time|$throughput|$memory_usage"
}

# Function to analyze memory leak results
analyze_memory_results() {
    local backend=$1
    local backend_dir=$(get_backend_dir $backend)
    local memory_file="${TEST_RESULTS_DIR}/${backend_dir}_memory.log"
    
    if [ ! -f "$memory_file" ]; then
        echo "NO_DATA"
        return
    fi
    
    if grep -q "memory leak detected" "$memory_file"; then
        echo "LEAK_DETECTED"
    elif grep -q "No memory leaks detected" "$memory_file"; then
        echo "NO_LEAK"
    else
        echo "UNKNOWN"
    fi
}

# Function to analyze stress test results
analyze_stress_results() {
    local backend=$1
    local backend_dir=$(get_backend_dir $backend)
    local stress_file="${TEST_RESULTS_DIR}/${backend_dir}_stress.log"
    
    if [ ! -f "$stress_file" ]; then
        echo "NO_DATA"
        return
    fi
    
    if grep -q "FAILED" "$stress_file"; then
        echo "FAILED"
    elif grep -q "PASSED" "$stress_file" || grep -q "passed" "$stress_file"; then
        echo "PASSED"
    else
        echo "UNKNOWN"
    fi
}

# Function to generate comprehensive report
generate_comprehensive_report() {
    local report_file="${REPORTS_DIR}/comprehensive_report_$(date +%Y%m%d_%H%M%S).html"
    
    mkdir -p "$REPORTS_DIR"
    
    cat > "$report_file" << 'EOF'
<!DOCTYPE html>
<html>
<head>
    <title>neuriplo Test Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .header { background-color: #f0f0f0; padding: 20px; border-radius: 5px; }
        .summary { margin: 20px 0; }
        .backend { margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }
        .passed { border-left: 5px solid #4CAF50; }
        .failed { border-left: 5px solid #f44336; }
        .not-tested { border-left: 5px solid #ff9800; }
        .metrics { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 10px; margin: 10px 0; }
        .metric { background-color: #f9f9f9; padding: 10px; border-radius: 3px; }
        .performance { background-color: #e3f2fd; }
        .memory { background-color: #f3e5f5; }
        .stress { background-color: #fff3e0; }
        table { width: 100%; border-collapse: collapse; margin: 10px 0; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #f2f2f2; }
        .status-passed { color: #4CAF50; font-weight: bold; }
        .status-failed { color: #f44336; font-weight: bold; }
        .status-unknown { color: #ff9800; font-weight: bold; }
    </style>
</head>
<body>
    <div class="header">
        <h1>neuriplo Test Report</h1>
        <p>Generated on: $(date)</p>
        <p>Project: neuriplo</p>
    </div>
    
    <div class="summary">
        <h2>Executive Summary</h2>
        <table>
            <tr>
                <th>Backend</th>
                <th>Test Status</th>
                <th>Performance</th>
                <th>Memory</th>
                <th>Stress Test</th>
            </tr>
EOF
    
    # Generate summary table
    for backend in "${BACKENDS[@]}"; do
        local backend_dir=$(get_backend_dir $backend)
        local test_status=$(analyze_xml_results $backend)
        local perf_data=$(analyze_performance_results $backend)
        local memory_status=$(analyze_memory_results $backend)
        local stress_status=$(analyze_stress_results $backend)
        
        # Parse performance data
        local perf_rating="NO_DATA"
        local avg_time="N/A"
        local throughput="N/A"
        local memory_usage="N/A"
        
        if [[ "$perf_data" != "NO_DATA" ]]; then
            IFS='|' read -r perf_rating avg_time throughput memory_usage <<< "$perf_data"
        fi
        
        # Determine CSS class
        local css_class="not-tested"
        if [ "$test_status" = "PASSED" ]; then
            css_class="passed"
        elif [ "$test_status" = "FAILED" ]; then
            css_class="failed"
        fi
        
        cat >> "$report_file" << EOF
            <tr class="$css_class">
                <td>$backend</td>
                <td class="status-$([ "$test_status" = "PASSED" ] && echo "passed" || echo "failed")">$test_status</td>
                <td>$perf_rating</td>
                <td>$memory_status</td>
                <td>$stress_status</td>
            </tr>
EOF
    done
    
    cat >> "$report_file" << 'EOF'
        </table>
    </div>
    
    <div class="details">
        <h2>Detailed Analysis</h2>
EOF
    
    # Generate detailed analysis for each backend
    for backend in "${BACKENDS[@]}"; do
        local backend_dir=$(get_backend_dir $backend)
        local test_status=$(analyze_xml_results $backend)
        local perf_data=$(analyze_performance_results $backend)
        local memory_status=$(analyze_memory_results $backend)
        local stress_status=$(analyze_stress_results $backend)
        
        # Parse performance data
        local perf_rating="NO_DATA"
        local avg_time="N/A"
        local throughput="N/A"
        local memory_usage="N/A"
        
        if [[ "$perf_data" != "NO_DATA" ]]; then
            IFS='|' read -r perf_rating avg_time throughput memory_usage <<< "$perf_data"
        fi
        
        # Determine CSS class
        local css_class="not-tested"
        if [ "$test_status" = "PASSED" ]; then
            css_class="passed"
        elif [ "$test_status" = "FAILED" ]; then
            css_class="failed"
        fi
        
        cat >> "$report_file" << EOF
        <div class="backend $css_class">
            <h3>$backend</h3>
            <div class="metrics">
                <div class="metric">
                    <strong>Test Status:</strong> $test_status
                </div>
                <div class="metric performance">
                    <strong>Performance Rating:</strong> $perf_rating<br>
                    <strong>Avg Time:</strong> $avg_time ms<br>
                    <strong>Throughput:</strong> $throughput fps
                </div>
                <div class="metric memory">
                    <strong>Memory Status:</strong> $memory_status<br>
                    <strong>Memory Usage:</strong> $memory_usage MB
                </div>
                <div class="metric stress">
                    <strong>Stress Test:</strong> $stress_status
                </div>
            </div>
EOF
        
        # Add detailed logs if available
        local test_log="${TEST_RESULTS_DIR}/${backend_dir}_test.log"
        if [ -f "$test_log" ]; then
            cat >> "$report_file" << EOF
            <details>
                <summary>Test Log</summary>
                <pre>$(head -50 "$test_log")</pre>
            </details>
EOF
        fi
        
        cat >> "$report_file" << 'EOF'
        </div>
EOF
    done
    
    cat >> "$report_file" << 'EOF'
    </div>
</body>
</html>
EOF
    
    log_success "Comprehensive HTML report generated: $report_file"
}

# Function to generate performance comparison chart
generate_performance_chart() {
    local chart_file="${REPORTS_DIR}/performance_comparison_$(date +%Y%m%d_%H%M%S).csv"
    
    mkdir -p "$REPORTS_DIR"
    
    # CSV header
    echo "Backend,Avg_Time_ms,Throughput_fps,Memory_Usage_MB,Performance_Rating" > "$chart_file"
    
    # Data for each backend
    for backend in "${BACKENDS[@]}"; do
        local perf_data=$(analyze_performance_results $backend)
        
        if [[ "$perf_data" != "NO_DATA" ]]; then
            IFS='|' read -r perf_rating avg_time throughput memory_usage <<< "$perf_data"
            echo "$backend,$avg_time,$throughput,$memory_usage,$perf_rating" >> "$chart_file"
        else
            echo "$backend,N/A,N/A,N/A,NO_DATA" >> "$chart_file"
        fi
    done
    
    log_success "Performance comparison CSV generated: $chart_file"
}

# Function to generate summary statistics
generate_summary_stats() {
    local stats_file="${REPORTS_DIR}/summary_stats_$(date +%Y%m%d_%H%M%S).txt"
    
    mkdir -p "$REPORTS_DIR"
    
    {
        echo "neuriplo Test Summary Statistics"
        echo "Generated on: $(date)"
        echo "========================================="
        echo ""
        
        # Count test results
        local total_backends=${#BACKENDS[@]}
        local passed_count=0
        local failed_count=0
        local not_tested_count=0
        
        for backend in "${BACKENDS[@]}"; do
            local status=$(analyze_xml_results $backend)
            case $status in
                "PASSED") ((passed_count++)) ;;
                "FAILED") ((failed_count++)) ;;
                *) ((not_tested_count++)) ;;
            esac
        done
        
        echo "Test Results Summary:"
        echo "  Total Backends: $total_backends"
        echo "  Passed: $passed_count"
        echo "  Failed: $failed_count"
        echo "  Not Tested: $not_tested_count"
        echo "  Success Rate: $((passed_count * 100 / total_backends))%"
        echo ""
        
        # Performance statistics
        echo "Performance Analysis:"
        local perf_count=0
        local total_avg_time=0
        local total_throughput=0
        local total_memory=0
        
        for backend in "${BACKENDS[@]}"; do
            local perf_data=$(analyze_performance_results $backend)
            if [[ "$perf_data" != "NO_DATA" ]]; then
                IFS='|' read -r perf_rating avg_time throughput memory_usage <<< "$perf_data"
                if [[ "$avg_time" =~ ^[0-9]+\.?[0-9]*$ ]]; then
                    total_avg_time=$(echo "$total_avg_time + $avg_time" | bc -l)
                    total_throughput=$(echo "$total_throughput + $throughput" | bc -l)
                    total_memory=$(echo "$total_memory + $memory_usage" | bc -l)
                    ((perf_count++))
                fi
            fi
        done
        
        if [ $perf_count -gt 0 ]; then
            local avg_inference_time=$(echo "scale=2; $total_avg_time / $perf_count" | bc -l)
            local avg_throughput=$(echo "scale=2; $total_throughput / $perf_count" | bc -l)
            local avg_memory=$(echo "scale=2; $total_memory / $perf_count" | bc -l)
            
            echo "  Backends with Performance Data: $perf_count"
            echo "  Average Inference Time: ${avg_inference_time}ms"
            echo "  Average Throughput: ${avg_throughput}fps"
            echo "  Average Memory Usage: ${avg_memory}MB"
        else
            echo "  No performance data available"
        fi
        echo ""
        
        # Memory leak analysis
        echo "Memory Leak Analysis:"
        local no_leak_count=0
        local leak_count=0
        local memory_unknown_count=0
        
        for backend in "${BACKENDS[@]}"; do
            local memory_status=$(analyze_memory_results $backend)
            case $memory_status in
                "NO_LEAK") ((no_leak_count++)) ;;
                "LEAK_DETECTED") ((leak_count++)) ;;
                *) ((memory_unknown_count++)) ;;
            esac
        done
        
        echo "  No Memory Leaks: $no_leak_count"
        echo "  Memory Leaks Detected: $leak_count"
        echo "  Unknown Memory Status: $memory_unknown_count"
        echo ""
        
        # Recommendations
        echo "Recommendations:"
        if [ $failed_count -gt 0 ]; then
            echo "  - Investigate failed backends and fix issues"
        fi
        if [ $leak_count -gt 0 ]; then
            echo "  - Address memory leaks in affected backends"
        fi
        if [ $not_tested_count -gt 0 ]; then
            echo "  - Set up testing environment for untested backends"
        fi
        if [ $perf_count -gt 0 ]; then
            echo "  - Consider performance optimization for slow backends"
        fi
        
    } > "$stats_file"
    
    log_success "Summary statistics generated: $stats_file"
}

# Function to show help
show_help() {
    echo "Usage: $0 [OPTIONS]"
    echo "Options:"
    echo "  --html-report       Generate comprehensive HTML report"
    echo "  --performance-chart Generate performance comparison CSV"
    echo "  --summary-stats     Generate summary statistics"
    echo "  --all               Generate all reports"
    echo "  --help              Show this help message"
}

# Main execution
main() {
    log_info "Starting Enhanced Test Results Analysis"
    
    # Parse command line arguments
    local generate_html=false
    local generate_chart=false
    local generate_stats=false
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --html-report)
                generate_html=true
                shift
                ;;
            --performance-chart)
                generate_chart=true
                shift
                ;;
            --summary-stats)
                generate_stats=true
                shift
                ;;
            --all)
                generate_html=true
                generate_chart=true
                generate_stats=true
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
    
    # If no specific option provided, generate all reports
    if [ "$generate_html" = false ] && [ "$generate_chart" = false ] && [ "$generate_stats" = false ]; then
        generate_html=true
        generate_chart=true
        generate_stats=true
    fi
    
    # Check if test results directory exists
    if [ ! -d "$TEST_RESULTS_DIR" ]; then
        log_error "Test results directory not found: $TEST_RESULTS_DIR"
        log_info "Run tests first using: ./scripts/test_backends.sh"
        exit 1
    fi
    
    # Generate reports
    if [ "$generate_html" = true ]; then
        log_info "Generating comprehensive HTML report..."
        generate_comprehensive_report
    fi
    
    if [ "$generate_chart" = true ]; then
        log_info "Generating performance comparison chart..."
        generate_performance_chart
    fi
    
    if [ "$generate_stats" = true ]; then
        log_info "Generating summary statistics..."
        generate_summary_stats
    fi
    
    log_success "Analysis completed successfully!"
    log_info "Reports available in: $REPORTS_DIR"
}

# Run main function
main "$@"
