#!/bin/bash

# Setup temporary virtual environment for unit testing
# This script creates a clean environment for testing and handles cleanup

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
VENV_NAME="inference_engines_test_env"
VENV_PATH="./${VENV_NAME}"
PYTHON_VERSION="3.8"  # Adjust as needed
REQUIREMENTS_FILE="./requirements.txt"

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

# Function to cleanup virtual environment
cleanup_venv() {
    if [ -d "$VENV_PATH" ]; then
        print_status "Cleaning up virtual environment..."
        rm -rf "$VENV_PATH"
        print_success "Virtual environment cleaned up"
    fi
}

# Function to check if Python is available
check_python() {
    if command -v python3 &> /dev/null; then
        PYTHON_CMD="python3"
    elif command -v python &> /dev/null; then
        PYTHON_CMD="python"
    else
        print_error "Python not found. Please install Python 3.8 or higher."
        exit 1
    fi
    
    # Check Python version
    PYTHON_VERSION_ACTUAL=$($PYTHON_CMD --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
    print_status "Found Python $PYTHON_VERSION_ACTUAL"
    
    if [ "$(printf '%s\n' "$PYTHON_VERSION" "$PYTHON_VERSION_ACTUAL" | sort -V | head -n1)" != "$PYTHON_VERSION" ]; then
        print_warning "Python version $PYTHON_VERSION_ACTUAL may be older than recommended $PYTHON_VERSION"
    fi
}

# Function to create virtual environment
create_venv() {
    print_status "Creating virtual environment at $VENV_PATH"
    
    # Clean up any existing environment
    cleanup_venv
    
    # Create new virtual environment
    $PYTHON_CMD -m venv "$VENV_PATH"
    
    if [ ! -d "$VENV_PATH" ]; then
        print_error "Failed to create virtual environment"
        exit 1
    fi
    
    print_success "Virtual environment created successfully"
}

# Function to activate virtual environment
activate_venv() {
    print_status "Activating virtual environment..."
    
    # Source the activation script
    source "$VENV_PATH/bin/activate"
    
    # Verify activation
    if [ "$VIRTUAL_ENV" != "$(pwd)/$VENV_NAME" ]; then
        print_error "Failed to activate virtual environment"
        exit 1
    fi
    
    print_success "Virtual environment activated"
    print_status "Python path: $(which python)"
    print_status "Pip path: $(which pip)"
}

# Function to install dependencies
install_dependencies() {
    print_status "Installing dependencies..."
    
    # Upgrade pip
    pip install --upgrade pip
    
    # Install requirements if file exists
    if [ -f "$REQUIREMENTS_FILE" ]; then
        print_status "Installing from requirements.txt"
        pip install -r "$REQUIREMENTS_FILE"
    else
        print_warning "No requirements.txt found, installing basic testing dependencies"
        pip install pytest pytest-cov pytest-mock pytest-benchmark
    fi
    
    # Install additional testing dependencies
    pip install numpy scipy matplotlib  # Common ML dependencies
    
    print_success "Dependencies installed successfully"
}

# Function to run tests
run_tests() {
    print_status "Running tests in virtual environment..."
    
    # Run pytest with coverage
    python -m pytest tests/ -v --cov=. --cov-report=html --cov-report=term-missing
    
    print_success "Tests completed"
}

# Function to deactivate and cleanup
deactivate_and_cleanup() {
    print_status "Deactivating virtual environment..."
    deactivate
    
    if [ "$1" = "cleanup" ]; then
        cleanup_venv
    else
        print_status "Virtual environment preserved at $VENV_PATH"
        print_status "To clean up later, run: rm -rf $VENV_PATH"
    fi
}

# Main execution
main() {
    print_status "Setting up temporary virtual environment for unit testing"
    
    # Check if we're in the right directory
    if [ ! -f "CMakeLists.txt" ] && [ ! -d "src" ]; then
        print_error "Please run this script from the project root directory"
        exit 1
    fi
    
    # Parse command line arguments
    CLEANUP_ON_EXIT=false
    SKIP_TESTS=false
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --cleanup)
                CLEANUP_ON_EXIT=true
                shift
                ;;
            --skip-tests)
                SKIP_TESTS=true
                shift
                ;;
            --help)
                echo "Usage: $0 [OPTIONS]"
                echo "Options:"
                echo "  --cleanup     Clean up virtual environment after tests"
                echo "  --skip-tests  Skip running tests, just setup environment"
                echo "  --help        Show this help message"
                exit 0
                ;;
            *)
                print_error "Unknown option: $1"
                exit 1
                ;;
        esac
    done
    
    # Setup trap for cleanup on exit
    if [ "$CLEANUP_ON_EXIT" = true ]; then
        trap cleanup_venv EXIT
    fi
    
    # Execute setup steps
    check_python
    create_venv
    activate_venv
    install_dependencies
    
    if [ "$SKIP_TESTS" = false ]; then
        run_tests
    else
        print_status "Skipping tests as requested"
    fi
    
    if [ "$CLEANUP_ON_EXIT" = true ]; then
        print_status "Cleaning up on exit..."
    else
        print_success "Virtual environment ready for use"
        print_status "To activate manually: source $VENV_PATH/bin/activate"
        print_status "To deactivate: deactivate"
        print_status "To clean up: rm -rf $VENV_PATH"
    fi
}

# Run main function
main "$@" 