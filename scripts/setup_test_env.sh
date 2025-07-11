#!/bin/bash

set -e

# Configuration
VENV_NAME="inference_engines_test_env"
VENV_PATH="./${VENV_NAME}"
PYTHON_VERSION="3.8"
REQUIREMENTS_FILE="./requirements.txt"

# Check Python
check_python() {
    PYTHON_CMD=$(command -v python3 || command -v python || { echo "Error: Python not found"; exit 1; })
    PYTHON_VERSION_ACTUAL=$($PYTHON_CMD --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
    [[ "$(printf '%s\n' "$PYTHON_VERSION" "$PYTHON_VERSION_ACTUAL" | sort -V | head -n1)" == "$PYTHON_VERSION" ]] || echo "Warning: Python $PYTHON_VERSION_ACTUAL may be older than $PYTHON_VERSION"
}

# Create virtual environment
create_venv() {
    rm -rf "$VENV_PATH"
    $PYTHON_CMD -m venv "$VENV_PATH" || { echo "Error: Failed to create virtual environment"; exit 1; }
}

# Activate virtual environment
activate_venv() {
    source "$VENV_PATH/bin/activate"
    [[ "$VIRTUAL_ENV" == "$(pwd)/$VENV_NAME" ]] || { echo "Error: Failed to activate virtual environment"; exit 1; }
}

# Install dependencies
install_dependencies() {
    pip install --upgrade pip
    [[ -f "$REQUIREMENTS_FILE" ]] && pip install -r "$REQUIREMENTS_FILE" || pip install pytest pytest-cov pytest-mock pytest-benchmark numpy scipy matplotlib
}

# Run tests
run_tests() {
    python -m pytest tests/ -v --cov=. --cov-report=html --cov-report=term-missing
}

# Main
CLEANUP_ON_EXIT=false
SKIP_TESTS=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --cleanup) CLEANUP_ON_EXIT=true; shift ;;
        --skip-tests) SKIP_TESTS=true; shift ;;
        --help) echo "Usage: $0 [--cleanup] [--skip-tests] [--help]"; exit 0 ;;
        *) echo "Error: Unknown option: $1"; exit 1 ;;
    esac
done

[[ -f "CMakeLists.txt" || -d "src" ]] || { echo "Error: Run from project root"; exit 1; }

[[ "$CLEANUP_ON_EXIT" == "true" ]] && trap "rm -rf $VENV_PATH" EXIT

check_python
create_venv
activate_venv
install_dependencies

if [[ "$SKIP_TESTS" == "false" ]]; then
    run_tests
else
    echo "Tests skipped"
fi

if [[ "$CLEANUP_ON_EXIT" == "false" ]]; then
    echo "Virtual environment ready at $VENV_PATH"
    echo "Activate: source $VENV_PATH/bin/activate"
    echo "Deactivate: deactivate"
    echo "Cleanup: rm -rf $VENV_PATH"
fi