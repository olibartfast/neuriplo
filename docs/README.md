# Inference Engines Documentation

This directory contains comprehensive documentation for building and setting up the inference engines project.

## Available Guides

### [TensorFlow Build Guide](TENSORFLOW_BUILD_GUIDE.md)
Complete step-by-step instructions for building TensorFlow 2.19.0 from source on Ubuntu Linux systems.

**Features:**
- Detailed prerequisites and system requirements
- Step-by-step installation of Bazel, Clang, and other dependencies
- Configuration for CPU-only and GPU builds
- Troubleshooting guide for common issues
- Performance optimization tips
- Version compatibility matrix

**Quick Start:**
```bash
# Check system requirements
./scripts/setup_tensorflow.sh --check-only

# Setup environment only
./scripts/setup_tensorflow.sh --setup-only

# Complete build and installation
./scripts/setup_tensorflow.sh
```

## Backend Testing Status

| Backend | Status | Notes |
|---------|--------|-------|
| **OPENCV_DNN** | ✅ PASSED | Working correctly |
| **ONNX_RUNTIME** | ✅ PASSED | Working correctly |
| **LIBTORCH** | ✅ PASSED | Working correctly |
| **TENSORRT** | ✅ PASSED | Fixed segmentation fault issues |
| **LIBTENSORFLOW** | ✅ PASSED | Automated testing available |
| **OPENVINO** | ❌ NOT_TESTED | Requires OpenVINO toolkit |

## Testing Backends

To test all available backends:

```bash
# List available backends
./scripts/test_backends.sh --list-backends

# Test specific backend
./scripts/test_backends.sh --backend TENSORRT

# Test all backends
./scripts/test_backends.sh --all
```

### Automated libtensorflow Testing

For libtensorflow backend testing, a dedicated automated script is available:

```bash
# Run complete libtensorflow testing process from scratch
./scripts/run_libtensorflow_tests.sh
```

This script automatically:
- Sets up a temporary Python environment with TensorFlow
- Downloads and exports ResNet50 as a TensorFlow SavedModel
- Builds the project with tests enabled
- Runs all libtensorflow unittests
- Provides clear success/failure reporting

**Prerequisites:** Python 3, CMake, Ninja or Make

## Environment Setup

### Python Dependencies
```bash
# Create temporary environment for testing
./scripts/setup_test_env.sh

# Install dependencies
pip install -r requirements.txt
```

### TensorRT Setup
```bash
# Set TensorRT environment variables
export LD_LIBRARY_PATH=/home/oli/dependencies/TensorRT-10.7.0.23/lib:$LD_LIBRARY_PATH
export PATH=/home/oli/dependencies/TensorRT-10.7.0.23/bin:$PATH

# Generate TensorRT engine
cd backends/tensorrt/test
./generate_trt_engine.sh
```

## Project Structure

```
inference-engines/
├── backends/           # Backend implementations
│   ├── tensorrt/      # TensorRT backend
│   ├── onnx-runtime/  # ONNX Runtime backend
│   ├── libtorch/      # LibTorch backend
│   ├── libtensorflow/ # TensorFlow backend
│   └── ...
├── scripts/           # Build and test scripts
│   ├── run_libtensorflow_tests.sh  # Automated TensorFlow testing
│   └── ...
├── docs/             # Documentation
├── test_results/     # Test output logs
└── cmake/           # CMake configuration
```

## Contributing

When adding new backends or making changes:

1. Update the test script to include your backend
2. Add appropriate documentation
3. Ensure all tests pass
4. Update this README with status changes
5. For libtensorflow changes, ensure the automated testing script still works

## Troubleshooting

### Common Issues

1. **TensorRT engine not found**
   - Ensure engine file is in the correct location
   - Set proper environment variables
   - Run engine generation script

2. **Missing dependencies**
   - Check system requirements
   - Install missing packages
   - Verify environment variables

3. **Build failures**
   - Check compiler versions
   - Verify CMake configuration
   - Review error logs in `test_results/`

4. **libtensorflow test failures**
   - Ensure Python 3, CMake, and build tools are installed
   - Check internet connection for TensorFlow download
   - Verify TensorFlow C++ API is properly installed
   - Run the automated testing script for complete setup

For more detailed troubleshooting, refer to the specific backend documentation or test logs. 