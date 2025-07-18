# Inference Engines Documentation

This directory contains comprehensive documentation for building and setting up the inference engines project.

## Available Documentation

### [Dependency Management](DEPENDENCY_MANAGEMENT.md)
Comprehensive guide for setting up and managing all backend dependencies.

**Features:**
- Step-by-step installation for all inference backends
- Environment setup and configuration
- Version compatibility and requirements
- Troubleshooting common installation issues
- Automated setup scripts for each backend

### [Testing Design](TESTING_DESIGN.md)
Detailed documentation of the testing framework and methodology.

**Features:**
- Unified testing framework architecture
- Backend-specific test patterns and requirements
- Automated model generation and testing
- Performance benchmarking guidelines
- Integration testing strategies

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

### Automated Backend Testing

The project includes comprehensive automated testing for all backends:

```bash
# Test all backends with unified framework
./scripts/test_backends.sh

# Test specific backend
./scripts/test_backends.sh --backend OPENCV_DNN
./scripts/test_backends.sh --backend ONNX_RUNTIME
./scripts/test_backends.sh --backend LIBTORCH
./scripts/test_backends.sh --backend TENSORRT
./scripts/test_backends.sh --backend LIBTENSORFLOW
./scripts/test_backends.sh --backend OPENVINO
```

#### TensorFlow Backend Testing
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

#### OpenVINO Backend Testing
The OpenVINO backend includes:
- Automatic IR model generation from ONNX
- CPU and GPU device support with graceful fallback
- Comprehensive test coverage including performance and memory tests
- Integration with the unified testing framework
- **Docker-based testing environment** (see `docker/README.openvino.md`)

**Prerequisites:** Python 3, CMake, Ninja or Make, OpenVINO toolkit

**Docker Testing:**
```bash
# Quick start with Docker
./docker/run_openvino_tests.sh

# Build and run separately
./docker/run_openvino_tests.sh --build-only
./docker/run_openvino_tests.sh --run-only
```

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

### OpenVINO Setup
```bash
# Set OpenVINO environment variables
export OPENVINO_DIR=/home/oli/dependencies/openvino_2025.2.0
export PATH=$OPENVINO_DIR/bin:$OPENVINO_DIR/python_env/bin:$PATH

# Generate OpenVINO IR models
cd backends/openvino/test
./generate_openvino_ir.sh
```

## Project Structure

```
inference-engines/
├── backends/           # Backend implementations
│   ├── tensorrt/      # TensorRT backend
│   ├── onnx-runtime/  # ONNX Runtime backend
│   ├── libtorch/      # LibTorch backend
│   ├── libtensorflow/ # TensorFlow backend
│   ├── openvino/      # OpenVINO backend
│   └── opencv-dnn/    # OpenCV DNN backend
├── scripts/           # Build and test scripts
│   ├── test_backends.sh           # Unified backend testing
│   ├── run_libtensorflow_tests.sh # Automated TensorFlow testing
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
6. For OpenVINO changes, ensure IR model generation works correctly
7. Follow the unified testing framework patterns established in `test_backends.sh`

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

5. **OpenVINO test failures**
   - Ensure OpenVINO toolkit is properly installed
   - Check that IR model generation scripts work
   - Verify OpenVINO environment variables are set
   - Ensure ONNX dependencies are available for model conversion

For more detailed troubleshooting, refer to the specific backend documentation or test logs. 