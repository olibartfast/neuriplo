# neuriplo Backend Testing Scripts

This directory contains comprehensive testing scripts for all neuriplo backends. The testing framework is designed with atomic and mockable principles to ensure reliable, maintainable tests.

## Quick Start

### 1. Setup Test Models
```bash
./scripts/setup_test_models.sh
```

### 2. Test All Backends
```bash
./scripts/test_backends.sh
```

### 3. Test Specific Backend
```bash
./scripts/test_backends.sh --backend OPENCV_DNN
```

## Available Scripts

### `test_backends.sh`
Main testing script that builds and tests each backend individually.

**Usage:**
```bash
./scripts/test_backends.sh [OPTIONS]
```

**Options:**
- `--backend BACKEND` - Test only specific backend
- `--clean` - Clean builds before testing
- `--skip-build` - Skip build step, only run tests
- `--help` - Show help message

**Available Backends:**
- `OPENCV_DNN` - OpenCV DNN module
- `ONNX_RUNTIME` - ONNX Runtime
- `LIBTORCH` - PyTorch LibTorch
- `LIBTENSORFLOW` - TensorFlow C++
- `TENSORRT` - NVIDIA TensorRT
- `OPENVINO` - Intel OpenVINO

**Examples:**
```bash
# Test all available backends
./scripts/test_backends.sh

# Test only OpenCV DNN backend
./scripts/test_backends.sh --backend OPENCV_DNN

# Clean build and test TensorRT
./scripts/test_backends.sh --backend TENSORRT --clean

# Skip build and only run tests
./scripts/test_backends.sh --skip-build
```

### `setup_test_models.sh`
Sets up test models for all backends.

**Usage:**
```bash
./scripts/setup_test_models.sh [OPTIONS]
```

**Options:**
- `--verify-only` - Only verify existing models, don't generate new ones
- `--help` - Show help message

**What it does:**
- Generates ResNet-18 ONNX model as base
- Creates backend-specific model formats
- Copies models to appropriate test directories
- Verifies model integrity

## Test Architecture

### Atomic Testing Principles
- Each test runs independently
- No shared state between tests
- Isolated from external dependencies
- Deterministic results

### Mockable Design
- `MockInferenceInterface` for unit testing
- `AtomicBackendTest` base fixture
- Easy error condition testing
- Framework-independent testing

### Test Categories

#### 1. Unit Tests
- **File**: `{Backend}InferTest.cpp`
- **Purpose**: Test individual methods
- **Dependencies**: Minimal, uses mocks where possible
- **Speed**: Fast execution

#### 2. Integration Tests
- **Purpose**: End-to-end backend testing
- **Dependencies**: Real backend libraries
- **Speed**: Moderate execution time

#### 3. Backend-Specific Tests
Each backend includes specialized tests:

**OpenCV DNN (`OCVDNNInferTest`)**
- CUDA availability detection
- Model format support (ONNX, Darknet)
- Backend/target selection

**ONNX Runtime (`ONNXRuntimeInferTest`)**
- Execution provider selection
- GPU/CPU fallback
- Optimization settings

**LibTorch (`LibtorchInferTest`)**
- JIT model loading
- Device selection (CPU/CUDA)
- Dynamic shapes

**TensorFlow (`TensorFlowInferTest`)**
- SavedModel loading
- Session management
- Input/output handling

**TensorRT (`TensorRTInferTest`)**
- Engine building/loading
- CUDA memory management
- Precision modes (FP32/FP16)

**OpenVINO (`OpenVINOInferTest`)**
- IR format loading
- Device plugin selection
- Dynamic batching

## Test Models

All tests use ResNet-18 classifier as the standard model:
- **Input**: 224x224x3 RGB images
- **Output**: 1000-class probability distribution
- **Formats**: Generated per backend requirements

### Model Files
- `resnet18.onnx` - ONNX format (OpenCV DNN, ONNX Runtime)
- `resnet18.pt` - TorchScript format (LibTorch)
- `saved_model/` - TensorFlow SavedModel (TensorFlow)
- `resnet18.engine` - TensorRT engine (TensorRT)
- `resnet18.xml/.bin` - OpenVINO IR (OpenVINO)

## Test Results

### Output Locations
- **Build artifacts**: `build/{backend}/`
- **Test results**: `test_results/`
- **XML reports**: `test_results/{backend}_results.xml`
- **Summary**: `test_results/summary.txt`

### Result Interpretation
- **PASSED**: All tests successful
- **FAILED**: One or more tests failed
- **NOT_TESTED**: Backend dependencies not available

## Continuous Integration

### CI/CD Integration
The testing framework is designed for CI/CD:
```yaml
# Example GitHub Actions step
- name: Test neuriplo Backends
  run: |
    ./scripts/setup_test_models.sh
    ./scripts/test_backends.sh
```

### Docker Testing
For containerized testing:
```dockerfile
# Install dependencies
RUN apt-get update && apt-get install -y \
    build-essential cmake \
    libopencv-dev libgoogle-glog-dev

# Copy and run tests
COPY scripts/ /app/scripts/
RUN ./scripts/test_backends.sh
```

## Troubleshooting

### Common Issues

#### 1. Missing Dependencies
**Error**: Backend not available
**Solution**: Install required libraries or use dependency setup scripts

#### 2. Model Generation Fails
**Error**: PyTorch/TensorFlow not found
**Solution**: Install required Python packages:
```bash
pip install torch torchvision tensorflow
```

#### 3. GPU Tests Fail
**Error**: CUDA not available
**Solution**: Tests will automatically fallback to CPU mode

#### 4. Build Failures
**Error**: CMake configuration fails
**Solution**: Check dependency paths and versions

### Debug Mode
For detailed debugging:
```bash
# Enable verbose output
export GLOG_v=2
./scripts/test_backends.sh --backend OPENCV_DNN

# Check individual test logs
cat test_results/opencv_dnn_results.xml
```

## Extending Tests

### Adding New Backend
1. Create test directory: `backends/{backend}/test/`
2. Implement test file: `{Backend}InferTest.cpp`
3. Create CMakeLists.txt with dependencies
4. Add model generation script
5. Update `test_backends.sh` script

### Adding New Test Cases
Use the provided base classes:
```cpp
#include "MockInferenceInterface.hpp"

class MyBackendTest : public AtomicBackendTest {
protected:
    void SetUp() override {
        AtomicBackendTest::SetUp();
        // Custom setup
    }
};

TEST_F(MyBackendTest, CustomTest) {
    // Test implementation
    ValidateInferenceResult(result);
}
```

## Performance Testing

### Benchmarking
Basic performance metrics are collected:
- Model loading time
- Inference time
- Memory usage

### Profiling
For detailed profiling:
```bash
# Use with profiling tools
valgrind --tool=callgrind ./backends/opencv-dnn/test/OCVDNNInferTest
```

## Contributing

### Guidelines
1. Follow atomic testing principles
2. Use provided mock interfaces
3. Include both positive and negative test cases
4. Add comprehensive documentation
5. Ensure CI/CD compatibility

### Code Style
Follow the existing patterns:
- Use `AtomicBackendTest` as base
- Validate results with provided utilities  
- Include setup/teardown in fixtures
- Use meaningful test names

## Testing Philosophy

The neuriplo testing follows a hybrid approach combining:

1. **Dependency Verification**: Tests verify installed dependencies against centralized versions in `cmake/versions.cmake`
2. **Integration Testing**: Real model inference tests when dependencies and models are available
3. **Fallback Mechanisms**: Dummy model creation when real models are unavailable
4. **Unit Testing**: Mock interfaces for pure unit tests that always pass

### Dependency Management Integration

- The testing system checks installed dependency versions against those defined in `cmake/versions.cmake`
- Warns if dependencies are missing or version mismatches exist
- Provides helpful feedback on how to install the correct versions
- Can install basic test dependencies automatically
