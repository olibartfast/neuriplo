# InferenceEngines Backend Testing Configuration

## Test Design Principles

The test suite for InferenceEngines follows these key principles for atomic and mockable testing:

### 1. Atomic Testing
- Each test is independent and can run in isolation
- Tests don't depend on external resources (models are generated locally)
- Each backend test suite can be run independently
- Database/file system interactions are minimized and isolated

### 2. Mockable Architecture
- Mock implementations provide fallback for unit testing when real models are unavailable
- Tests can run without actual ML frameworks installed (using mocks)
- Easy to test error conditions and edge cases
- Graceful degradation from real model to mock testing

### 3. Dependency Management
- Tests verify dependency versions against centralized `cmake/versions.cmake`
- Warns when installed versions don't match expected versions
- Test setup gracefully handles missing dependencies via mocks
- Ensures version consistency across build and test environments

### 4. Test Structure
Each backend test includes:
- **Basic Inference tests**: Core functionality with standard inputs
- **Integration tests**: Real model testing when available
- **Mock Unit tests**: Mock-only testing when real models unavailable
- **GPU tests**: GPU availability testing (where applicable)
- **Model info tests**: Metadata retrieval
- **Performance benchmarks**: Basic performance metrics
- **Memory leak detection**: Memory safety validation
- **Stress tests**: Multi-threaded and high-load testing

### 5. Test Categories

#### Unit Tests (Atomic)
- Test individual methods in isolation
- Use mocks for dependencies
- Fast execution
- No external dependencies

#### Integration Tests  
- Test with actual backend implementations
- Use real models (generated locally)
- Test end-to-end workflows
- Validate against known outputs

#### Backend-Specific Tests
- **OpenCV DNN**: CUDA availability, model format support
- **ONNX Runtime**: Provider selection, optimization
- **LibTorch**: JIT compilation, device selection
- **TensorFlow**: SavedModel loading, session management, NCHW/NHWC format handling
- **TensorRT**: Engine building, precision modes
- **OpenVINO**: IR format, device plugins

### 6. Model Generation

Test models are generated programmatically:
- **Base model**: ResNet-50 for classification (1000 classes)
- **Input format**: 224x224x3 RGB images
- **Output format**: 1000-dimensional probability vector
- **Formats**: ONNX, TorchScript, SavedModel, TensorRT Engine, OpenVINO IR

**TensorFlow Model Generation**: Automatically generates SavedModel using ResNet-50 from Keras Applications during test execution.

### 7. Test Execution

#### Single Backend Testing
```bash
./scripts/test_backends.sh --backend LIBTENSORFLOW
```

#### All Backends Testing
```bash
./scripts/test_backends.sh
```

#### Clean Build Testing
```bash
./scripts/test_backends.sh --clean
```

#### Parallel Testing
```bash
./scripts/test_backends.sh --parallel
```

### 8. Test Validation

Each test validates:
- **Correctness**: Output shapes and types
- **Performance**: Basic performance metrics
- **Robustness**: Error handling and edge cases
- **Memory**: No memory leaks or excessive usage

#### TensorFlow-Specific Validation
- **Session Management**: Proper TensorFlow session lifecycle via SavedModelBundle
- **Memory Safety**: No double-closing of sessions or memory corruption
- **Model Loading**: SavedModel format compatibility with Keras 3+
- **Inference Execution**: End-to-end inference pipeline with proper tensor format conversion
- **Error Recovery**: Graceful handling of TensorFlow errors and fallback to mock testing
- **Format Conversion**: Proper NCHW to NHWC tensor format conversion for OpenCV blob inputs

### 9. Continuous Integration

Tests are designed to run in CI/CD environments:
- Deterministic results
- Reasonable execution time
- Clear pass/fail criteria
- Detailed error reporting
- XML test reports (compatible with CI tools)

### 10. Debugging Support

- Comprehensive logging with TensorFlow and glog integration
- XML test reports (compatible with CI tools)
- Detailed error messages with stack traces
- Mock implementations for isolated debugging
- Signal handlers for graceful crash handling

### 11. Extension Guidelines

To add tests for new backends:
1. Create test files in `backends/{backend}/test/`
2. Follow the naming convention: `{Backend}InferTest.cpp`
3. Implement mock class for unit testing
4. Use model path detection via `model_path.txt` file
5. Implement model generation if needed
6. Update CMakeLists.txt with dependencies
7. Add backend to `test_backends.sh` script

#### Backend Integration Requirements
- **Test Executable**: Must be named `{Backend}InferTest` and located at `backends/{backend}/test/{Backend}InferTest`
- **Model Path**: Read from `model_path.txt` in build directory
- **Mock Support**: Implement mock class for fallback testing
- **Error Handling**: Graceful degradation when real models unavailable
- **Test Categories**: Implement BasicInference, IntegrationTest, MockUnitTest, GPUTest, ModelInfoTest

### 12. Automated Testing Framework

#### Unified Testing System
The project uses a single, unified testing framework (`test_backends.sh`) that:
- **Automatically detects** backend availability
- **Generates models** on-demand (TensorFlow)
- **Handles dependencies** gracefully
- **Provides consistent** test execution across all backends
- **Supports parallel** testing for faster execution
- **Generates comprehensive** test reports

#### TensorFlow Backend Integration
The TensorFlow backend is fully integrated into the main testing framework:
- **Automatic model generation**: Creates SavedModel during test execution
- **Environment setup**: Creates temporary Python environment for model generation
- **Format handling**: Properly converts between OpenCV NCHW and TensorFlow NHWC formats
- **Memory safety**: Robust session and tensor management
- **Error recovery**: Graceful fallback to mock testing on failures

#### Test Execution Flow
1. **Backend Detection**: Check for required dependencies and libraries
2. **Model Setup**: Generate or locate test models (TensorFlow: auto-generate)
3. **Build Process**: Configure and build with specific backend
4. **Test Execution**: Run comprehensive test suite
5. **Performance Testing**: Execute performance benchmarks
6. **Memory Testing**: Run memory leak detection
7. **Stress Testing**: Execute multi-threaded stress tests
8. **Report Generation**: Create detailed test reports

### 13. File Organization

#### Essential Files
- `scripts/test_backends.sh` - Main testing framework
- `scripts/setup_tensorflow_pip.sh` - TensorFlow C++ development setup
- `scripts/setup_test_env.sh` - Python test environment setup
- `scripts/model_downloader.py` - Model download utility
- `backends/{backend}/test/{Backend}InferTest.cpp` - Backend-specific tests

#### Removed Redundant Files
- ~~`scripts/run_libtensorflow_tests.sh`~~ - Replaced by integrated testing
- ~~`scripts/README_libtensorflow_tests.md`~~ - No longer needed
- ~~`scripts/test_hybrid_approach.sh`~~ - Unused
- ~~`generate_saved_model.py`~~ - Duplicate functionality
- ~~`backends/libtensorflow/test/generate_*.py`~~ - Integrated into main framework

### 14. Best Practices

#### For Backend Developers
1. **Follow the pattern**: Use existing backend tests as templates
2. **Implement mocks**: Always provide mock implementation for unit testing
3. **Handle errors gracefully**: Fall back to mock testing when real models unavailable
4. **Test format conversions**: Ensure proper tensor format handling
5. **Memory safety**: Avoid memory leaks and corruption
6. **Documentation**: Update this document when adding new backends

#### For Test Maintainers
1. **Keep tests atomic**: Each test should be independent
2. **Use consistent naming**: Follow established naming conventions
3. **Handle dependencies**: Graceful degradation when dependencies missing
4. **Performance monitoring**: Include basic performance validation
5. **Memory validation**: Check for memory leaks and corruption
6. **Error reporting**: Provide clear, actionable error messages
