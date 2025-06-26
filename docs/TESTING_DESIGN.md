# InferenceEngines Backend Testing Configuration

## Test Design Principles

The test suite for InferenceEngines follows these key principles for atomic and mockable testing:

### 1. Atomic Testing
- Each test is independent and can run in isolation
- Tests don't depend on external resources (models are generated locally)
- Each backend test suite can be run independently
- Database/file system interactions are minimized and isolated

### 2. Mockable Architecture
- `MockInferenceInterface` provides a mock implementation for unit testing
- `AtomicBackendTest` fixture provides common utilities
- Tests can run without actual ML frameworks installed (using mocks)
- Easy to test error conditions and edge cases

### 3. Dependency Management
- Tests verify dependency versions against centralized `cmake/versions.cmake`
- Warns when installed versions don't match expected versions
- Test setup gracefully handles missing dependencies via mocks
- Ensures version consistency across build and test environments

### 4. Test Structure
Each backend test includes:
- **Initialization tests**: CPU and GPU (where applicable)
- **Inference tests**: Basic functionality with standard inputs
- **Model info tests**: Metadata retrieval
- **Batch size tests**: Different batch configurations
- **Error handling tests**: Invalid inputs and error conditions

### 4. Test Categories

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
- **TensorFlow**: SavedModel loading, session management
- **TensorRT**: Engine building, precision modes
- **OpenVINO**: IR format, device plugins

### 5. Model Generation

Test models are generated programmatically:
- **Base model**: ResNet-18 for classification (1000 classes)
- **Input format**: 224x224x3 RGB images
- **Output format**: 1000-dimensional probability vector
- **Formats**: ONNX, TorchScript, SavedModel, TensorRT Engine, OpenVINO IR

### 6. Test Execution

#### Single Backend Testing
```bash
./scripts/test_backends.sh --backend OPENCV_DNN
```

#### All Backends Testing
```bash
./scripts/test_backends.sh
```

#### Model Setup
```bash
./scripts/setup_test_models.sh
```

#### Clean Build Testing
```bash
./scripts/test_backends.sh --clean
```

### 7. Test Validation

Each test validates:
- **Correctness**: Output shapes and types
- **Performance**: Basic performance metrics
- **Robustness**: Error handling and edge cases
- **Memory**: No memory leaks or excessive usage

### 8. Continuous Integration

Tests are designed to run in CI/CD environments:
- Deterministic results
- Reasonable execution time
- Clear pass/fail criteria
- Detailed error reporting

### 9. Debugging Support

- Comprehensive logging
- XML test reports (compatible with CI tools)
- Detailed error messages
- Mock implementations for isolated debugging

### 10. Extension Guidelines

To add tests for new backends:
1. Create test files in `backends/{backend}/test/`
2. Follow the naming convention: `{Backend}InferTest.cpp`
3. Use `AtomicBackendTest` as base fixture
4. Implement model generation script
5. Update CMakeLists.txt with dependencies
6. Add backend to `test_backends.sh` script
