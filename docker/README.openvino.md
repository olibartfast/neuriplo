# OpenVINO Backend Docker Testing

This directory contains Docker configuration for running OpenVINO backend unit tests in a containerized environment.

## Overview

The OpenVINO Docker setup provides a complete testing environment that:
- Installs OpenVINO 2025.2.0 with all dependencies
- Builds the inference engines project with OpenVINO backend
- Generates test models (ResNet-18 ONNX and OpenVINO IR)
- Runs comprehensive unit tests
- Provides isolated, reproducible testing environment

## Files

- `Dockerfile.openvino` - Multi-stage Docker build for OpenVINO testing
- `run_openvino_tests.sh` - Helper script for building and running tests
- `README.openvino.md` - This documentation file

## Quick Start

### Prerequisites

- Docker installed and running
- At least 8GB of available disk space
- Internet connection for downloading dependencies

### Basic Usage

```bash
# Build and run tests in one command
./docker/run_openvino_tests.sh

# Or build and run separately
./docker/run_openvino_tests.sh --build-only
./docker/run_openvino_tests.sh --run-only
```

### Advanced Usage

```bash
# Build only the Docker image
./docker/run_openvino_tests.sh --build-only

# Run tests with verbose output
./docker/run_openvino_tests.sh --run-only --verbose

# Clean up Docker resources
./docker/run_openvino_tests.sh --clean

# Show help
./docker/run_openvino_tests.sh --help
```

## Manual Docker Commands

If you prefer to use Docker commands directly:

```bash
# Build the image
docker build --rm -t neuriplo:openvino -f docker/Dockerfile.openvino .

# Run tests
docker run --rm neuriplo:openvino

# Run tests with volume mount for results
docker run --rm \
  -v $(pwd)/test_results:/app/test_results \
  neuriplo:openvino

# Interactive shell for debugging
docker run --rm -it neuriplo:openvino /bin/bash
```

## Docker Image Structure

The Dockerfile uses a multi-stage build approach:

### Stage 1: Base Dependencies
- Ubuntu 24.04 base image
- System dependencies (CMake, build tools, OpenCV, glog)
- Python 3 and development tools

### Stage 2: OpenVINO Installation
- Downloads and installs OpenVINO 2025.2.0
- Sets up environment variables
- Installs Python dependencies for OpenVINO

### Stage 3: Python Testing Environment
- Creates virtual environment for testing
- Installs PyTorch, ONNX, and testing frameworks
- Sets up model generation dependencies

### Stage 4: Application Build
- Copies source code
- Builds the project with OpenVINO backend
- Enables test compilation

### Stage 5: Model Generation
- Generates ResNet-18 ONNX model
- Converts to OpenVINO IR format
- Prepares test models

### Stage 6: Final Runtime
- Creates non-root user for security
- Copies built binaries and test files
- Sets up test execution environment

## Test Models

The Docker build automatically generates:

1. **ResNet-18 ONNX Model** (`resnet18.onnx`)
   - Generated using PyTorch and torchvision
   - Input shape: [1, 3, 224, 224]
   - Output: Classification probabilities

2. **OpenVINO IR Model** (`resnet18.xml` + `resnet18.bin`)
   - Converted from ONNX using OpenVINO Model Optimizer
   - Optimized for CPU inference
   - FP16 precision for performance

## Test Execution

The container runs the following test executable:
```
/app/build/backends/openvino/test/OpenVINOInferTest
```

This test suite includes:
- OpenVINO backend initialization tests
- Model loading and inference tests
- Performance and memory tests
- Error handling tests

## Environment Variables

Key environment variables set in the container:

```bash
OPENVINO_DIR=/opt/openvino_2025.2.0
PATH=/opt/openvino_2025.2.0/bin:/opt/openvino_2025.2.0/python_env/bin:$PATH
LD_LIBRARY_PATH=/opt/openvino_2025.2.0/lib:$LD_LIBRARY_PATH
PYTHONPATH=/opt/openvino_2025.2.0/python:$PYTHONPATH
```

## Troubleshooting

### Common Issues

1. **Build fails with OpenVINO download error**
   ```bash
   # Check internet connection and retry
   ./docker/run_openvino_tests.sh --clean
   ./docker/run_openvino_tests.sh --build-only
   ```

2. **Model generation fails**
   ```bash
   # Run interactive shell to debug
   docker run --rm -it neuriplo:openvino /bin/bash
   cd /app/build/backends/openvino/test
   python3 export_torchvision_classifier.py
   ./generate_openvino_ir.sh
   ```

3. **Test execution fails**
   ```bash
   # Check if models exist
   docker run --rm neuriplo:openvino ls -la /app/build/backends/openvino/test/
   
   # Run with verbose output
   ./docker/run_openvino_tests.sh --run-only --verbose
   ```

4. **Out of disk space**
   ```bash
   # Clean up Docker resources
   ./docker/run_openvino_tests.sh --clean
   docker system prune -a
   ```

### Debug Mode

For debugging, you can run an interactive shell:

```bash
# Build the image first
./docker/run_openvino_tests.sh --build-only

# Run interactive shell
docker run --rm -it neuriplo:openvino /bin/bash

# Inside the container, you can:
cd /app/build/backends/openvino/test
./OpenVINOInferTest --gtest_list_tests
./OpenVINOInferTest --gtest_filter="*Init*"
```

## Performance Considerations

- The Docker build process takes 10-20 minutes depending on your system
- The image size is approximately 4-6GB
- Test execution typically takes 1-5 minutes
- Consider using Docker build cache for faster rebuilds

## Integration with CI/CD

The Docker setup is designed to work with CI/CD pipelines:

```yaml
# Example GitHub Actions workflow
- name: Build and test OpenVINO
  run: |
    ./docker/run_openvino_tests.sh --build-only
    ./docker/run_openvino_tests.sh --run-only
```

## Security Notes

- The container runs as a non-root user (`testuser`)
- No sensitive data is stored in the image
- All dependencies are from official sources
- The container is ephemeral and doesn't persist data

## Contributing

When modifying the OpenVINO Docker setup:

1. Test the build process locally
2. Verify all tests pass
3. Update this documentation
4. Consider backward compatibility
5. Test on different Docker versions if possible

## Support

For issues with the OpenVINO Docker setup:
1. Check the troubleshooting section above
2. Review the test logs in `test_results/`
3. Run with verbose output for more details
4. Check the OpenVINO documentation for version-specific issues 