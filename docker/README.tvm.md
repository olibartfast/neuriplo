# TVM Backend Docker Testing

This directory contains Docker configuration for running TVM backend unit tests in a containerized environment.

## Overview

The TVM Docker setup provides a complete testing environment that:
- Installs TVM 0.18.0 with LLVM support from source
- Builds the neuriplo project with TVM backend
- Compiles test models (ResNet-18) to TVM format
- Runs comprehensive unit tests
- Provides isolated, reproducible testing environment

## Files

- `Dockerfile.tvm` - Multi-stage Docker build for TVM testing
- `run_tvm_tests.sh` - Helper script for building and running tests (if available)
- `README.tvm.md` - This documentation file

## Quick Start

### Prerequisites

- Docker installed and running
- At least 10GB of available disk space (TVM build is large)
- Internet connection for downloading dependencies
- Minimum 4GB RAM recommended

### Basic Usage

```bash
# Build and run tests with one command
docker build --rm -t neuriplo:tvm -f docker/Dockerfile.tvm . && \
docker run --rm neuriplo:tvm

# Or build and run separately
docker build --rm -t neuriplo:tvm -f docker/Dockerfile.tvm .
docker run --rm neuriplo:tvm
```

### Advanced Usage

```bash
# Build with custom TVM version
docker build --rm -t neuriplo:tvm \
  --build-arg TVM_VERSION=0.18.0 \
  -f docker/Dockerfile.tvm .

# Run tests with volume mount for results
docker run --rm \
  -v $(pwd)/test_results:/app/test_results \
  neuriplo:tvm

# Interactive shell for debugging
docker run --rm -it neuriplo:tvm /bin/bash

# Run specific test filter
docker run --rm neuriplo:tvm ./TVMInferTest --gtest_filter="*ModelLoad*"

# List all available tests
docker run --rm neuriplo:tvm ./TVMInferTest --gtest_list_tests
```

## Docker Image Structure

The Dockerfile uses a multi-stage build approach for efficiency:

### Stage 1: Base Dependencies
- Ubuntu 24.04 base image
- System dependencies (CMake, build tools, OpenCV, glog)
- LLVM 12 for TVM compilation and optimization
- Python 3 and development tools

### Stage 2: TVM Installation
- Clones TVM repository (version 0.18.0)
- Configures TVM with LLVM backend support
- Builds TVM from source (~10-20 minutes)
- Installs TVM Python package and dependencies
- Verifies installation integrity

### Stage 3: Model Compilation
- Downloads ResNet-18 ONNX model
- Compiles ONNX to TVM optimized format (.so)
- Uses Relay IR for model representation
- Applies optimization level 3 for best performance

### Stage 4: Application Build
- Copies neuriplo source code
- Builds project with TVM backend enabled
- Compiles test suite

### Stage 5: Model Preparation
- Copies compiled TVM model to test directory
- Creates model configuration files
- Prepares test environment

### Stage 6: Final Runtime
- Creates non-root user for security
- Copies built binaries and test files
- Sets up TVM runtime environment
- Configures library paths

## Test Models

The Docker build automatically generates:

1. **ResNet-18 ONNX Model** (`resnet18.onnx`)
   - Downloaded from ONNX Model Zoo
   - Input shape: [1, 3, 224, 224]
   - Output: Classification probabilities (1000 classes)

2. **TVM Compiled Model** (`resnet18_tvm.so`)
   - Compiled from ONNX using TVM compiler
   - Optimized for CPU with LLVM (core-avx2)
   - Graph executor format for inference
   - Optimization level: 3 (aggressive)

## Test Execution

The container runs the following test executable:
```
/app/build/backends/tvm/test/TVMInferTest
```

This test suite includes:
- TVM backend initialization tests
- Model loading and inference tests
- Tensor input/output validation tests
- Performance benchmarking tests
- Error handling tests

## Environment Variables

Key environment variables set in the container:

```bash
TVM_HOME=/opt/tvm
TVM_DIR=/root/dependencies/tvm
PYTHONPATH=/opt/tvm/python:${PYTHONPATH}
LD_LIBRARY_PATH=/app/build/lib:/opt/tvm/build:$LD_LIBRARY_PATH
```

## TVM Configuration

The TVM installation uses the following configuration:

```cmake
USE_LLVM=/usr/lib/llvm-12/bin/llvm-config  # LLVM for CPU optimization
USE_CUDA=OFF                                # GPU support disabled
USE_RTTI=ON                                 # C++ RTTI enabled
USE_GRAPH_EXECUTOR=ON                       # Graph executor enabled
USE_GRAPH_EXECUTOR_DEBUG=ON                 # Debug mode for graph executor
BUILD_SHARED_LIBS=ON                        # Build shared libraries
```

## Troubleshooting

### Common Issues

1. **Build fails during TVM compilation**
   ```bash
   # This is normal - TVM build takes 10-20 minutes
   # Ensure you have enough disk space (10GB+)
   # Check build logs for specific errors
   docker build --progress=plain -t neuriplo:tvm -f docker/Dockerfile.tvm .
   ```

2. **Model compilation fails**
   ```bash
   # Run interactive shell to debug
   docker run --rm -it neuriplo:tvm /bin/bash
   cd /opt/models
   python3 compile_model.py
   ```

3. **Test execution fails**
   ```bash
   # Check if TVM libraries are loaded
   docker run --rm neuriplo:tvm ldd ./TVMInferTest

   # Verify model exists
   docker run --rm neuriplo:tvm ls -la /app/build/backends/tvm/test/

   # Run with verbose output
   docker run --rm neuriplo:tvm ./TVMInferTest --gtest_verbose
   ```

4. **Out of memory during TVM build**
   ```bash
   # Reduce parallel jobs in Dockerfile
   # Change: make -j$(nproc)
   # To: make -j2
   # Edit Dockerfile.tvm line 76
   ```

5. **LLVM not found**
   ```bash
   # Verify LLVM is installed
   docker run --rm neuriplo:tvm llvm-config-12 --version

   # Check LLVM path
   docker run --rm neuriplo:tvm ls -la /usr/lib/llvm-12/bin/llvm-config
   ```

### Debug Mode

For debugging, you can run an interactive shell:

```bash
# Build the image first
docker build --rm -t neuriplo:tvm -f docker/Dockerfile.tvm .

# Run interactive shell
docker run --rm -it neuriplo:tvm /bin/bash

# Inside the container, you can:
cd /app/build/backends/tvm/test

# List available tests
./TVMInferTest --gtest_list_tests

# Run specific test
./TVMInferTest --gtest_filter="*Init*"

# Check TVM installation
python3 -c "import tvm; print(tvm.__version__)"

# Verify model
ls -lh resnet18_tvm.so
file resnet18_tvm.so
```

## Performance Considerations

- **Build Time**: 15-30 minutes depending on your system (TVM compilation is intensive)
- **Image Size**: Approximately 6-8GB (TVM is a large framework)
- **Test Execution**: Typically takes 1-5 minutes
- **Memory Usage**: 2-4GB during build, 1-2GB during runtime
- **CPU**: TVM uses LLVM optimization, benefits from multiple cores

### Optimization Tips

```bash
# Use Docker build cache for faster rebuilds
docker build --rm -t neuriplo:tvm -f docker/Dockerfile.tvm .

# Clean build (no cache)
docker build --rm --no-cache -t neuriplo:tvm -f docker/Dockerfile.tvm .

# Multi-core compilation (default uses all cores)
# Edit Dockerfile to limit: make -j4
```

## TVM Compilation Options

The model compilation script supports customization:

```python
# Target options (edit /opt/models/compile_model.py in container):
target = "llvm -mcpu=core-avx2"           # Intel CPUs with AVX2
target = "llvm -mcpu=skylake"             # Intel Skylake
target = "llvm -mtriple=aarch64-linux"    # ARM64
target = "cuda"                            # NVIDIA GPU (requires CUDA)

# Optimization levels:
opt_level=0  # No optimization
opt_level=1  # Basic optimizations
opt_level=2  # Standard optimizations (default)
opt_level=3  # Aggressive optimizations (used in Docker)
```

## Integration with CI/CD

The Docker setup is designed to work with CI/CD pipelines:

```yaml
# Example GitHub Actions workflow
- name: Build and test TVM backend
  run: |
    docker build --rm -t neuriplo:tvm -f docker/Dockerfile.tvm .
    docker run --rm neuriplo:tvm

# Example GitLab CI
test:tvm:
  script:
    - docker build --rm -t neuriplo:tvm -f docker/Dockerfile.tvm .
    - docker run --rm neuriplo:tvm
  timeout: 45m  # TVM build takes time
```

## Comparison with Other Backends

| Feature | TVM | ONNX Runtime | TensorRT |
|---------|-----|--------------|----------|
| Build Time | 15-30 min | 2-5 min | 5-10 min |
| Image Size | 6-8 GB | 2-3 GB | 4-5 GB |
| CPU Performance | Excellent | Good | N/A |
| GPU Support | Yes (optional) | Yes | Yes |
| Auto-optimization | Yes (LLVM) | Limited | Excellent |
| Compilation Required | Yes | No | Yes |

## Security Notes

- The container runs as a non-root user (`testuser`)
- No sensitive data is stored in the image
- All dependencies are from official sources (Apache TVM, ONNX)
- The container is ephemeral and doesn't persist data
- LLVM is from Ubuntu official repositories

## Advanced Features

### Auto-Tuning (Not included in Docker, but available)

TVM supports auto-tuning for optimal performance:

```python
# Auto-tuning example (run outside Docker or in custom script)
from tvm import autotvm
from tvm.autotvm.tuner import XGBTuner

# Extract tuning tasks
tasks = autotvm.task.extract_from_program(mod["main"], target=target, params=params)

# Tune each task
for task in tasks:
    tuner = XGBTuner(task)
    tuner.tune(n_trial=1000)
```

### Custom Model Compilation

To compile your own models:

```bash
# Run container with volume mount
docker run --rm -it \
  -v $(pwd)/models:/models \
  neuriplo:tvm /bin/bash

# Inside container, compile custom model
python3 << 'EOF'
import onnx
import tvm
from tvm import relay

onnx_model = onnx.load("/models/your_model.onnx")
mod, params = relay.frontend.from_onnx(onnx_model, shape_dict)

target = "llvm -mcpu=core-avx2"
with tvm.transform.PassContext(opt_level=3):
    lib = relay.build(mod, target=target, params=params)
lib.export_library("/models/your_model_tvm.so")
EOF
```

## Contributing

When modifying the TVM Docker setup:

1. Test the build process locally (allow 30+ minutes)
2. Verify all tests pass
3. Update this documentation
4. Test with different TVM versions if changing version
5. Consider build time and image size impacts
6. Validate LLVM compatibility

## Resources

- **TVM Documentation**: https://tvm.apache.org/docs/
- **TVM Installation Guide**: https://tvm.apache.org/docs/install/from_source.html
- **TVM Tutorials**: https://tvm.apache.org/docs/tutorials/
- **TVM GitHub**: https://github.com/apache/tvm
- **LLVM**: https://llvm.org/

## Support

For issues with the TVM Docker setup:

1. Check the troubleshooting section above
2. Review TVM build logs for compilation errors
3. Verify LLVM installation with `llvm-config-12 --version`
4. Check TVM documentation for version-specific issues
5. Ensure sufficient disk space and memory
6. Try building with `--progress=plain` for detailed output

## Known Limitations

- GPU support (CUDA) is disabled by default for compatibility
- Auto-tuning is not included (adds significant build time)
- ROCm (AMD GPU) support not configured
- OpenCL and Vulkan backends not enabled
- Build requires significant time and resources

## Future Enhancements

Potential improvements for future versions:

- [ ] Add CUDA support for GPU testing
- [ ] Include auto-tuning example scripts
- [ ] Support multiple model formats (PyTorch, TensorFlow)
- [ ] Add benchmark comparison tools
- [ ] Reduce image size with multi-stage optimization
- [ ] Support ARM64 builds
