#!/bin/bash
rm -rf build && mkdir -p build && cd build

FAILURES=()

# Helper to build
build_backend() {
    BACKEND=$1
    shift
    ARGS="$@"
    echo "========================================"
    echo "Building $BACKEND $ARGS..."
    echo "========================================"
    rm -f CMakeCache.txt
    if ! cmake .. -DDEFAULT_BACKEND=$BACKEND $ARGS > /dev/null; then
        echo "CMake failed for $BACKEND"
        return 1
    fi
    if ! make clean > /dev/null; then
        echo "Make clean failed for $BACKEND"
        return 1
    fi
    if ! make -j4 neuriplo; then
        echo "Make failed for $BACKEND"
        return 1
    fi
    return 0
}

# LibTorch - Try disabling CUDA
export USE_CUDA=0
if ! build_backend "LIBTORCH" "-DCUDA_FOUND=OFF"; then
    FAILURES+=("LIBTORCH")
fi
unset USE_CUDA

# OpenVINO
if ! build_backend "OPENVINO"; then
    FAILURES+=("OPENVINO")
fi

# TensorFlow
if ! build_backend "LIBTENSORFLOW"; then
    FAILURES+=("LIBTENSORFLOW")
fi

# GGML
if ! build_backend "GGML"; then
    FAILURES+=("GGML")
fi

# TVM
if ! build_backend "TVM"; then
    FAILURES+=("TVM")
fi

# OpenCV
if ! build_backend "OPENCV_DNN"; then
    FAILURES+=("OPENCV_DNN")
fi

echo "========================================"
if [ ${#FAILURES[@]} -eq 0 ]; then
    echo "All backends compiled successfully!"
else
    echo "Failures: ${FAILURES[*]}"
    exit 1
fi
echo "========================================"
