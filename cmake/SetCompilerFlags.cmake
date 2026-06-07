# Treat warnings as errors in CI (opt-in via -DWERROR=ON)
option(WERROR "Treat compiler warnings as errors" OFF)
if(WERROR)
    add_compile_options(-Wall -Wextra -Wpedantic -Werror)
    message(STATUS "Strict warnings: -Wall -Wextra -Wpedantic -Werror enabled")
endif()

# Enable ASan + UBSan (opt-in via -DSANITIZERS=ON). Use Debug builds and run tests with:
#   ASAN_OPTIONS=detect_leaks=1:abort_on_error=1 UBSAN_OPTIONS=halt_on_error=1 ctest ...
# Local helper: ./scripts/quality/sanitizers.sh
option(SANITIZERS "Enable AddressSanitizer and UndefinedBehaviorSanitizer" OFF)
if(SANITIZERS)
    add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer -g)
    add_link_options(-fsanitize=address,undefined)
    message(STATUS "Sanitizers: AddressSanitizer + UndefinedBehaviorSanitizer enabled (use Debug + ctest)")
endif()

if(CMAKE_CUDA_COMPILER)
    # If CUDA is available but not using TensorRT or LibTorch, set the CUDA flags
    set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

    # Set any specific CUDA flags for non-TensorRT and non-LibTorch code here
    set(CUDA_ARCH_FLAG "--expt-extended-lambda") # CUDA compiler option that enables support for C++11 lambdas in device code.
else()
    # If CUDA is not available, set CPU flags
    # Use a portable baseline per architecture to avoid illegal-instruction
    # crashes when Docker layer cache crosses machines with different CPU
    # capabilities (e.g. AVX-512 vs AVX2 on x86_64).
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|amd64|AMD64")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=x86-64-v3")
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64|ARM64")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=armv8.2-a")
    endif()
    set(CUDA_ARCH_FLAG "")
endif()

# Check if LibTorch is enabled
if(USE_LIBTORCH)
    # Add LibTorch-specific flags, including ${TORCH_CXX_FLAGS}
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
else()
    # Keep aggressive optimization on release profiles only.
    string(APPEND CMAKE_CXX_FLAGS_RELEASE " -O3 -ffast-math")
    string(APPEND CMAKE_CXX_FLAGS_RELWITHDEBINFO " -O2")
endif()

# Set debug flags
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0")

# Combine CUDA flags with common flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${CUDA_ARCH_FLAG}")

# Suppress warnings from TensorRT headers (TensorRT 10.x has many deprecated
# APIs and unused parameters in its own headers).  We use -Wno-error= rather
# than -Wno- so the warnings are still visible but won't break -Werror builds.
# add_compile_options is used (instead of CMAKE_CXX_FLAGS) so these flags
# appear after any earlier -Werror on the command line.
if(DEFAULT_BACKEND STREQUAL "TENSORRT")
    add_compile_options(
        -Wno-error=deprecated-declarations
        -Wno-error=unused-parameter
    )
endif()

if(DEFAULT_BACKEND STREQUAL "LITERT")
    add_compile_options(
        -Wno-error=unused-parameter
        -Wno-error=pedantic
    )
endif()

message("CMake CXX Flags Debug: ${CMAKE_CXX_FLAGS_DEBUG}")
message("CMake CXX Flags: ${CMAKE_CXX_FLAGS}")
message("CMake CUDA Flags: ${CMAKE_CUDA_FLAGS}")
