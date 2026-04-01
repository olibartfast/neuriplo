# Treat warnings as errors in CI (opt-in via -DWERROR=ON)
option(WERROR "Treat compiler warnings as errors" OFF)
if(WERROR)
    add_compile_options(-Wall -Wextra -Wpedantic -Werror)
    message(STATUS "Strict warnings: -Wall -Wextra -Wpedantic -Werror enabled")
endif()

# Enable sanitizers in debug builds (opt-in via -DSANITIZERS=ON)
option(SANITIZERS "Enable AddressSanitizer and UndefinedBehaviorSanitizer" OFF)
if(SANITIZERS)
    add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer)
    add_link_options(-fsanitize=address,undefined)
    message(STATUS "Sanitizers: AddressSanitizer + UndefinedBehaviorSanitizer enabled")
endif()

if(CMAKE_CUDA_COMPILER)
    # If CUDA is available but not using TensorRT or LibTorch, set the CUDA flags
    set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

    # Set any specific CUDA flags for non-TensorRT and non-LibTorch code here
    set(CUDA_ARCH_FLAG "--expt-extended-lambda") # CUDA compiler option that enables support for C++11 lambdas in device code.
else()
    # If CUDA is not available, set CPU flags
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
    set(CUDA_ARCH_FLAG "")
endif()

# Check if LibTorch is enabled
if(USE_LIBTORCH)
    # Add LibTorch-specific flags, including ${TORCH_CXX_FLAGS}
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
else()
    # If LibTorch is not enabled, set common optimization flags
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -ffast-math")
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

message("CMake CXX Flags Debug: ${CMAKE_CXX_FLAGS_DEBUG}")
message("CMake CXX Flags: ${CMAKE_CXX_FLAGS}")
message("CMake CUDA Flags: ${CMAKE_CUDA_FLAGS}")
