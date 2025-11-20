# Platform-specific compiler flags
if(MSVC)
    # Windows MSVC compiler flags
    if (CMAKE_CUDA_COMPILER)
        # CUDA flags for MSVC
        set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
        set(CUDA_ARCH_FLAG "--expt-extended-lambda")
    else()
        set(CUDA_ARCH_FLAG "")
    endif()
    
    # Check if LibTorch is enabled
    if (USE_LIBTORCH)
        # Add LibTorch-specific flags for MSVC
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
    else()
        # MSVC optimization flags
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /O2 /fp:fast")
    endif()
    
    # MSVC-specific flags
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W3 /EHsc /std:c++17")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /Od /Zi")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /O2 /DNDEBUG")
    
    # Disable specific warnings
    add_compile_definitions(_CRT_SECURE_NO_WARNINGS)
    
else()
    # Unix/Linux/macOS compiler flags (GCC/Clang)
    if (CMAKE_CUDA_COMPILER)
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
    if (USE_LIBTORCH)
        # Add LibTorch-specific flags, including ${TORCH_CXX_FLAGS}
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
    else()
        # If LibTorch is not enabled, set common optimization flags
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -ffast-math")
    endif()

    # Set debug flags
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0")
endif()

# Combine CUDA flags with common flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${CUDA_ARCH_FLAG}")

message("CMake CXX Flags Debug: ${CMAKE_CXX_FLAGS_DEBUG}")
message("CMake CXX Flags: ${CMAKE_CXX_FLAGS}")
message("CMake CUDA Flags: ${CMAKE_CUDA_FLAGS}")
