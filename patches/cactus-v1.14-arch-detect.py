"""
Patch cactus/CMakeLists.txt to add host-architecture detection.

Cactus v1.14 unconditionally applies ARM SIMD flags and builds the
ARM i8mm kernel on every non-Apple platform.  This script wraps those
sections in CMAKE_SYSTEM_PROCESSOR guards so the same CMakeLists works
on both aarch64 and x86_64 hosts.
"""
import sys

path = sys.argv[1]
content = open(path).read()

# ── Fix 1: architecture-guard the non-Apple CXX_FLAGS block ──────────────────
old1 = (
    'else()\n'
    '    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=armv8.2-a+fp16+simd+dotprod+i8mm -pthread -Wall -Wextra -pedantic -O3 -Wno-missing-field-initializers")\n'
    '    add_compile_definitions(\n'
    '        __ARM_NEON=1\n'
    '        __ARM_FEATURE_FP16_VECTOR_ARITHMETIC=1\n'
    '        __ARM_FEATURE_DOTPROD=1\n'
    '        __ARM_FEATURE_MATMUL_INT8=1\n'
    '    )\n'
    'endif()'
)
new1 = (
    'else()\n'
    '    if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm")\n'
    '        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=armv8.2-a+fp16+simd+dotprod+i8mm -pthread -Wall -Wextra -pedantic -O3 -Wno-missing-field-initializers")\n'
    '        add_compile_definitions(\n'
    '            __ARM_NEON=1\n'
    '            __ARM_FEATURE_FP16_VECTOR_ARITHMETIC=1\n'
    '            __ARM_FEATURE_DOTPROD=1\n'
    '            __ARM_FEATURE_MATMUL_INT8=1\n'
    '        )\n'
    '    else()\n'
    '        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread -Wall -Wextra -pedantic -O3 -Wno-missing-field-initializers")\n'
    '    endif()\n'
    'endif()'
)

# ── Fix 2: architecture-guard the i8mm object-library target ─────────────────
old2 = (
    '# I8MM support: compile kernel_i8mm.cpp with +i8mm\n'
    'set(I8MM_MARCH_FLAG "-march=armv8.2-a+fp16+simd+dotprod+i8mm")\n'
    'add_library(cactus_i8mm_obj OBJECT ${I8MM_SOURCE})\n'
    'target_include_directories(cactus_i8mm_obj PRIVATE ${COMMON_INCLUDES})\n'
    'target_compile_options(cactus_i8mm_obj PRIVATE ${I8MM_MARCH_FLAG})\n'
    'set_target_properties(cactus_i8mm_obj PROPERTIES POSITION_INDEPENDENT_CODE ON)\n'
    'target_compile_definitions(cactus_i8mm_obj PRIVATE CACTUS_COMPILE_I8MM __ARM_FEATURE_MATMUL_INT8=1)\n'
    '\n'
    'set(EXTRA_OBJECTS $<TARGET_OBJECTS:cactus_i8mm_obj>)'
)
new2 = (
    'if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm")\n'
    '    # I8MM support: compile kernel_i8mm.cpp with +i8mm (ARM only)\n'
    '    set(I8MM_MARCH_FLAG "-march=armv8.2-a+fp16+simd+dotprod+i8mm")\n'
    '    add_library(cactus_i8mm_obj OBJECT ${I8MM_SOURCE})\n'
    '    target_include_directories(cactus_i8mm_obj PRIVATE ${COMMON_INCLUDES})\n'
    '    target_compile_options(cactus_i8mm_obj PRIVATE ${I8MM_MARCH_FLAG})\n'
    '    set_target_properties(cactus_i8mm_obj PROPERTIES POSITION_INDEPENDENT_CODE ON)\n'
    '    target_compile_definitions(cactus_i8mm_obj PRIVATE CACTUS_COMPILE_I8MM __ARM_FEATURE_MATMUL_INT8=1)\n'
    '    set(EXTRA_OBJECTS $<TARGET_OBJECTS:cactus_i8mm_obj>)\n'
    'else()\n'
    '    set(EXTRA_OBJECTS "")\n'
    'endif()'
)

# ── Fix 3: only set CACTUS_COMPILE_I8MM on ARM targets ───────────────────────
old3 = (
    'add_library(cactus STATIC ${COMMON_SOURCES} ${EXTRA_OBJECTS})\n'
    'configure_cactus_target(cactus)\n'
    'target_compile_definitions(cactus PRIVATE CACTUS_COMPILE_I8MM)'
)
new3 = (
    'add_library(cactus STATIC ${COMMON_SOURCES} ${EXTRA_OBJECTS})\n'
    'configure_cactus_target(cactus)\n'
    'if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm")\n'
    '    target_compile_definitions(cactus PRIVATE CACTUS_COMPILE_I8MM)\n'
    'endif()'
)

old4 = 'target_compile_definitions(cactus_ffi PRIVATE CACTUS_COMPILE_I8MM)'
new4 = (
    'if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm")\n'
    '    target_compile_definitions(cactus_ffi PRIVATE CACTUS_COMPILE_I8MM)\n'
    'endif()'
)

for old, new, label in [(old1, new1, "CXX_FLAGS"), (old2, new2, "i8mm target"),
                         (old3, new3, "cactus COMPILE_I8MM"), (old4, new4, "cactus_ffi COMPILE_I8MM")]:
    if old not in content:
        print(f"WARNING: pattern not found for fix '{label}' — skipping", file=sys.stderr)
        continue
    content = content.replace(old, new)
    print(f"Applied fix: {label}")

open(path, 'w').write(content)
print("Patch complete.")
