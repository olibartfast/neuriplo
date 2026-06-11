cmake_minimum_required(VERSION 3.10)

# Read version from VERSION file
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/VERSION" PROJECT_VERSION_RAW)
string(STRIP "${PROJECT_VERSION_RAW}" PROJECT_VERSION_RAW)
# Extract semver (strip -dev or other suffixes)
string(REGEX MATCH "^[0-9]+\\.[0-9]+\\.[0-9]+" PROJECT_SEMVER "${PROJECT_VERSION_RAW}")

project(neuriplo VERSION ${PROJECT_SEMVER})

include(cmake/CompileSpeed.cmake)

set(CMAKE_CXX_STANDARD 17)

# Include centralized backend metadata before version/dependency checks.
include(cmake/BackendRegistry.cmake)

# Include centralized version management first
include(cmake/versions.cmake)

# Include dependency validation
include(cmake/DependencyValidation.cmake)

find_package(OpenCV REQUIRED)
find_package(Glog REQUIRED)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
message(STATUS "neuriplo Cmake module path: ${CMAKE_MODULE_PATH}")

# Read DEFAULT_BACKEND from the cache. If not set, provide a default.
if(NOT DEFINED DEFAULT_BACKEND)
    set(DEFAULT_BACKEND "OPENCV_DNN" CACHE STRING "Default backend for inference")
endif()

neuriplo_get_supported_backends(SUPPORTED_BACKENDS)
message(STATUS "Supported backends: ${SUPPORTED_BACKENDS}")

# Validate that all backends have corresponding versions in versions.env
validate_backend_versions()

neuriplo_is_supported_backend("${DEFAULT_BACKEND}" DEFAULT_BACKEND_SUPPORTED)
if(NOT DEFAULT_BACKEND_SUPPORTED)
    if(PROJECT_IS_TOP_LEVEL)
        message(FATAL_ERROR "Unsupported default backend: ${DEFAULT_BACKEND}")
    else()
        message(WARNING "neuriplo: Unsupported default backend: ${DEFAULT_BACKEND}")
    endif()
endif()

message(STATUS "Default backend: ${DEFAULT_BACKEND}")

# Backends compiled into this build. DEFAULT_BACKEND stays the runtime default
# and is always included, so single-backend callers are unaffected.
set(NEURIPLO_BACKENDS "${DEFAULT_BACKEND}" CACHE STRING
    "Semicolon-separated backend ids to compile into this build")
set(NEURIPLO_ENABLED_BACKENDS ${NEURIPLO_BACKENDS})
if(NOT DEFAULT_BACKEND IN_LIST NEURIPLO_ENABLED_BACKENDS)
    list(PREPEND NEURIPLO_ENABLED_BACKENDS "${DEFAULT_BACKEND}")
endif()
list(REMOVE_DUPLICATES NEURIPLO_ENABLED_BACKENDS)
foreach(enabled_backend IN LISTS NEURIPLO_ENABLED_BACKENDS)
    neuriplo_is_supported_backend("${enabled_backend}" enabled_backend_supported)
    if(NOT enabled_backend_supported)
        message(FATAL_ERROR "Unsupported backend in NEURIPLO_BACKENDS: ${enabled_backend}")
    endif()
endforeach()
neuriplo_check_backend_conflicts("${NEURIPLO_ENABLED_BACKENDS}")
message(STATUS "Enabled backends: ${NEURIPLO_ENABLED_BACKENDS}")

# Backends built as dlopen plugins (cmake/PluginBackends.cmake). Declared here
# so dependency validation can cover plugin backends as well; plugins are
# dependency-isolated at runtime (RTLD_LOCAL), so the conflict ban only applies
# to the compiled-in set.
set(NEURIPLO_PLUGIN_BACKENDS "" CACHE STRING
    "Semicolon-separated backend ids to build as dlopen plugins")
set(NEURIPLO_REQUESTED_BACKENDS ${NEURIPLO_ENABLED_BACKENDS} ${NEURIPLO_PLUGIN_BACKENDS})
list(REMOVE_DUPLICATES NEURIPLO_REQUESTED_BACKENDS)
if(NEURIPLO_PLUGIN_BACKENDS)
    message(STATUS "Plugin backends: ${NEURIPLO_PLUGIN_BACKENDS}")
endif()

set(INFER_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/backends)

# Validate dependencies before proceeding
validate_all_dependencies()

# Add source files for inference engines
set(SOURCES
    ${CMAKE_CURRENT_LIST_DIR}/backends/src/BackendRuntimeRegistry.cpp
    ${CMAKE_CURRENT_LIST_DIR}/backends/src/InferenceInterface.cpp
    ${CMAKE_CURRENT_LIST_DIR}/backends/src/InferenceMetadata.cpp
    ${CMAKE_CURRENT_LIST_DIR}/backends/src/ModelRunner.cpp
    ${CMAKE_CURRENT_LIST_DIR}/backends/src/plugin/PluginLoader.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/InferenceBackendSetup.cpp)

include(SelectBackend)

# Create the inference engines library
add_library(neuriplo SHARED ${SOURCES})
target_include_directories(neuriplo PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_include_directories(neuriplo PRIVATE
    ${INFER_ROOT}/src
)

target_link_libraries(neuriplo PRIVATE
    ${OpenCV_LIBS}
    ${GLOG_LIBRARIES}
    ${CMAKE_DL_LIBS}
)

# The registry resolves the default registration by id when several backends
# are compiled in; with one backend this matches the old single-entry behavior.
target_compile_definitions(neuriplo PRIVATE NEURIPLO_DEFAULT_BACKEND="${DEFAULT_BACKEND}")

include(LinkBackend)
include(PluginBackends)
include(SetCompilerFlags)

# Add GoogleTest
# Include directories for tests
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include ${gtest_SOURCE_DIR}/include)
option(BUILD_INFERENCE_ENGINE_TESTS "Build tests (optional)" ON)
if(BUILD_INFERENCE_ENGINE_TESTS)
    include(SetupTests)
endif(BUILD_INFERENCE_ENGINE_TESTS)
