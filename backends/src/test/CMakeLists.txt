# Backend-agnostic tests for the design-pattern scaffolding (State machine,
# ModelRunner bridge, decorators, mock). These depend only on the common
# InferenceInterface contract, so they build under any DEFAULT_BACKEND.

set(PATTERNS_TEST_SOURCES
    ${CMAKE_CURRENT_LIST_DIR}/PatternsTest.cpp
)

find_package(Glog REQUIRED)

add_executable(PatternsTest ${PATTERNS_TEST_SOURCES})

target_compile_definitions(PatternsTest PRIVATE
    NEURIPLO_DEFAULT_BACKEND="${DEFAULT_BACKEND}"
)

target_include_directories(PatternsTest PRIVATE
    ${CMAKE_SOURCE_DIR}/backends/src
    ${CMAKE_SOURCE_DIR}/include
    ${GLOG_INCLUDE_DIRS}
)

target_link_libraries(PatternsTest PRIVATE
    neuriplo
    ${GLOG_LIBRARIES}
    GTest::gtest
    GTest::gtest_main
    glog::glog
)

add_test(NAME PatternsTest COMMAND PatternsTest)

# Compile coverage for BackendTestTemplate.hpp / MockInferenceInterface.hpp.
# They are kept for docs/REFACTOR_DESIGN_PATTERNS.md S6.3 but no other
# translation unit includes them, so nothing type-checked them until now.
#
# GTest::gmock comes from GTestConfig.cmake's OPTIONAL include of
# GMockTargets.cmake, i.e. from libgmock-dev alongside libgtest-dev. CMake's own
# FindGTest module only defines the target from 3.23, so on an older CMake it is
# absent even when gmock is installed.
#
# Debian/Ubuntu split GoogleTest: libgtest-dev installs the gtest binaries but
# ships the full sources (googlemock included) under /usr/src/googletest, while
# only libgmock-dev installs the gmock binaries. When gmock is missing, build it
# from those sources rather than failing configure over a package split -- the
# header still gets its compile check, which is the point of the target.
# Only when building neuriplo itself, and only if nothing else has already
# defined these targets: a consumer that vendors its own GoogleTest would
# otherwise get a duplicate-target error from the add_subdirectory below.
if(NOT TARGET GTest::gmock AND NOT TARGET gmock AND CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    find_path(NEURIPLO_GOOGLETEST_SOURCE_DIR
        NAMES googlemock/CMakeLists.txt
        PATHS /usr/src/googletest
        NO_DEFAULT_PATH)

    if(NEURIPLO_GOOGLETEST_SOURCE_DIR)
        message(STATUS "GTest::gmock not installed; building GoogleMock from ${NEURIPLO_GOOGLETEST_SOURCE_DIR}")
        set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
        add_subdirectory(${NEURIPLO_GOOGLETEST_SOURCE_DIR}
                         ${CMAKE_CURRENT_BINARY_DIR}/_googletest EXCLUDE_FROM_ALL)
        # That add_subdirectory defines its own GTest::gmock alias, so testing
        # for the target below can no longer tell the two GoogleTests apart.
        # Record which route was taken instead.
        set(NEURIPLO_GMOCK_FROM_SOURCE TRUE)
    endif()
endif()

# Link one GoogleTest build or the other, never a mix: the source-built gmock
# carries its own gtest, so pairing it with the imported GTest::gtest would put
# two copies of gtest in the same binary.
#
# If neither route yields gmock this only warns: it must not turn a normal build
# of this repo into one that needs a package it did not need before. CI
# configures with -DNEURIPLO_REQUIRE_GMOCK=ON, where the Dockerfiles do install
# libgmock-dev, so the check still cannot be skipped unnoticed there.
if(NEURIPLO_GMOCK_FROM_SOURCE)
    # gmock_main pulls the source-built gtest with it; GTest::gtest here would
    # be the imported system one, and the binary would get two gtests.
    set(NEURIPLO_GMOCK_LIBS gmock_main)
elseif(TARGET GTest::gmock)
    set(NEURIPLO_GMOCK_LIBS GTest::gtest GTest::gtest_main GTest::gmock)
else()
    set(NEURIPLO_GMOCK_LIBS "")
endif()

if(NEURIPLO_GMOCK_LIBS)
    add_executable(TestTemplateCompileTest ${CMAKE_CURRENT_LIST_DIR}/TestTemplateCompileTest.cpp)

    target_include_directories(TestTemplateCompileTest PRIVATE
        ${CMAKE_SOURCE_DIR}/backends/src
        ${CMAKE_SOURCE_DIR}/include
        ${GLOG_INCLUDE_DIRS}
    )

    target_link_libraries(TestTemplateCompileTest PRIVATE
        neuriplo
        ${GLOG_LIBRARIES}
        ${NEURIPLO_GMOCK_LIBS}
        glog::glog
    )

    add_test(NAME TestTemplateCompileTest COMMAND TestTemplateCompileTest)
elseif(NEURIPLO_REQUIRE_GMOCK)
    message(FATAL_ERROR
        "GTest::gmock not found and no GoogleTest sources to build it from, so "
        "BackendTestTemplate.hpp cannot be compile-checked, and NEURIPLO_REQUIRE_GMOCK "
        "is ON. Install libgmock-dev alongside libgtest-dev (CMake 3.23+ is needed for "
        "the module-mode target), point -DCMAKE_PREFIX_PATH at a GoogleTest that "
        "provides gmock, or configure with -DNEURIPLO_REQUIRE_GMOCK=OFF.")
else()
    message(WARNING "GTest::gmock not found: BackendTestTemplate.hpp will not be compile-checked. "
                    "Install libgmock-dev (or the gmock half of your GoogleTest package) to enable it.")
endif()

# Plugin loader tests: only meaningful when this build produces plugins.
if(NEURIPLO_PLUGIN_BACKENDS)
    add_executable(PluginLoaderTest ${CMAKE_CURRENT_LIST_DIR}/PluginLoaderTest.cpp)

    target_compile_definitions(PluginLoaderTest PRIVATE
        NEURIPLO_DEFAULT_BACKEND="${DEFAULT_BACKEND}"
        NEURIPLO_TEST_PLUGIN_DIR="${NEURIPLO_PLUGIN_OUTPUT_DIR}"
    )

    target_include_directories(PluginLoaderTest PRIVATE
        ${CMAKE_SOURCE_DIR}/backends/src
        ${CMAKE_SOURCE_DIR}/include
        ${GLOG_INCLUDE_DIRS}
    )

    target_link_libraries(PluginLoaderTest PRIVATE
        neuriplo
        ${GLOG_LIBRARIES}
        GTest::gtest
        GTest::gtest_main
        glog::glog
    )

    foreach(neuriplo_plugin_backend IN LISTS NEURIPLO_PLUGIN_BACKENDS)
        string(TOLOWER "${neuriplo_plugin_backend}" plugin_backend_lower)
        add_dependencies(PluginLoaderTest "neuriplo_backend_${plugin_backend_lower}")
    endforeach()

    add_test(NAME PluginLoaderTest COMMAND PluginLoaderTest)
endif()

# Plugin ABI contract suite (specs/2026-09-22-plugin-abi-loader-hardening).
# Built in every configuration: the fixtures are first-party C against
# include/neuriplo/plugin_abi.h with no vendor SDK, so plugin failure behavior
# is part of the default gate instead of an opt-in extra.
set(NEURIPLO_FIXTURE_PLUGIN_DIR "${CMAKE_BINARY_DIR}/plugin_fixtures")
# The duplicate-id fixture lives apart so a directory scan of the main fixture
# directory never races it against the plugin it duplicates.
set(NEURIPLO_FIXTURE_DUPLICATE_DIR "${NEURIPLO_FIXTURE_PLUGIN_DIR}/duplicate")

function(neuriplo_add_fixture_plugin name backend_id api_defect output_dir)
    set(target "neuriplo_backend_fixture_${name}")
    add_library(${target} MODULE ${CMAKE_CURRENT_LIST_DIR}/plugin_fixtures/fixture_backend.c)
    target_include_directories(${target} PRIVATE ${CMAKE_SOURCE_DIR}/include)
    target_compile_definitions(${target} PRIVATE
        FIXTURE_BACKEND_ID="${backend_id}"
        FIXTURE_API_DEFECT=${api_defect}
    )
    # The loader scans for libneuriplo_backend_* on every platform.
    set_target_properties(${target} PROPERTIES
        PREFIX "lib"
        C_VISIBILITY_PRESET hidden
        LIBRARY_OUTPUT_DIRECTORY "${output_dir}"
        RUNTIME_OUTPUT_DIRECTORY "${output_dir}"
    )
    foreach(config IN ITEMS DEBUG RELEASE RELWITHDEBINFO MINSIZEREL)
        set_target_properties(${target} PROPERTIES
            LIBRARY_OUTPUT_DIRECTORY_${config} "${output_dir}"
            RUNTIME_OUTPUT_DIRECTORY_${config} "${output_dir}"
        )
    endforeach()
    set(NEURIPLO_FIXTURE_TARGETS ${NEURIPLO_FIXTURE_TARGETS} ${target} PARENT_SCOPE)
endfunction()

set(NEURIPLO_FIXTURE_TARGETS "")
neuriplo_add_fixture_plugin(good            FIXTURE_GOOD             0 "${NEURIPLO_FIXTURE_PLUGIN_DIR}")
neuriplo_add_fixture_plugin(scripted        FIXTURE_SCRIPTED         0 "${NEURIPLO_FIXTURE_PLUGIN_DIR}")
neuriplo_add_fixture_plugin(wrong_abi       FIXTURE_WRONG_ABI        1 "${NEURIPLO_FIXTURE_PLUGIN_DIR}")
neuriplo_add_fixture_plugin(null_entry      FIXTURE_NULL_ENTRY       2 "${NEURIPLO_FIXTURE_PLUGIN_DIR}")
neuriplo_add_fixture_plugin(incomplete      FIXTURE_INCOMPLETE       3 "${NEURIPLO_FIXTURE_PLUGIN_DIR}")
neuriplo_add_fixture_plugin(missing_entry   FIXTURE_MISSING_ENTRY    4 "${NEURIPLO_FIXTURE_PLUGIN_DIR}")
neuriplo_add_fixture_plugin(duplicate       FIXTURE_GOOD             0 "${NEURIPLO_FIXTURE_DUPLICATE_DIR}")

add_executable(PluginAbiContractTest ${CMAKE_CURRENT_LIST_DIR}/PluginAbiContractTest.cpp)

target_compile_definitions(PluginAbiContractTest PRIVATE
    NEURIPLO_DEFAULT_BACKEND="${DEFAULT_BACKEND}"
    NEURIPLO_FIXTURE_PLUGIN_DIR="${NEURIPLO_FIXTURE_PLUGIN_DIR}"
    NEURIPLO_FIXTURE_DUPLICATE_DIR="${NEURIPLO_FIXTURE_DUPLICATE_DIR}"

    NEURIPLO_FIXTURE_PLUGIN_SUFFIX="${CMAKE_SHARED_MODULE_SUFFIX}"
)

target_include_directories(PluginAbiContractTest PRIVATE
    ${CMAKE_SOURCE_DIR}/backends/src
    ${CMAKE_SOURCE_DIR}/include
    ${GLOG_INCLUDE_DIRS}
)

find_package(Threads REQUIRED)

target_link_libraries(PluginAbiContractTest PRIVATE
    neuriplo
    ${GLOG_LIBRARIES}
    GTest::gtest
    GTest::gtest_main
    glog::glog
    Threads::Threads
    ${CMAKE_DL_LIBS}
)

add_dependencies(PluginAbiContractTest ${NEURIPLO_FIXTURE_TARGETS})

# One process per test case: the loader's plugin table is process-global and
# never unloads, so cases must not observe each other's loads.
include(GoogleTest)
gtest_discover_tests(PluginAbiContractTest DISCOVERY_TIMEOUT 60)

# ---------------------------------------------------------------------------
# Consumer C ABI acceptance suite (specs/2026-09-25-consumer-c-abi). Built in
# every configuration against the same SDK-free fixtures as the plugin suite.
# Owned by the specifier; read-only to every implementer.
# ---------------------------------------------------------------------------

if(MSVC)
    set(NEURIPLO_CAPI_STRICT_FLAGS /W4 /WX)
else()
    set(NEURIPLO_CAPI_STRICT_FLAGS -Wall -Wextra -Wpedantic -Werror)
endif()

# [V-1], [V-11]: neuriplo_c.h compiles as strict C99 beside plugin_abi.h, with
# struct layouts, enum widths, enumerator values, and every function type
# pinned. Build-time: a violation fails the build, not a test.
add_library(neuriplo_capi_header_check_c OBJECT ${CMAKE_CURRENT_LIST_DIR}/CApiHeaderCheck.c)
set_target_properties(neuriplo_capi_header_check_c PROPERTIES
    C_STANDARD 99
    C_STANDARD_REQUIRED ON
    C_EXTENSIONS OFF
)
target_include_directories(neuriplo_capi_header_check_c PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_compile_options(neuriplo_capi_header_check_c PRIVATE ${NEURIPLO_CAPI_STRICT_FLAGS})

# [V-1], [V-10]: the same headers plus the C++ wrapper as strict C++17.
add_library(neuriplo_capi_header_check_cxx OBJECT ${CMAKE_CURRENT_LIST_DIR}/CApiHeaderCheck.cpp)
set_target_properties(neuriplo_capi_header_check_cxx PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
)
target_include_directories(neuriplo_capi_header_check_cxx PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_compile_options(neuriplo_capi_header_check_cxx PRIVATE ${NEURIPLO_CAPI_STRICT_FLAGS})

# [R-1], [V-10]: include audit. neuriplo_c.h may include only <stddef.h> and
# <stdint.h>; neuriplo.hpp only "neuriplo_c.h" and standard C++ headers.
add_custom_target(neuriplo_capi_include_audit ALL
    COMMAND ${CMAKE_COMMAND}
        -DHEADER=${CMAKE_SOURCE_DIR}/include/neuriplo/neuriplo_c.h
        -DALLOWED_ANGLE=stddef.h,stdint.h
        -P ${CMAKE_CURRENT_LIST_DIR}/CheckHeaderIncludes.cmake
    COMMAND ${CMAKE_COMMAND}
        -DHEADER=${CMAKE_SOURCE_DIR}/include/neuriplo/neuriplo.hpp
        -DALLOWED_QUOTED=neuriplo_c.h
        -P ${CMAKE_CURRENT_LIST_DIR}/CheckHeaderIncludes.cmake
    COMMENT "Auditing consumer header includes"
    VERBATIM
)

set(NEURIPLO_CAPI_TEST_DEFINITIONS
    NEURIPLO_DEFAULT_BACKEND="${DEFAULT_BACKEND}"
    NEURIPLO_FIXTURE_PLUGIN_DIR="${NEURIPLO_FIXTURE_PLUGIN_DIR}"
    NEURIPLO_FIXTURE_DUPLICATE_DIR="${NEURIPLO_FIXTURE_DUPLICATE_DIR}"
    NEURIPLO_FIXTURE_PLUGIN_SUFFIX="${CMAKE_SHARED_MODULE_SUFFIX}"
)

# Plain-C contract suite: compiled as C99, linked with the C linker against
# libneuriplo only (plus threads and the dynamic loader for fixture counters).
add_executable(CApiContractTest ${CMAKE_CURRENT_LIST_DIR}/CApiContractTest.c)
set_target_properties(CApiContractTest PROPERTIES
    C_STANDARD 99
    C_STANDARD_REQUIRED ON
    C_EXTENSIONS OFF
    LINKER_LANGUAGE C
)
target_compile_definitions(CApiContractTest PRIVATE ${NEURIPLO_CAPI_TEST_DEFINITIONS})
target_link_libraries(CApiContractTest PRIVATE
    neuriplo
    Threads::Threads
    ${CMAKE_DL_LIBS}
)
add_dependencies(CApiContractTest ${NEURIPLO_FIXTURE_TARGETS})

# One ctest test per case, one process each (the plugin table is process-global
# and a log callback must precede the first scan). The case table in the source
# is the single list: every `CAPI_CASE(Group, Name),` line becomes the test
# `CApi.Group.Name`, so `ctest -R "^CApi\.Lifecycle\."` selects a group.
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${CMAKE_CURRENT_LIST_DIR}/CApiContractTest.c)
file(STRINGS ${CMAKE_CURRENT_LIST_DIR}/CApiContractTest.c neuriplo_capi_case_lines
     REGEX "^[ ]*CAPI_CASE\\([A-Za-z0-9]+, [A-Za-z0-9]+\\),")
if(NOT neuriplo_capi_case_lines)
    message(FATAL_ERROR "No CAPI_CASE entries found in CApiContractTest.c")
endif()
foreach(neuriplo_capi_case_line IN LISTS neuriplo_capi_case_lines)
    string(REGEX REPLACE "^[ ]*CAPI_CASE\\(([A-Za-z0-9]+), ([A-Za-z0-9]+)\\),.*$" "\\1.\\2"
           neuriplo_capi_case "${neuriplo_capi_case_line}")
    add_test(NAME CApi.${neuriplo_capi_case} COMMAND CApiContractTest ${neuriplo_capi_case})
    # An inherited NEURIPLO_PLUGIN_DIR would add plugins the cases do not expect.
    set_tests_properties(CApi.${neuriplo_capi_case} PROPERTIES
        ENVIRONMENT "NEURIPLO_PLUGIN_DIR="
        TIMEOUT 300
    )
endforeach()

# C++ wrapper suite [V-10]: GoogleTest, includes only neuriplo/neuriplo.hpp and
# links no glog -- the application never has to know neuriplo uses it.
add_executable(CApiWrapperTest ${CMAKE_CURRENT_LIST_DIR}/CApiWrapperTest.cpp)
target_compile_definitions(CApiWrapperTest PRIVATE ${NEURIPLO_CAPI_TEST_DEFINITIONS})
target_link_libraries(CApiWrapperTest PRIVATE
    neuriplo
    GTest::gtest
    GTest::gtest_main
    ${CMAKE_DL_LIBS}
)
add_dependencies(CApiWrapperTest ${NEURIPLO_FIXTURE_TARGETS} neuriplo_capi_include_audit)
gtest_discover_tests(CApiWrapperTest
    DISCOVERY_TIMEOUT 60
    PROPERTIES ENVIRONMENT "NEURIPLO_PLUGIN_DIR="
)
