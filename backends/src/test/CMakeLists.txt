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
