# Backend-agnostic tests for the design-pattern scaffolding (State machine,
# ModelRunner bridge, decorators, mock). These depend only on the common
# InferenceInterface contract, so they build under any DEFAULT_BACKEND.

set(PATTERNS_TEST_SOURCES
    ${CMAKE_CURRENT_LIST_DIR}/PatternsTest.cpp
)

find_package(OpenCV REQUIRED)
find_package(Glog REQUIRED)

add_executable(PatternsTest ${PATTERNS_TEST_SOURCES})

target_compile_definitions(PatternsTest PRIVATE
    NEURIPLO_DEFAULT_BACKEND="${DEFAULT_BACKEND}"
)

target_include_directories(PatternsTest PRIVATE
    ${CMAKE_SOURCE_DIR}/backends/src
    ${CMAKE_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
    ${GLOG_INCLUDE_DIRS}
)

target_link_libraries(PatternsTest PRIVATE
    neuriplo
    ${OpenCV_LIBS}
    ${GLOG_LIBRARIES}
    gtest
    gtest_main
    glog::glog
)

add_test(NAME PatternsTest COMMAND PatternsTest)

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
        ${OpenCV_INCLUDE_DIRS}
        ${GLOG_INCLUDE_DIRS}
    )

    target_link_libraries(PluginLoaderTest PRIVATE
        neuriplo
        ${OpenCV_LIBS}
        ${GLOG_LIBRARIES}
        gtest
        gtest_main
        glog::glog
    )

    foreach(neuriplo_plugin_backend IN LISTS NEURIPLO_PLUGIN_BACKENDS)
        string(TOLOWER "${neuriplo_plugin_backend}" plugin_backend_lower)
        add_dependencies(PluginLoaderTest "neuriplo_backend_${plugin_backend_lower}")
    endforeach()

    add_test(NAME PluginLoaderTest COMMAND PluginLoaderTest)
endif()
