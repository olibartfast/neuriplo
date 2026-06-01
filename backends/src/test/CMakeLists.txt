# Backend-agnostic tests for the design-pattern scaffolding (State machine,
# ModelRunner bridge, decorators, mock). These depend only on the common
# InferenceInterface contract, so they build under any DEFAULT_BACKEND.

set(PATTERNS_TEST_SOURCES
    ${CMAKE_CURRENT_LIST_DIR}/PatternsTest.cpp
)

find_package(OpenCV REQUIRED)
find_package(Glog REQUIRED)

add_executable(PatternsTest ${PATTERNS_TEST_SOURCES})

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
