set(TEST_SOURCES
    ${CMAKE_CURRENT_LIST_DIR}/LiteRTInferTest.cpp
)

find_package(GTest REQUIRED)

add_executable(LiteRTInferTest ${TEST_SOURCES})

target_include_directories(LiteRTInferTest PRIVATE
    ${CMAKE_SOURCE_DIR}/backends/litert/src
    ${CMAKE_SOURCE_DIR}/backends/src
    ${CMAKE_SOURCE_DIR}/include
    ${GLOG_INCLUDE_DIRS}
)

target_include_directories(LiteRTInferTest SYSTEM PRIVATE
    ${LITERT_DIR}/include
)

target_link_libraries(LiteRTInferTest PRIVATE
    neuriplo
    ${LITERT_LINK_LIBRARIES}
    ${GLOG_LIBRARIES}
    GTest::gtest
    GTest::gtest_main
    glog::glog
)

add_test(NAME LiteRTInferTest COMMAND LiteRTInferTest)
