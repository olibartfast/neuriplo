set(TEST_SOURCES
    ${CMAKE_CURRENT_LIST_DIR}/DALIInferTest.cpp
)

find_package(GTest REQUIRED)

add_executable(DALIInferTest ${TEST_SOURCES})

target_include_directories(DALIInferTest PRIVATE
    ${CMAKE_SOURCE_DIR}/backends/dali/src
    ${CMAKE_SOURCE_DIR}/backends/src
    ${CMAKE_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
    ${GLOG_INCLUDE_DIRS}
)

target_include_directories(DALIInferTest SYSTEM PRIVATE
    ${DALI_DIR}/include
)

target_link_directories(DALIInferTest PRIVATE ${DALI_DIR})

target_link_libraries(DALIInferTest PRIVATE
    neuriplo
    dali
    dali_operators
    ${OpenCV_LIBS}
    ${GLOG_LIBRARIES}
    gtest
    gtest_main
    glog::glog
)

add_test(NAME DALIInferTest COMMAND DALIInferTest)
