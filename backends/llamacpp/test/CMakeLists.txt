set(TEST_SOURCES
    ${CMAKE_CURRENT_LIST_DIR}/LlamaCppInferTest.cpp
)

find_package(OpenCV REQUIRED)
find_package(Glog REQUIRED)

add_executable(LlamaCppInferTest ${TEST_SOURCES})

target_include_directories(LlamaCppInferTest PRIVATE
    ${CMAKE_SOURCE_DIR}/backends/llamacpp/src
    ${CMAKE_SOURCE_DIR}/backends/src
    ${CMAKE_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
    ${GLOG_INCLUDE_DIRS}
    ${LLAMACPP_DIR}/include
)

target_link_libraries(LlamaCppInferTest PRIVATE
    neuriplo
    ${OpenCV_LIBS}
    ${GLOG_LIBRARIES}
    ${LLAMACPP_DIR}/lib/libllama.so
    ${LLAMACPP_DIR}/lib/libggml.so
    gtest
    gtest_main
    glog::glog
)

add_test(NAME LlamaCppInferTest COMMAND LlamaCppInferTest)

configure_file(${CMAKE_CURRENT_LIST_DIR}/generate_model.sh
               ${CMAKE_CURRENT_BINARY_DIR}/generate_model.sh
               FILE_PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ)
configure_file(${CMAKE_CURRENT_LIST_DIR}/create_test_model.py
               ${CMAKE_CURRENT_BINARY_DIR}/create_test_model.py COPYONLY)
