set(TEST_SOURCES
${CMAKE_CURRENT_LIST_DIR}/LibtorchInferTest.cpp
)

add_executable(LibtorchInferTest ${TEST_SOURCES})
target_include_directories(${PROJECT_NAME} PRIVATE  ${CMAKE_CURRENT_LIST_DIR}/../src)
target_link_libraries(LibtorchInferTest PRIVATE
    InferenceEngines
    ${TORCH_LIBRARIES}
    gtest
    gtest_main
)



add_test(NAME LibtorchInferTest COMMAND LibtorchInferTest)