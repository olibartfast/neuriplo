set(TEST_SOURCES
    ${CMAKE_CURRENT_LIST_DIR}/TensorFlowInferTest.cpp
)

find_package(Glog REQUIRED)

# Add cmake modules path to find our custom FindTensorFlow.cmake
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
find_package(TensorFlow REQUIRED)

add_executable(TensorFlowInferTest ${TEST_SOURCES})

target_include_directories(TensorFlowInferTest PRIVATE
    ${CMAKE_SOURCE_DIR}/backends/libtensorflow/src
    ${CMAKE_SOURCE_DIR}/backends/src
    ${CMAKE_SOURCE_DIR}/include
    ${TensorFlow_INCLUDE_DIR}
    ${GLOG_INCLUDE_DIRS}
)

target_link_libraries(TensorFlowInferTest PRIVATE
    neuriplo
    ${TensorFlow_FRAMEWORK_LIBRARY}
    ${TensorFlow_CC_LIBRARY}
    ${GLOG_LIBRARIES}
    glog::glog
    GTest::gtest
    GTest::gtest_main
)

add_test(NAME TensorFlowInferTest COMMAND TensorFlowInferTest)

# Copy model_downloader.py script to the build directory
configure_file(${CMAKE_SOURCE_DIR}/scripts/model_downloader.py  ${CMAKE_CURRENT_BINARY_DIR}/model_downloader.py COPYONLY)

# A supplied SavedModel must fail rather than switching to mock results.
foreach(failure_kind IN ITEMS empty missing invalid)
    set(failure_dir "${CMAKE_CURRENT_BINARY_DIR}/supplied-${failure_kind}")
    file(MAKE_DIRECTORY "${failure_dir}")
    if(failure_kind STREQUAL "empty")
        file(WRITE "${failure_dir}/model_path.txt" "")
    else()
        file(WRITE "${failure_dir}/model_path.txt" "${failure_dir}/saved_model\n")
        if(failure_kind STREQUAL "invalid")
            file(MAKE_DIRECTORY "${failure_dir}/saved_model")
            file(WRITE "${failure_dir}/saved_model/saved_model.pb" "invalid SavedModel")
        endif()
    endif()
    add_test(NAME TensorFlowSuppliedFixture_${failure_kind}
        COMMAND TensorFlowInferTest --gtest_filter=TensorFlowInferTest.BasicInference)
    set_tests_properties(TensorFlowSuppliedFixture_${failure_kind} PROPERTIES
        WORKING_DIRECTORY "${failure_dir}" WILL_FAIL TRUE)
endforeach()
