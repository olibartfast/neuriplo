
set(TEST_SOURCES
    ${CMAKE_CURRENT_LIST_DIR}/OCVDNNInferTest.cpp
    ${CMAKE_CURRENT_LIST_DIR}/CompatibilityTest.cpp
)

find_package(OpenCV REQUIRED)
if(DEFINED OPENCV_TEST_VERSION AND NOT OpenCV_VERSION VERSION_EQUAL OPENCV_TEST_VERSION)
    message(FATAL_ERROR "Compatibility gate expected OpenCV ${OPENCV_TEST_VERSION}, found ${OpenCV_VERSION}")
endif()
find_package(Glog REQUIRED)

add_executable(OCVDNNInferTest ${TEST_SOURCES})
configure_file(${CMAKE_CURRENT_LIST_DIR}/fixtures/relu.onnx
    ${CMAKE_CURRENT_BINARY_DIR}/relu.onnx COPYONLY)
target_compile_definitions(OCVDNNInferTest PRIVATE
    OPENCV_COMPAT_FIXTURE="${CMAKE_CURRENT_BINARY_DIR}/relu.onnx")

target_include_directories(OCVDNNInferTest PRIVATE
    ${CMAKE_SOURCE_DIR}/backends/opencv-dnn/src
    ${CMAKE_SOURCE_DIR}/backends/src
    ${CMAKE_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
    ${GLOG_INCLUDE_DIRS}
)


target_link_libraries(OCVDNNInferTest PRIVATE
    neuriplo
    ${TORCH_LIBRARIES}
    ${OpenCV_LIBS}
    ${GLOG_LIBRARIES}
    GTest::gtest
    GTest::gtest_main
    glog::glog
)

add_test(NAME OCVDNNInferTest COMMAND OCVDNNInferTest)

# A supplied classifier must never be replaced with mock output. These runs
# must return a normal GTest failure; a success means the fallback regressed.
foreach(failure_kind IN ITEMS empty missing invalid)
    set(failure_dir "${CMAKE_CURRENT_BINARY_DIR}/supplied-${failure_kind}")
    file(MAKE_DIRECTORY "${failure_dir}")
    if(failure_kind STREQUAL "empty")
        file(WRITE "${failure_dir}/model_path.txt" "")
    else()
        file(WRITE "${failure_dir}/model_path.txt" "${failure_dir}/classifier.onnx\n")
        if(failure_kind STREQUAL "invalid")
            file(WRITE "${failure_dir}/classifier.onnx" "invalid ONNX")
        endif()
    endif()
    add_test(NAME OpenCVSuppliedFixture_${failure_kind}
        COMMAND OCVDNNInferTest --gtest_filter=OCVDNNInferTest.BasicInference)
    set_tests_properties(OpenCVSuppliedFixture_${failure_kind} PROPERTIES
        WORKING_DIRECTORY "${failure_dir}" WILL_FAIL TRUE)
endforeach()


# Copy scripts to the build directory
configure_file(${CMAKE_CURRENT_LIST_DIR}/generate_model.sh  ${CMAKE_CURRENT_BINARY_DIR}/generate_model.sh FILE_PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ)
configure_file(${CMAKE_CURRENT_LIST_DIR}/export_torchvision_classifier.py  ${CMAKE_CURRENT_BINARY_DIR}/export_torchvision_classifier.py COPYONLY)

