set(TEST_SOURCES
    ${CMAKE_CURRENT_LIST_DIR}/ExecuTorchInferTest.cpp
)

find_package(Glog REQUIRED)

list(APPEND CMAKE_PREFIX_PATH "${EXECUTORCH_DIR}")
if(NOT TARGET executorch)
    find_package(executorch REQUIRED CONFIG)
endif()

add_executable(ExecuTorchInferTest ${TEST_SOURCES})

target_include_directories(ExecuTorchInferTest PRIVATE
    ${CMAKE_SOURCE_DIR}/backends/executorch/src
    ${CMAKE_SOURCE_DIR}/backends/src
    ${CMAKE_SOURCE_DIR}/include
    ${GLOG_INCLUDE_DIRS}
)

# Runtime and operator registration must have one owner (libneuriplo).
# Linking SDK archives into this executable duplicates static registrations.
target_include_directories(ExecuTorchInferTest SYSTEM PRIVATE ${EXECUTORCH_INCLUDE_DIRS})
target_compile_definitions(ExecuTorchInferTest PRIVATE
    $<TARGET_PROPERTY:executorch_core,INTERFACE_COMPILE_DEFINITIONS>)

target_link_libraries(ExecuTorchInferTest PRIVATE
    neuriplo
    GTest::gtest
    GTest::gtest_main
    glog::glog
)

add_test(NAME ExecuTorchInferTest COMMAND ExecuTorchInferTest)

configure_file(${CMAKE_CURRENT_LIST_DIR}/generate_model.sh
    ${CMAKE_CURRENT_BINARY_DIR}/generate_model.sh COPYONLY FILE_PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ)
configure_file(${CMAKE_CURRENT_LIST_DIR}/export_executorch_classifier.py
    ${CMAKE_CURRENT_BINARY_DIR}/export_executorch_classifier.py COPYONLY)
