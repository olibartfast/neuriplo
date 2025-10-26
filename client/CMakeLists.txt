# Client CMakeLists.txt
cmake_minimum_required(VERSION 3.10)

# Find required dependencies
find_package(OpenCV REQUIRED)
find_package(glog REQUIRED)

# Use the same httplib and json from server
include(FetchContent)
FetchContent_Declare(
  httplib
  GIT_REPOSITORY https://github.com/yhirose/cpp-httplib.git
  GIT_TAG v0.14.3
)
FetchContent_MakeAvailable(httplib)

FetchContent_Declare(
  json
  GIT_REPOSITORY https://github.com/nlohmann/json.git
  GIT_TAG v3.11.2
)
FetchContent_MakeAvailable(json)

# Client library
add_library(neuriplo_client INTERFACE)
target_include_directories(neuriplo_client INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/backends/src
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/server/src  # For Serialization.hpp
)
target_link_libraries(neuriplo_client INTERFACE
    httplib::httplib
    nlohmann_json::nlohmann_json
    ${OpenCV_LIBS}
    glog::glog
)

# Client example executable
add_executable(neuriplo_client_example neuriplo_client_example.cpp)
target_link_libraries(neuriplo_client_example PRIVATE
    neuriplo_client
    inference_interface
    ${OpenCV_LIBS}
    glog::glog
)

# Install client example
install(TARGETS neuriplo_client_example
    RUNTIME DESTINATION bin
)

# Install client headers
install(FILES 
    src/InferenceClient.hpp
    DESTINATION include/neuriplo/client
)
