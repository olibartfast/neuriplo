# Server CMakeLists.txt
cmake_minimum_required(VERSION 3.10)

# Find required dependencies
find_package(OpenCV REQUIRED)
find_package(glog REQUIRED)

# Download cpp-httplib if not found
include(FetchContent)
FetchContent_Declare(
  httplib
  GIT_REPOSITORY https://github.com/yhirose/cpp-httplib.git
  GIT_TAG v0.14.3
)
FetchContent_MakeAvailable(httplib)

# Download nlohmann/json if not found
FetchContent_Declare(
  json
  GIT_REPOSITORY https://github.com/nlohmann/json.git
  GIT_TAG v3.11.2
)
FetchContent_MakeAvailable(json)

# Server library
add_library(neuriplo_server INTERFACE)
target_include_directories(neuriplo_server INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/backends/src
    ${CMAKE_SOURCE_DIR}/include
)
target_link_libraries(neuriplo_server INTERFACE
    httplib::httplib
    nlohmann_json::nlohmann_json
    ${OpenCV_LIBS}
    glog::glog
)

# Server executable
add_executable(neuriplo_server_exec neuriplo_server.cpp)
set_target_properties(neuriplo_server_exec PROPERTIES OUTPUT_NAME neuriplo_server)
target_link_libraries(neuriplo_server_exec PRIVATE
    neuriplo_server
    inference_interface
    ${OpenCV_LIBS}
    glog::glog
)

# Install server executable
install(TARGETS neuriplo_server_exec
    RUNTIME DESTINATION bin
)

# Install server headers
install(FILES 
    src/InferenceServer.hpp
    src/Serialization.hpp
    DESTINATION include/neuriplo/server
)
