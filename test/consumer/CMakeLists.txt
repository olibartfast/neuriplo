# Consumer proof ([T-14], [V-12]): builds a C program and a C++ program
# against an INSTALLED neuriplo package, exactly as a third-party project
# would. Standalone on purpose -- the root CMakeLists never configures it, and
# it sees nothing of the source tree. Driven by run.sh:
#
#   cmake -S test/consumer -B <build> -DCMAKE_PREFIX_PATH=<install prefix>

cmake_minimum_required(VERSION 3.10)
project(neuriplo_consumer C CXX)

find_package(neuriplo CONFIG REQUIRED)

add_executable(consumer_c consumer.c)
set_target_properties(consumer_c PROPERTIES C_STANDARD 99 C_STANDARD_REQUIRED ON C_EXTENSIONS OFF)
target_link_libraries(consumer_c PRIVATE neuriplo::neuriplo)

add_executable(consumer_cpp consumer.cpp)
set_target_properties(consumer_cpp PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON CXX_EXTENSIONS OFF)
target_link_libraries(consumer_cpp PRIVATE neuriplo::neuriplo)
