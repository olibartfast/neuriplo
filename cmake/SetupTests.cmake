
message(STATUS "Test enabled")
find_package(GTest REQUIRED)
enable_testing()

# Backend-agnostic pattern tests (Bridge/Decorator/State/Mock). Built regardless
# of which backend is selected, since they only use the common interface.
add_subdirectory("${CMAKE_SOURCE_DIR}/backends/src/test")

neuriplo_add_backend_tests("${DEFAULT_BACKEND}")
