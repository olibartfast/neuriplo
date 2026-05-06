
message(STATUS "Test enabled")
find_package(GTest REQUIRED)
enable_testing()
neuriplo_add_backend_tests("${DEFAULT_BACKEND}")
