# Include the CMake module for every enabled backend. Each module appends its
# sources to SOURCES and adds its USE_* compile definition; modules compose, so
# any subset of backends can be enabled together.
foreach(neuriplo_enabled_backend IN LISTS NEURIPLO_ENABLED_BACKENDS)
    neuriplo_include_backend_module("${neuriplo_enabled_backend}")
endforeach()
