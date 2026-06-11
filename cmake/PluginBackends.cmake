# Builds each backend listed in NEURIPLO_PLUGIN_BACKENDS as a standalone
# dlopen()-able shared library (libneuriplo_backend_<id>.so) exporting only the
# C ABI entry point from include/neuriplo/plugin_abi.h. A backend may be both
# compiled in (NEURIPLO_BACKENDS) and built as a plugin; at runtime the
# compiled-in registration wins id collisions.

set(NEURIPLO_PLUGIN_BACKENDS "" CACHE STRING
    "Semicolon-separated backend ids to build as dlopen plugins")

set(NEURIPLO_PLUGIN_OUTPUT_DIR "${CMAKE_BINARY_DIR}/plugins")

# neuriplo may be built standalone or via add_subdirectory (e.g. from
# neuriplo-kserve-runtime), so resolve sources against this repo's root
# rather than CMAKE_SOURCE_DIR.
set(NEURIPLO_PLUGIN_REPO_ROOT "${CMAKE_CURRENT_LIST_DIR}/..")

function(neuriplo_add_backend_plugin backend)
    neuriplo_is_supported_backend("${backend}" plugin_backend_supported)
    if(NOT plugin_backend_supported)
        message(FATAL_ERROR "Unsupported backend in NEURIPLO_PLUGIN_BACKENDS: ${backend}")
    endif()

    neuriplo_get_backend_property("${backend}" SOURCE_DIR backend_source_dir)
    neuriplo_get_backend_property("${backend}" FACTORY_HEADER PLUGIN_FACTORY_HEADER)
    neuriplo_get_backend_property("${backend}" FACTORY_CLASS PLUGIN_FACTORY_CLASS)
    neuriplo_get_backend_property("${backend}" DISPLAY_NAME PLUGIN_DISPLAY_NAME)
    neuriplo_get_backend_property("${backend}" FORCE_GPU PLUGIN_FORCE_GPU)
    set(PLUGIN_BACKEND_ID "${backend}")

    string(TOLOWER "${backend}" backend_lower)
    set(target "neuriplo_backend_${backend_lower}")

    set(entry_file "${CMAKE_BINARY_DIR}/plugin_entries/${backend_lower}_entry.cpp")
    configure_file("${NEURIPLO_PLUGIN_REPO_ROOT}/cmake/plugin_entry.cpp.in" "${entry_file}" @ONLY)

    file(GLOB backend_sources "${NEURIPLO_PLUGIN_REPO_ROOT}/${backend_source_dir}/*.cpp")

    add_library(${target} MODULE
        ${entry_file}
        ${backend_sources}
        ${NEURIPLO_PLUGIN_REPO_ROOT}/backends/src/InferenceInterface.cpp
        ${NEURIPLO_PLUGIN_REPO_ROOT}/backends/src/InferenceMetadata.cpp
    )

    target_include_directories(${target} PRIVATE
        ${NEURIPLO_PLUGIN_REPO_ROOT}/include
        ${NEURIPLO_PLUGIN_REPO_ROOT}/backends/src
        ${NEURIPLO_PLUGIN_REPO_ROOT}/backends/src/plugin
        ${NEURIPLO_PLUGIN_REPO_ROOT}/${backend_source_dir}
        ${OpenCV_INCLUDE_DIRS}
        ${GLOG_INCLUDE_DIRS}
    )
    target_link_libraries(${target} PRIVATE
        ${OpenCV_LIBS}
        ${GLOG_LIBRARIES}
        glog::glog
    )

    # Only the C entry point is exported: hidden visibility plus a version
    # script keep every framework and C++ symbol private to the plugin.
    set_target_properties(${target} PROPERTIES
        CXX_VISIBILITY_PRESET hidden
        VISIBILITY_INLINES_HIDDEN ON
        LIBRARY_OUTPUT_DIRECTORY "${NEURIPLO_PLUGIN_OUTPUT_DIR}"
        PREFIX "lib"
    )
    target_link_options(${target} PRIVATE
        "-Wl,--version-script=${NEURIPLO_PLUGIN_REPO_ROOT}/cmake/plugin_export.map"
        "-Wl,--exclude-libs,ALL"
    )

    neuriplo_link_backend_to(${target} "${backend}")
endfunction()

foreach(neuriplo_plugin_backend IN LISTS NEURIPLO_PLUGIN_BACKENDS)
    neuriplo_add_backend_plugin("${neuriplo_plugin_backend}")
endforeach()
