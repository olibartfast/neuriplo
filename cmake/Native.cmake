# NATIVE backend module — first-party engine, no external SDK.
#
# Unlike the third-party modules, this one appends nothing to SOURCES and adds
# no USE_* compile definition: the engine is a first-party static library built
# separately so it can stay decoupled from the backend abstraction layer. The
# backend adapter that links it arrives in Group 6.
#
# [T-5] order note: the registry's VERSION_VAR for NATIVE is
# NEURIPLO_NO_EXTERNAL_SDK, which validate_backend_versions() accepts as a
# declared-null case before this module is ever included.

add_subdirectory("${CMAKE_CURRENT_LIST_DIR}/../engine" engine)
