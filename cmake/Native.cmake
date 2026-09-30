# NATIVE backend module — the first-party engine, no external SDK.
#
# Unlike the third-party modules, this one appends nothing to SOURCES and adds
# no USE_* compile definition: the engine is a first-party static library built
# separately so it stays decoupled from the backend abstraction layer. The
# backend adapter that links it arrives later.

add_subdirectory("${CMAKE_CURRENT_LIST_DIR}/../engine" engine)
