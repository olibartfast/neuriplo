# Build-time include audit for the consumer headers ([R-1], [V-10]).
#
#   cmake -DHEADER=<file> -DALLOWED_QUOTED=a,b -DALLOWED_ANGLE=x,y -P CheckHeaderIncludes.cmake
#
# (Comma-separated, so the lists survive add_custom_command unsplit.)
#
# Every #include "..." must be in ALLOWED_QUOTED. Every #include <...> must be
# in ALLOWED_ANGLE when that list is given; otherwise it must be a standard
# header -- a bare name with no '/' and no '.' (so no glog/..., opencv2/...,
# neuriplo/..., or *.hpp). Owned by the specifier.

cmake_minimum_required(VERSION 3.10) # IN_LIST (CMP0057) in -P mode

if(NOT HEADER OR NOT EXISTS "${HEADER}")
    message(FATAL_ERROR "CheckHeaderIncludes: HEADER '${HEADER}' does not exist")
endif()

string(REPLACE "," ";" ALLOWED_QUOTED "${ALLOWED_QUOTED}")
string(REPLACE "," ";" ALLOWED_ANGLE "${ALLOWED_ANGLE}")

file(STRINGS "${HEADER}" include_lines REGEX "^[ \t]*#[ \t]*include")
set(violations "")
foreach(line IN LISTS include_lines)
    if(line MATCHES "#[ \t]*include[ \t]*\"([^\"]+)\"")
        set(target "${CMAKE_MATCH_1}")
        if(NOT target IN_LIST ALLOWED_QUOTED)
            list(APPEND violations "\"${target}\"")
        endif()
    elseif(line MATCHES "#[ \t]*include[ \t]*<([^>]+)>")
        set(target "${CMAKE_MATCH_1}")
        if(ALLOWED_ANGLE)
            if(NOT target IN_LIST ALLOWED_ANGLE)
                list(APPEND violations "<${target}>")
            endif()
        elseif(target MATCHES "[/.]")
            list(APPEND violations "<${target}>")
        endif()
    else()
        list(APPEND violations "unparsed: ${line}")
    endif()
endforeach()

if(violations)
    string(REPLACE ";" ", " violations "${violations}")
    message(FATAL_ERROR "${HEADER} includes something a consumer header may not: ${violations}")
endif()
