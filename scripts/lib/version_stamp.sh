#!/bin/bash
# Version stamping for backends that are built from source.
#
# The prebuilt backends leave something behind that names their version:
# TensorRT has NvInferVersion.h, LibTorch build-version, ONNX Runtime
# VERSION_NUMBER, OpenVINO runtime/version.txt. Both their setup scripts and
# cmake/DependencyValidation.cmake read those back and can therefore tell when
# an installation has drifted away from versions.env.
#
# The source-built ones leave nothing. They install into a directory whose name
# carries no version (~/dependencies/ggml, .../tvm, .../llamacpp, ...) and whose
# contents look identical whichever tag they came from. Their setup scripts
# consequently asked only "does libfoo.so exist?", which any past build answers
# yes to forever -- so bumping the pin in versions.env silently did nothing, and
# the build had no way to notice. Record the pin each install was built from so
# both ends can compare.
#
# Source this from a setup script:
#   source "${SCRIPT_DIR}/lib/version_stamp.sh"

NEURIPLO_VERSION_STAMP="neuriplo-version.txt"

# Echo the pin recorded in an install directory; echo nothing when unstamped.
neuriplo_read_stamp() {
    local stamp="$1/${NEURIPLO_VERSION_STAMP}"
    [ -f "$stamp" ] || return 0
    head -n 1 "$stamp" | tr -d '[:space:]'
}

# Record the versions.env pin an installation was built from. Call this only
# after the build and install have succeeded, so that a stamp always describes a
# tree that is actually usable.
neuriplo_write_stamp() {
    local install_dir="$1" version="$2"
    mkdir -p "$install_dir"
    printf '%s\n' "$version" > "${install_dir}/${NEURIPLO_VERSION_STAMP}"
}

# True when an existing installation was built from the pin we want, i.e. when
# it can be kept. Explains the mismatch on stdout when it cannot.
#
# An unstamped directory is treated as a mismatch: it predates stamping, so
# there is no evidence of what it holds, and guessing "probably current" is what
# let these installs drift in the first place.
neuriplo_stamp_matches() {
    local install_dir="$1" version="$2" name="$3"
    local stamped
    stamped="$(neuriplo_read_stamp "$install_dir")"

    if [ "$stamped" = "$version" ]; then
        return 0
    fi

    if [ -z "$stamped" ]; then
        echo "${name} at ${install_dir} carries no version stamp, so what it was built from is unknown"
        echo "(it predates stamping). versions.env pins ${version}."
    else
        echo "${name} at ${install_dir} was built from ${stamped}, but versions.env pins ${version}."
    fi
    echo "Re-run with FORCE=true to rebuild it from ${version}."
    return 1
}
