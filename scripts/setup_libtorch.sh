#!/bin/bash

# Setup script for LibTorch backend
# This script is called by the unified setup_dependencies.sh

set -e

echo "Setting up LibTorch backend..."

# Load versions from versions.env
if [ -f "versions.env" ]; then
    source versions.env
else
    echo "Error: versions.env file not found"
    exit 1
fi

# Default installation directory
version="$PYTORCH_VERSION"
DEPENDENCY_ROOT="${DEPENDENCY_ROOT:-$HOME/dependencies}"
dir="$DEPENDENCY_ROOT/libtorch"

# The install directory carries no version, so "does the directory exist" is
# satisfied by any LibTorch forever and the pin can never take effect -- that is
# how an install drifts away from versions.env unnoticed. Compare the version
# LibTorch stamps in build-version instead.
#
# build-version looks like "2.3.0+cpu" or "2.0.1+cu118": a version and the
# compute variant it was built for. Both halves matter, so read both.
installed_version=""
installed_variant=""
if [ -f "$dir/build-version" ]; then
    installed_build="$(tr -d '[:space:]' < "$dir/build-version")"
    installed_version="${installed_build%%+*}"
    if [ "$installed_build" != "$installed_version" ]; then
        installed_variant="${installed_build#*+}"
    fi
fi

# ── Compute variant ───────────────────────────────────────────────────────────
# PyTorch publishes LibTorch as separate CPU and CUDA builds. This script always
# downloaded the CPU one, so on a GPU machine "install the pinned LibTorch"
# quietly produced a build in which torch::cuda::is_available() is false --
# LibtorchInfer then places every tensor on the CPU no matter what the caller
# asked for (backends/libtorch/src/LibtorchInfer.cpp:21). Nothing fails; the
# work just silently stops using the GPU. Choose the variant deliberately.
#
# Override with LIBTORCH_VARIANT=cpu|cu118|cu121|... to pin an exact build.

# The highest CUDA release this machine can run. The driver decides, not the
# toolkit: the shared-with-deps archives bundle their own CUDA runtime, so a
# cu121 LibTorch runs against a 12.0 toolkit but not against a driver too old
# for it. Fall back to the toolkit version when nvidia-smi is unavailable.
detect_cuda_release() {
    local release=""
    if command -v nvidia-smi > /dev/null 2>&1; then
        release="$(nvidia-smi 2>/dev/null | sed -n 's/.*CUDA Version: *\([0-9][0-9]*\.[0-9][0-9]*\).*/\1/p' | head -1)"
    fi
    if [ -z "$release" ] && command -v nvcc > /dev/null 2>&1; then
        release="$(nvcc --version 2>/dev/null | sed -n 's/.*release \([0-9][0-9]*\.[0-9][0-9]*\).*/\1/p' | head -1)"
    fi
    printf '%s' "$release"
}

libtorch_url() {
    printf 'https://download.pytorch.org/libtorch/%s/libtorch-cxx11-abi-shared-with-deps-%s%%2B%s.zip' \
        "$1" "$version" "$1"
}

variant_published() {
    wget -q --spider --tries=1 --timeout=20 "$(libtorch_url "$1")" 2> /dev/null
}

# Which CUDA builds exist differs per PyTorch release -- 2.3.0 ships cu118 and
# cu121 but no cu120, so deriving the variant from the local CUDA version alone
# produces a URL that 404s. Ask the server instead: walk down from the release
# this machine supports and take the newest build it actually publishes.
newest_published_cuda_variant() {
    local release="$1" major minor candidate
    major="${release%%.*}"
    minor="${release#*.}"

    while [ "$major" -ge 10 ]; do
        while [ "$minor" -ge 0 ]; do
            candidate="cu${major}${minor}"
            if variant_published "$candidate"; then
                printf '%s' "$candidate"
                return 0
            fi
            minor=$((minor - 1))
        done
        major=$((major - 1))
        minor=9
    done
    return 1
}

cuda_release="$(detect_cuda_release)"

if [ -n "${LIBTORCH_VARIANT:-}" ]; then
    variant="$LIBTORCH_VARIANT"
    if ! variant_published "$variant"; then
        echo "Error: LibTorch $version has no '$variant' build published." >&2
        echo "  Tried: $(libtorch_url "$variant")" >&2
        exit 1
    fi
    echo "Using LibTorch variant $variant (from LIBTORCH_VARIANT)"
else
    # An existing installation's variant is a statement of intent: replacing a
    # CUDA LibTorch with a CPU one moves every consumer's inference onto the CPU,
    # and an upgrade must never do that as a side effect. Keep its family.
    case "$installed_variant" in
        cu*) want_cuda=true;  reason="the installed build is $installed_variant" ;;
        cpu) want_cuda=false; reason="the installed build is CPU-only" ;;
        *)
            if [ -n "$cuda_release" ]; then
                want_cuda=true;  reason="CUDA $cuda_release was detected"
            else
                want_cuda=false; reason="no CUDA was detected"
            fi
            ;;
    esac

    if [ "$want_cuda" = "true" ]; then
        if [ -z "$cuda_release" ]; then
            echo "Error: a CUDA LibTorch is wanted ($reason) but no CUDA was detected here." >&2
            echo "  Install the CUDA toolkit or driver, or set LIBTORCH_VARIANT explicitly" >&2
            echo "  (LIBTORCH_VARIANT=cpu to deliberately move to a CPU-only build)." >&2
            exit 1
        fi
        if ! variant="$(newest_published_cuda_variant "$cuda_release")"; then
            # Falling back to CPU here is exactly the silent downgrade this
            # selection exists to prevent, so stop and let a human decide.
            echo "Error: LibTorch $version publishes no CUDA build for CUDA $cuda_release or older." >&2
            echo "  Pick one explicitly with LIBTORCH_VARIANT=cuXYZ, or accept a CPU-only" >&2
            echo "  build with LIBTORCH_VARIANT=cpu." >&2
            exit 1
        fi
        echo "Using LibTorch variant $variant ($reason; CUDA $cuda_release available)"
    else
        variant="cpu"
        echo "Using LibTorch variant cpu ($reason)"
    fi
fi

# ── Already installed? ────────────────────────────────────────────────────────
if [[ -d "$dir" && "$FORCE" != "true" ]]; then
    if [ "$installed_version" = "$version" ] && [ "$installed_variant" = "$variant" ]; then
        echo "✓ LibTorch ${version}+${variant} already installed at $dir"
        exit 0
    fi
    echo "LibTorch at $dir is ${installed_build:-an unrecognised build}, but this run wants ${version}+${variant}."
    echo "Re-run with FORCE=true to replace it (this removes the existing $dir)."
    exit 0
fi

echo "Installing LibTorch ${version}+${variant}..."

# Create directory and download
mkdir -p "$DEPENDENCY_ROOT" && cd "$DEPENDENCY_ROOT"
wget -q "$(libtorch_url "$variant")" -O tmp.zip
# Unzip merges into an existing tree, leaving a mix of both versions behind.
rm -rf "$dir"
unzip -q tmp.zip && rm tmp.zip

# The archive is named for the build it should contain, but only build-version
# says what was actually unpacked. Check, so a mismatch surfaces here rather
# than as tensors quietly landing on the CPU at inference time.
unpacked="$(tr -d '[:space:]' < "$dir/build-version" 2> /dev/null || true)"
if [ "$unpacked" != "${version}+${variant}" ]; then
    echo "Error: expected LibTorch ${version}+${variant} but ${dir}/build-version says '${unpacked:-nothing}'." >&2
    exit 1
fi

echo "✓ LibTorch ${version}+${variant} installed successfully at $dir"
