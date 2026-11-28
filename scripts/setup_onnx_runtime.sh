#!/bin/bash

# Setup script for ONNX Runtime backend
# This script is called by the unified setup_dependencies.sh

set -e

echo "Setting up ONNX Runtime backend..."

# Load versions from versions.env
if [ -f "versions.env" ]; then
    source versions.env
else
    echo "Error: versions.env file not found"
    exit 1
fi

# Default installation directory
version="$ONNX_RUNTIME_VERSION"
dir="$DEPENDENCY_ROOT/onnxruntime-linux-x64-gpu-$version"

# Check if already installed
if [[ -d "$dir" && "$FORCE" != "true" ]]; then
    echo "✓ ONNX Runtime already installed at $dir"
    exit 0
fi

echo "Installing ONNX Runtime $version..."

# ONNX Runtime 1.28 split the GPU build per CUDA major and renamed it
# (onnxruntime-linux-x64-gpu_cuda13-<ver>); older releases publish a single
# onnxruntime-linux-x64-gpu-<ver>. Try the build matching the pinned CUDA
# first, then the old name. Either way the install lands at $dir, the path
# cmake/versions.cmake and the other scripts expect.
cuda_major="${CUDA_VERSION%%.*}"
base_url="https://github.com/microsoft/onnxruntime/releases/download/v$version"
mkdir -p "$DEPENDENCY_ROOT"
tmp_dir="$(mktemp -d "$DEPENDENCY_ROOT/.onnxruntime.XXXXXX")"
trap 'rm -rf "$tmp_dir"' EXIT

downloaded=""
for archive in "onnxruntime-linux-x64-gpu_cuda${cuda_major}-$version" "onnxruntime-linux-x64-gpu-$version"; do
    if wget -q "$base_url/$archive.tgz" -O "$tmp_dir/ort.tgz"; then
        downloaded="$archive"
        break
    fi
done
if [[ -n "$downloaded" ]]; then
    echo "Downloaded $downloaded"
    tar -xzf "$tmp_dir/ort.tgz" -C "$tmp_dir"
    rm -rf "$dir"
    mv "$tmp_dir/$downloaded" "$dir"
else
    # The upstream GitHub repo is currently private (security investigation),
    # so release assets 404. Fall back to the official Microsoft NuGet
    # packages of the exact pinned version (GPU Linux, then CPU). The nupkg
    # layout differs from the release tgz, so headers and native libs are
    # mapped into the same include/lib tree cmake validates.
    echo "GitHub release unavailable; trying official NuGet packages for ORT $version..."
    nuget_base="https://api.nuget.org/v3-flatcontainer"
    for pkg in "microsoft.ml.onnxruntime.gpu.linux" "microsoft.ml.onnxruntime"; do
        if wget -q "$nuget_base/$pkg/$version/$pkg.$version.nupkg" -O "$tmp_dir/ort.nupkg"; then
            echo "Downloaded NuGet $pkg $version"
            rm -rf "$tmp_dir/ort-nuget" && mkdir -p "$tmp_dir/ort-nuget"
            if command -v unzip >/dev/null 2>&1; then
                unzip -q "$tmp_dir/ort.nupkg" -d "$tmp_dir/ort-nuget"
            else
                python3 -c "import zipfile,sys; zipfile.ZipFile('$tmp_dir/ort.nupkg').extractall('$tmp_dir/ort-nuget')"
            fi
            inc_dir="$(find "$tmp_dir/ort-nuget" -type f -name onnxruntime_cxx_api.h | head -n 1 | xargs dirname)"
            lib_dir="$(find "$tmp_dir/ort-nuget" -type f -path "*linux-x64*" -name "libonnxruntime.so*" | head -n 1 | xargs dirname)"
            if [[ -n "$inc_dir" && -n "$lib_dir" ]]; then
                rm -rf "$dir"
                mkdir -p "$dir/include" "$dir/lib"
                cp -r "$inc_dir/." "$dir/include/"
                cp "$lib_dir/"* "$dir/lib/"
                # The NuGet native lib carries SONAME libonnxruntime.so.1 but
                # ships no versioned file (the release tgz has the .so.1
                # chain). Without it every dependent link fails with
                # "libonnxruntime.so.1 not found" / undefined OrtGetApiBase.
                if [[ -f "$dir/lib/libonnxruntime.so" && ! -e "$dir/lib/libonnxruntime.so.1" ]]; then
                    ln -s libonnxruntime.so "$dir/lib/libonnxruntime.so.1"
                fi
                echo "$version" > "$dir/VERSION_NUMBER"
                downloaded="nuget:$pkg"
                echo "Warning: using NuGet $pkg $version instead of the GitHub release archive." >&2
                break
            else
                echo "NuGet $pkg layout unexpected (missing include/lib), trying next..." >&2
            fi
        else
            echo "  NuGet $pkg not available" >&2
        fi
    done
fi
if [[ -z "$downloaded" ]]; then
    echo "Error: no ONNX Runtime $version build found (GitHub releases and NuGet exhausted)" >&2
    exit 1
fi

echo "✓ ONNX Runtime $version installed successfully at $dir"
