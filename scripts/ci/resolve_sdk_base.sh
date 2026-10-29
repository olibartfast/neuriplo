#!/usr/bin/env bash
# Resolve a reusable SDK, or preserve the Dockerfile's complete source stage.
set -euo pipefail
backend="$1"
image="$2"
fallback="$3"
ref="$fallback"
if [ -z "${ACT:-}" ]; then
    # Backend identifiers and dependency pins have independent names.
    case "$backend" in
        LIBTENSORFLOW) version_key=TENSORFLOW_VERSION ;;
        LIBTORCH) version_key=PYTORCH_VERSION ;;
        *) version_key="${backend}_VERSION" ;;
    esac
    version="$(grep -E "^${version_key}=" versions.env | cut -d= -f2)"
    test -n "$version"
    candidate="${image}:${version}"
    if docker pull "$candidate"; then
        ref="$candidate"
    else
        echo "SDK pull failed; using source stage: $fallback" >&2
    fi
fi
printf 'ref=%s\n' "$ref" >> "$GITHUB_OUTPUT"
