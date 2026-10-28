#!/usr/bin/env bash
# Resolve a reusable SDK, or preserve the Dockerfile's complete source stage.
set -euo pipefail
backend="$1"
image="$2"
fallback="$3"
ref="$fallback"
if [ -z "${ACT:-}" ]; then
    version="$(grep -E "^${backend}_VERSION=" versions.env | cut -d= -f2)"
    test -n "$version"
    candidate="${image}:${version}"
    if docker pull "$candidate"; then
        ref="$candidate"
    else
        echo "SDK pull failed; using source stage: $fallback" >&2
    fi
fi
printf 'ref=%s\n' "$ref" >> "$GITHUB_OUTPUT"
