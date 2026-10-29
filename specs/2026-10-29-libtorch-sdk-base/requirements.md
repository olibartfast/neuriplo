# Reusable LibTorch SDK dependencies

Publish a versioned CPU LibTorch dependency image alongside the existing SDK
packages. Reuse the C++ SDK and matching Python CPU exporter environment in
clang-tidy, strict-warning, sanitizer and CPU jobs. Tag the platform explicitly;
a CPU SDK must never satisfy a CUDA request. Validate actual SDK build-version
and Python Torch versions before reuse. Preserve a complete local source
fallback and all existing analysis/tests. No application/version bump.

The observed 18m42s clang-tidy job spent 13m14s in analysis, 3m09s cleaning the
runner and 2m07s building Docker. Dependency caching alone is not a promise to
remove the analysis cost. Record cold/warm build behavior and remote CI outcome.
