# Plan

1. Introduce a reusable SDK/export dependency stage, guarded by actual SDK and
   Python versions. Keep a cold source path and platform-specific tags.
2. Add the GHCR publisher and wire all LibTorch CI matrix entries to the base.
   Reuse the existing pull/fallback resolver with explicit PyTorch pin mapping.
3. Cover missing/mismatched SDK and resolver routing, build cold and warm paths,
   validate workflows locally, then publish and monitor exact-head CI.
