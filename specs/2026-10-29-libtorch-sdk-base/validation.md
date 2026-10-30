# Validation defined before implementation

- Resolver tests: CPU-specific image tag, correct PYTORCH_VERSION pin, missing
  registry/source fallback, act offline behavior, every LibTorch matrix row.
- A correct reusable image skips SDK/Python downloads; a mismatched SDK version
  or compute platform is rejected explicitly.
- Matching Python Torch CPU version and C++ archive version are inspected.
- Build cold SDK and a builder inheriting that SDK, with a C++ link smoke test;
  compare logs to prove the warm path skips dependency installation.
- Existing formatting, cppcheck, SDK resolver and backend matrix checks pass.
  Run act dry/full on applicable safe jobs and inspect remote workflow results.
- Synthetic/build/cache evidence is distinct from real-model matrix validation.

## Local evidence

- Five SDK guard tests and three resolver/routing/matrix tests pass.
- Cold SDK image built successfully with LibTorch/Python Torch 2.14.0 CPU.
- Native C++20 link/runtime smoke passes (`CPU tensor sum = 6`) and verifies the
  linked library's compiled Torch version and CPU configuration.
- A full builder inheriting the dependency image compiles successfully.
  Warm logs confirm SDK and Python downloads are skipped. The compiled SDK
  smoke executable is reused only with matching source checksums, and still
  runs against the current runtime on every warm build.
- A Docker build requesting `cu130` from the CPU SDK is rejected before download.
- Formatting, SDK docs sync, actionlint, act changes dry/full and publisher
  dry-run pass. Registry publication and exact-head hosted CI remain pending.
