# Validation

- V1 -> R1/R4: configure/build/test against both declared OpenCV versions; assert numerical and raw output equality.
- V2 -> R2: run the classifier suite with an invalid supplied model and require failure.
- V3 -> R3/R4: OpenCV 5 Darknet and CUDA-request tests; hardware CUDA execution is outside scope.
- V4: format check, generated docs check, relative-link check, act dry run and full matrix job.

## Results (2026-10-26)

Local toolchain: Ubuntu GCC 13.3.0, CMake 3.28.3, clang-format 18.

- V1: **PASS**, 80/80 CTest entries on both OpenCV 4.6.0 (system package)
  and 5.0.0 (source installation at commit
  `40738fb16ceddb5fb3fea747585f7ce6abb0605b`), Release with `WERROR=ON`.
- V2: **PASS**, empty, missing, and malformed supplied classifiers fail on
  both versions; these negative checks are also registered with CTest.
- V3: **PASS**, mandatory real ReLU inference checks values, rank-4 shape,
  typed output and raw FP32 bytes. OpenCV 5 Darknet and GPU rejection tests pass
  on a CPU-only OpenCV installation. No hardware GPU execution claimed.
- Wrong-version configuration (expecting 5.0.0 against 4.6.0) fails as intended.
- Installed C/C++ CMake, pkg-config and Python ctypes consumers pass for both
  builds; the exported C ABI matches all 19 committed symbols.
- Format, generated docs, full-tree cppcheck, diff and documentation link
  checks pass. `act` dry run passes for both matrix cells using cached actions
  (`--action-offline-mode` avoids repeated action downloads).
- Default NERT regression: **PASS**, 76/76 fixture-free tests, fresh Release
  build with `WERROR=ON`, fetching the declared `v0.2.0` tag. This is common
  contract coverage, not a NERT model-parity or GPU claim.
- Full source-build `act` matrix: **PASS**, both clean source builds and
  80/80 tests per version with `WERROR=ON`. The first run was interrupted
  by a concurrent dry run reusing its container names (exit 137); it is not
  validation evidence. The clean sequential run of the final source passed
  without manual container changes.

Reproduction:

```bash
cmake -S . -B /tmp/neuriplo-opencv4-gap -DDEFAULT_BACKEND=OPENCV_DNN \
  -DBUILD_INFERENCE_ENGINE_TESTS=ON -DCMAKE_BUILD_TYPE=Release \
  -DWERROR=ON -DOPENCV_TEST_VERSION=4.6.0
cmake -S . -B /tmp/neuriplo-opencv5-gap -DDEFAULT_BACKEND=OPENCV_DNN \
  -DBUILD_INFERENCE_ENGINE_TESTS=ON -DCMAKE_BUILD_TYPE=Release \
  -DWERROR=ON -DOPENCV_TEST_VERSION=5.0.0 \
  -DOpenCV_DIR=/home/oli/dependencies/opencv-5.0.0/install/lib/cmake/opencv5
cmake --build /tmp/neuriplo-opencv4-gap -j4
cmake --build /tmp/neuriplo-opencv5-gap -j4
ctest --test-dir /tmp/neuriplo-opencv4-gap --output-on-failure
ctest --test-dir /tmp/neuriplo-opencv5-gap --output-on-failure
scripts/abi/check_symbols.sh /tmp/neuriplo-opencv4-gap
scripts/abi/check_symbols.sh /tmp/neuriplo-opencv5-gap
test/consumer/run.sh /tmp/neuriplo-opencv4-gap
LD_LIBRARY_PATH=/home/oli/dependencies/opencv-5.0.0/install/lib \
  test/consumer/run.sh /tmp/neuriplo-opencv5-gap
act push -W .github/workflows/opencv-compatibility.yml \
  --job opencv-compatibility --dryrun --action-offline-mode
act push -W .github/workflows/opencv-compatibility.yml \
  --job opencv-compatibility --verbose --action-offline-mode
```

Session-local evidence: `/tmp/opencv{4,5}-{config,build,test,consumer}.log`,
`/tmp/opencv-act-dryrun.log`, `/tmp/opencv-act-full-final.log`,
`/tmp/opencv-cppcheck.log`.
Fixture SHA256: `481bc4ead99aa5c52588883193ffa83298b6c146fdda92e1be820f758560bea5`.
