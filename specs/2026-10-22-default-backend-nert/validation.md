# Validation — Default backend migration to NERT

## Checks

- [V-1] → [R-1]: multi-backend `NERT;ONNX_RUNTIME` build with a NERT fixture;
  the full CApi suite passes, including the restored index-0 and `ids.front()`
  assertions.
- [V-2] → [R-2]: fresh default configure (no `-DDEFAULT_BACKEND`, no
  `NERT_FIXTURE`), build, and full ctest. `ctest -N` contains no
  `NertInferTest`.
- [V-3] → [R-2]: the same build with `-DNERT_FIXTURE=<path>` registers
  `NertInferTest`; the test passes and fails rather than skips when the fixture
  is missing.
- [V-4] → [R-3]: default configure with `-DBUILD_INFERENCE_ENGINE_TESTS=ON`
  emits no CMP0077 developer warning. The NERT + CMake < 3.16 path is an
  explicit `FATAL_ERROR`. `-DWERROR=ON` reaches the `neuriplo` target and the
  fetched NERT target, and the build plus full ctest pass.
- [V-8] → [R-4]: `-DSANITIZERS=ON` reaches the `neuriplo` target and the
  fetched NERT target; ASan+UBSan full ctest passes. Malformed plugin dtype
  values are inspected as raw bytes before enum validation.
- [V-5] → [R-4]: explicit `-DDEFAULT_BACKEND=OPENCV_DNN` configures and builds;
  `gen_backend_docs.py --check` and `format.sh --check` pass.
- [V-6] → [R-5]: workflow review shows the backend-free `default-nert` job and
  the retained `nert` parity job.
- [V-7] → [R-6]: documentation and metadata review.

## Acceptance

```sh
: "${NERT_REPO:?set NERT_REPO to a NERT checkout}"
BUILD="${NERT_DEFAULT_BUILD:-build-default-nert}"
rm -rf "$BUILD"
cmake -S . -B "$BUILD" \
  -DFETCHCONTENT_SOURCE_DIR_NERT="$NERT_REPO" \
  -DBUILD_INFERENCE_ENGINE_TESTS=ON \
  -DCMAKE_BUILD_TYPE=Release \
  && cmake --build "$BUILD" --parallel "$(nproc)" \
  && ctest --test-dir "$BUILD" --output-on-failure
```

## Evidence

| ID | Command/Check | Result | Date | Notes |
|----|---------------|--------|------|-------|
| V-1 | `cmake -DNEURIPLO_BACKENDS=ONNX_RUNTIME -DNERT_FIXTURE="$NERT_FIXTURE"`; `ctest -R "CApi|NertInferTest"` | 43/43 passed | 2026-10-22 | Local ORT 1.19.2 used; version-drift warning only |
| V-2 | Default configure/build; full ctest; `ctest -N` | 76/76 passed; no `NertInferTest` entry | 2026-10-22 | Fresh build dir, no fixture |
| V-3 | Multi-backend fixture run; direct `NertInferTest` after removing the host fallback | `NertInferTest` passed (13/13, 38.2 s) | 2026-10-22 | Fixture supplied through `NERT_FIXTURE`; the test binary no longer reads a machine-local path. |
| V-4 | `cmake ... -DWERROR=ON`; inspect `flags.make`; full ctest | 76/76 passed; `-Werror` present on `neuriplo` and `nert` | 2026-10-22 | NERT has two upstream unused-symbol warning exceptions; `InferenceInterface.cpp` unused parameter removed. |
| V-8 | `-DSANITIZERS=ON -DCMAKE_BUILD_TYPE=Debug`; ASan+UBSan full ctest | 76/76 passed | 2026-10-22 | Plugin loader now reads untrusted dtype as raw underlying bytes before validation, so expected malformed-plugin tests do not trip UBSan enum checks. |
| V-5 | Explicit `OPENCV_DNN` configure/build + `ctest -R "^CApi"`; docs/format checks | Configured and built; 42/42 passed; both checks passed | 2026-10-22 | OpenCV 4.6.0 |
| V-6 | `.github/workflows/ci.yml` review | `default-nert` added with Werror/full ctest/CApi/ASan; `nert` retained with CApi filters | 2026-10-22 | No SDK/torch in the new job |
| V-7 | Diff review | Docs/meta/packet updated | 2026-10-22 | — |
| — | Final acceptance command above | Run once after targeted checks | 2026-10-22 | Scoreboard result reported by the implementer |

## Deviations

- The local ONNX Runtime tree is 1.19.2 while `versions.env` pins 1.28.0; CMake
  reports version drift and uses the local tree. This does not change the
  default-first assertion under test.
