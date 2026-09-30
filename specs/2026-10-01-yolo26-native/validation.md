# Validation — YOLO26 Detection on the Native Engine

Written before implementation. Nothing marked checked unless run.

```text
# Detection build and test (all phases)
cmake -S . -B build-native -DDEFAULT_BACKEND=NATIVE -DBUILD_INFERENCE_ENGINE_TESTS=ON
cmake --build build-native
ctest --test-dir build-native --output-on-failure
```

```text
# YOLO parity (Phase Y4)
cmake -S . -B build-parity -DDEFAULT_BACKEND=NATIVE -DNEURIPLO_BACKENDS=ONNX_RUNTIME \
  -DBUILD_INFERENCE_ENGINE_TESTS=ON
cmake --build build-parity --target native_yolo_parity_fixture
cmake --build build-parity
ctest --test-dir build-parity -R yolo_parity --output-on-failure
```

- [ ] [YV-1] → [R1]: loader folds all 64 survey `Constant` nodes by dtype;
  node/initializer counts asserted; `Flatten` loads; unknown op/attr still
  rejected with node + op in the message.
- [ ] [YV-2] → [R2]: int64 and bool intermediates flow end to end
  (`Shape`→int64, `Equal`→bool→`Where`); arena plan covers multi-dtype
  buffers; float32-only graph I/O enforced.
- [ ] [YV-3] → [R3]: every new kernel has a hand-computed unit case;
  configurable ops tested off-default (axes, `Mod` signs, `Cast` pairs,
  `Resize` mode, `TopK` K).
- [ ] [YV-4] → [R4]: fixture infers fully static shapes to `[1,300,6]`,
  finite output, one arena allocation per run.
- [ ] [YV-5] → [R5]: `native_yolo_parity` passes; observed max abs diff
  recorded (budget 1e-4); missing fixture fails, never skips.
- [ ] [YV-6] → constraints: `run.sh` + `format.sh --check` clean; ResNet
  `native_parity` still green (no regression); default `OPENCV_DNN` 77/77.

## Evidence Log

| ID | Command/Check | Result | Date | Notes |
|----|---------------|--------|------|-------|
| YV-1 | `ctest -R engine_loader` | | | |
| YV-2 | `ctest -R "engine_plan\|engine_executor"` | | | |
| YV-3 | `ctest -R engine_kernels` | | | |
| YV-4 | fixture end-to-end | | | |
| YV-5 | `ctest -R yolo_parity` | | | observed max abs diff: |
| YV-6 | quality + regression | | | |

## Deviations

- Record here any criterion not fully met, why, and how it is tracked.

## Definition of Done

- [ ] Every check executed with evidence and dates
- [ ] Merge hold lifted → PR into `develop`
- [ ] Roadmap N2 status and CHANGELOG updated for the shipped scope
