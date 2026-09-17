# Validation — Native Engine: CPU Reference Spine

Written before implementation. Every check below is defined against a
requirement in `requirements.md`; nothing is marked checked unless the command
was actually run.

## Automated Checks

Run from the repository root.

```text
# Existing default path must stay green — NATIVE must not intrude on it
cmake -S . -B build-ocv -DDEFAULT_BACKEND=OPENCV_DNN -DBUILD_INFERENCE_ENGINE_TESTS=ON
cmake --build build-ocv
ctest --test-dir build-ocv --output-on-failure
# expected: configures with no CUDA and no engine sources compiled; all tests pass
```

```text
# Native backend build and test
cmake -S . -B build-native -DDEFAULT_BACKEND=NATIVE -DBUILD_INFERENCE_ENGINE_TESTS=ON
cmake --build build-native
ctest --test-dir build-native --output-on-failure
# expected: engine unit tests, kernel tests, and shared contract tests pass
```

```text
# Parity against ONNX Runtime on the same file
cmake -S . -B build-parity -DDEFAULT_BACKEND=NATIVE -DNEURIPLO_BACKENDS=ONNX_RUNTIME \
  -DBUILD_INFERENCE_ENGINE_TESTS=ON
cmake --build build-parity
ctest --test-dir build-parity -R parity --output-on-failure
```

- [ ] [V-1] → [R-1], [R-4]: no translation unit under `engine/` includes
      `include/neuriplo/`, `backends/src/`, or `src/`. Enforced by the [T-6]
      guard; independently spot-checked with
      `grep -rn "neuriplo\|InferenceInterface" engine/ --include=*.hpp --include=*.cpp`
      → expected: no matches.
- [ ] [V-2] → [R-2]: loader test asserts node count, op types, initializer
      count, and declared input/output shapes for the ResNet-18 fixture.
- [ ] [V-3] → [R-3]: three negative tests — unsupported op, unsupported
      attribute combination, unresolvable dynamic dimension — each throwing
      `ModelLoadException` with the node name and op type in the message.
      Asserted on the message content, not just the exception type.
- [ ] [V-4] → [R-5]: every kernel has a unit test including at least one
      hand-computed expected value and one non-default configuration
      (stride/pad/dilation for `Conv`, transpose flags for `Gemm`).
- [ ] [V-5] → [R-6]: arena test asserts (a) allocation count per inference ≤ 1
      and (b) planned arena size is strictly less than the sum of all
      intermediate tensor sizes for this graph — proving reuse rather than
      concatenation.
- [ ] [V-6] → [R-10], [A-2]: parity test, elementwise max absolute difference
      against `ONNX_RUNTIME` on identical input. Threshold 1e-4; the
      **observed** figure is recorded in the evidence log, not just pass/fail.
- [ ] [V-7] → [R-8]: shared backend contract tests pass with
      `DEFAULT_BACKEND=NATIVE`, including the `get_infer_results_raw` path and
      metadata assertions.
- [ ] [V-8] → [R-9]: requesting GPU on `NATIVE` (legacy `use_gpu = true` and
      the `EngineOptions` path) raises with a message naming the limitation.
      Asserted: no inference runs and no silent CPU execution.
- [ ] [V-9] → [R-11]: `python3 scripts/gen_backend_docs.py --check` clean; the
      15 IDs in `cmake/BackendRegistry.cmake` and `docs/backends.yaml` agree.
- [ ] [V-10] → constraints: `./scripts/quality/run.sh` and
      `./scripts/quality/format.sh --check` clean; engine tests additionally run
      under ASan and UBSan with no findings.
- [ ] [V-11] → constraints: the `OPENCV_DNN` configure/build/test path above is
      unchanged, and no OpenCV target appears in the `NATIVE` link line
      (`grep -i opencv build-native/CMakeCache.txt` inspected, not assumed).
- [ ] [V-12] → [R-1]: `git subtree split --prefix=engine/ -b engine-extract`
      succeeds and the resulting tree builds on its own. Proves the extraction
      option [D-1] claims is real rather than aspirational.
- [ ] [V-13] → [R-12], [A-3]: the planner is called twice with two different
      input shapes for the same loaded graph, and both plans are correct — the
      second produced with no reload, no reparse, and no change to any kernel.
      Asserted on the plans, not on a comment. Proves the dynamic-shape work in
      Phase N2 is additive; a failure here means [A-3] was wrong and N2 is a
      rewrite, which is worth knowing now rather than then.

## Manual Checks

- [ ] [M-1] → [D-4]: `engine/README.md` states the interpreter design, the
      one-device-per-graph rule, and that CPU kernels are a correctness
      reference and not a performance target. Read as a newcomer would.
- [ ] [M-2] → [A-1]: node list of the actual exported fixture compared against
      the [R-5] kernel set; any extra op is either implemented or the fixture
      export is pinned to avoid it, with the choice recorded.
- [ ] [M-3] → [R-7]: code inspection confirming no host↔device transfer or
      device-selection decision exists at node granularity anywhere in the
      executor. This is cheap to verify now and expensive to unwind after
      Phase N1.
- [ ] [M-4] → [Q-2]: adding, renaming, or removing `NATIVE` locally does not
      leave the maintained inventories silently inconsistent, given the
      `version_var: null` / `setup_script: null` case.
- [ ] [M-5] → constraints: constitution amendment reviewed — `mission.md` and
      `tech-stack.md` no longer contradict the existence of a first-party
      runtime, and still read in under five minutes each.
- [ ] [M-6] → [D-2]: the OpenCV 5 `dnn` engine appears in this branch as a
      design reference and nowhere else. Checked: no OpenCV include, link, or
      `find_package` under `engine/`; `versions.env` `OPENCV_VERSION` and the
      `OPENCV_DNN` backend untouched by this change
      (`git diff origin/develop -- versions.env cmake/ backends/opencv-dnn/`
      → expected: empty).

## Evidence Log

| ID | Command/Check | Result | Date | Notes |
|----|---------------|--------|------|-------|
| V-1 | `grep` + configure guard | | | |
| V-2 | `ctest -R engine_loader` | | | |
| V-3 | `ctest -R engine_loader_negative` | | | |
| V-4 | `ctest -R engine_kernels` | | | |
| V-5 | `ctest -R engine_plan` | | | |
| V-6 | `ctest -R parity` | | | observed max abs diff: |
| V-7 | `ctest --test-dir build-native` | | | |
| V-8 | `ctest -R native_device_request` | | | |
| V-9 | `gen_backend_docs.py --check` | | | |
| V-10 | `scripts/quality/run.sh`, ASan/UBSan | | | |
| V-11 | `OPENCV_DNN` path + link inspection | | | |
| V-12 | `git subtree split` + standalone build | | | |
| V-13 | `ctest -R engine_plan_reshape` | | | |
| M-1 | README read-through | | | |
| M-2 | fixture node list vs kernel set | | | |
| M-3 | executor inspection | | | |
| M-4 | inventory drift walkthrough | | | |
| M-5 | constitution review | | | |
| M-6 | OpenCV containment check | | | |

## Deviations

- Record here any criterion not fully met, why, and how it is tracked.

## Definition of Done (integration)

- [ ] Every automated and manual check executed, with evidence and dates recorded
- [ ] Evidence traced to [R-n]; deviations documented
- [ ] Spec, code, roadmap, changelog, and generated docs agree — merged as one
      coherent change into `develop`
- [ ] Durable discoveries propagated to `specs/tech-stack.md` (notably the
      [Q-1] parsing decision and the [Q-4] vendor-kernel policy before Phase N1)
