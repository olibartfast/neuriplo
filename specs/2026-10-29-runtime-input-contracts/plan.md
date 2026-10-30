# Plan

1. Add non-skipping synthetic TorchScript regressions for missing types, mixed
   FP32/INT32/INT64 inputs, malformed archives and exact buffer sizes. Record
   failures against the existing loader before the implementation change.
2. Add a shared strict datatype/byte-count helper and an exporter helper that
   captures actual example tensor dtypes. Read TorchScript extra files or the
   AOTInductor metadata map automatically; never guess unknown scalar types.
3. Update fixture producers and reject explicitly supplied broken fixtures
   instead of silently replacing them with mocks. Cover exporter serialization
   and AOTInductor mixed-input loading with synthetic packages.
4. Correct MIGraphX metadata, argument shapes and result conversion. Exercise
   typed metadata and CPU-reference models where the SDK is available; then run
   the same typed contracts on the authorized AMD GPU and record separate evidence.
5. Run focused regressions, complete applicable unit suites, quality checks and
   exact-head remote CI. Then pin the candidate application to the corrected
   sources and run the real model/task/backend matrix through the stack gate.
