# Validation defined before implementation

- Real serialized TorchScript fixtures are created within the tests, using
  typed synthetic example contracts rather than model-family constants.
- The current loader must fail the missing-type rejection and mixed-type
  metadata regressions. Do not run a short-buffer old-code case that could read
  out of bounds; use a safe oversized-buffer red case first.
- After correction: FP32 plus INT32/INT64 forward values agree with an independent
  direct LibTorch call. Missing/malformed/count-mismatched/unsupported metadata,
  graph/archive disagreements, invalid dimensions, overflow and short/extra
  buffers are rejected. No test skips the newly added contract cases.
- Python exporter tests verify that tags come from actual input tensors and
  survive save/load. Core fixture caches without the contract are regenerated.
- AOTInductor cases inspect the package metadata map and verify mixed typed
  probes/forward inputs, not just successful FP32 classification.
- MIGraphX cases inspect actual SDK shape types and typed CPU-reference outputs;
  repeat on the authorized AMD GPU, including typed values and malformed buffer
  rejection. Real weights require the application stack gate first.
- Unit/compile/CI evidence stays separate from real-weight/reference/browser
  matrix evidence. No release follows merely from a green focused suite.

## Evidence so far (2026-10-29)

- LibTorch 2.14.0 CPU: four original serialized contract regressions fail on
  the old source and pass after correction. Final focused component filter passes
  eleven tests with zero skips. A broader earlier filter also selected one
  pre-existing factory test that skips without an AOTI fixture.
- Python 2.14.0 CPU export to C++: six tests pass, including actual TorchScript
  and compiled AOTInductor packages with INT32/INT64 auxiliary inputs, output
  value comparisons, byte-count rejection and shape mismatch rejection.
- MIGraphX 2.12.0.60401-83, ROCm 6.4.1 container, MI300X gfx942: four initial
  contract tests fail on old source on CPU and GPU. After correction all six
  expanded cases pass, three CPU and three GPU, with zero skips. They check
  FP32+INT32/INT64 inputs, exact arithmetic values, INT64 outputs, host/device
  copies, model-only input metadata and invalid buffer/count rejection.
- GPU compilation previously exposed internal output parameters as model
  inputs. Model input names are captured before compilation and explicit
  offload copies match the public host-buffer interface.
- These synthetic results do not establish learned-model/task/browser matrix
  correctness. That evidence requires clean remote candidate sources and the
  application's exact-binary stack gate. No release is approved by this packet.
