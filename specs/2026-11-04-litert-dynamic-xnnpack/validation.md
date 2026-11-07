# Validation defined before implementation

- Existing LiteRT fixture-free tests pass.
- Fixture-backed test feeds identical captured app input bytes and compares all EdgeCrafter outputs to ORT: exact labels and boxes within 0.01 pixel for the six high-confidence queries, and all 300 sorted scores within 0.0001. Low-score tied queries can exchange labels/boxes and are not required to retain ordering.
- Failure to load/create/apply delegate fails explicitly, with no silent runtime substitution.
- App EdgeCrafter high-confidence detections match ORT; RF-DETR native detection/seg and YOLO smoke remain correct.
- Record SDK version, artifact hashes, commands and transport scope. No claim about invalid old converter artifacts.

## Executed results — 2026-11-04

- Full LiteRT build and 77/77 CTest entries pass with the real-model parity fixture.
- Focused LiteRT binary after expanding all-score assertions: 8 pass, optional model_path.txt smoke skipped.
- Full-tree format and cppcheck pass; no SDK version change.
- Downstream app local and KServe HTTP EdgeCrafter scores match ORT within 4.49e-6; HTTP tensors equal local exactly. Native RF-DETR detection/seg and YOLO regression checks recorded downstream.
- The model fixture uses a pre-existing reconversion, not a newly converted model. Old invalid converter artifacts are not claimed fixed.
- Linking initially failed because delegate archives refer back to libtensorflow-lite.so outside the rescan group. Moving the runtime into that existing group resolves both the test and consumer builds.
- Downstream packet: /home/oli/repos/neuriplo-infer/specs/2026-11-04-known-bad-backends/validation.md. Artifact hashes and raw error summary: /home/oli/repos/neuriplo-infer/docs/evidence/known-bad-regate-2026-11-04.json.
- No release or pin update, remote CI, GPU or gRPC validation. The maintainer subsequently authorized commit and push of the feature branches; publication does not constitute release validation.

Fresh native RF-DETR detection and segmentation were exported by the downstream helper and ran through this backend; numerical comparisons match the previously gated artifacts. YOLOv8 regression passed with expected dog/bicycle/car detections. The temporary HTTP server was stopped after verification.
