# Validation before implementation

- Small ONNX fixtures: mixed float and INT64 inputs, integer outputs, incorrect byte counts and input counts, legacy single-input regression.
- Compile and test against both system OpenCV 4.6 and installed OpenCV 5.
- Identical-input raw output parity against ORT for RT-DETR/OWLv2; inspect rendered results where runtime succeeds.
- Preserve Slurm jobs, artifact hashes and precise upstream limitations.

Discovery: public getMainGraph(), argData() and named setInput() expose the required contracts. RT-DETR inputs are images FP32 and orig_target_sizes INT64; OWLv2 has FP32 pixel_values and INT64 input_ids/attention_mask.

## Results (2026-11-04)

- OpenCV 4.6 and 5: 80/80 owning-repo CTest entries pass, including the new
  264-byte mixed-input ONNX fixture. It binds named FP32 and INT64 tensors,
  preserves integer values above 2**35 in variant/raw outputs and rejects
  malformed counts, bytes and dimensions.
- Downstream OpenCV 5 app: 191/191 CTest entries pass.
- Original RT-DETR: labels identical to ORT; max score error 1.8402934e-6,
  max box error 0.003860474 pixels. Render inspected (dog/bicycle/car).
- OWLv2: max raw errors logits 0.0010671616, objectness 0.0007429123,
  normalized boxes 0.00013381243; rendered prompts match ORT/OpenVINO
  (bicycle 0.34, dog 0.27).
- RT-DETRv4's original vector MatMul remains unsupported by OpenCV; the
  tasks-owned export adapter supplies an equivalent prepared artifact.
  Use task type rtdetr (rtdetrv4 is a preset, not a TaskFactory alias).
- Slurm job 39 reproduced the adapter blocker; repaired concurrent tests
  could not be submitted after the existing controller went offline.
  No repaired Slurm or released-pin claim is made.

Commands and artifact hashes are retained in the downstream blocked-Slurm
validation packet and docs/evidence/blocked-slurm-2026-11-04.json.
Local logs: /tmp/blocked-dnn5-tests.log, /tmp/blocked-dnn46-tests.log,
/tmp/blocked-dnn-app-tests.log. No model weights committed.
