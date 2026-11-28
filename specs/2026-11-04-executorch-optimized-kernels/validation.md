# Validation before implementation

- Shell syntax and CMake configure/build for existing portable and optimized SDKs.
- Backend CTest regression suite; no operator registration collisions.
- Slurm baseline proves missing operator; repaired jobs prove runtime success and compare outputs to ORT.
- Preserve exact source/SDK/model hashes and resource limits.

## Discovery

v1.2.0 source includes optimized linear.out and optimized_native_cpu_ops_lib; current SDK built without EXECUTORCH_BUILD_KERNELS_OPTIMIZED. No upgrade required.

## Results (2026-11-04)

Pinned SDK source v1.2.0 (0b0e2c5) built into /tmp/blocked-et-sdk using
/tmp/blocked-executorch-src/executorch and /tmp/blocked-et-build. Optimized
kernels, XNNPACK and the module/tensor/data-loader extensions enabled; PIC
enabled. Existing installed SDKs and release pins were preserved.

- Owning-repo CTest: 77/77 pass with NEURIPLO_EXECUTORCH_PARITY_FIXTURE set.
  Direct real-model regression passes, checking every RF-DETR nano output
  value against ORT within 0.001. Fixture at /tmp/blocked-et-fixture contains
  model.pte, input0.bin, output0.bin/output1.bin.
- App: 191/191 CTest entries pass; existing portable-only SDK app also builds.
- Detection, segmentation and keypoint PTE artifacts execute. Raw maximum
  differences versus ORT: detection boxes 0.000066787, logits 0.000302315;
  seg boxes 0.000502378, logits 0.00747204, mask logits 0.212135 (mean
  0.00137990), mask sign agreement 0.999997428; pose boxes 0.0000211895,
  logits 0.0000109673, keypoints 0.000194073. No bitwise-equivalence claim.
- SDK tests must inherit executorch_core compile definitions (including
  ET_LOG_ENABLED) without linking a second runtime/registration archive.
  Otherwise primitive registration can abort before main.
- Slurm job 40 reproduced missing aten::linear.out for all three original
  PTE artifacts. Repaired parallel rerun is pending: controller offline.

Logs: /tmp/blocked-et-configure.log, /tmp/blocked-et-build-fast.log,
/tmp/blocked-et-install.log, /tmp/blocked-et-tests.log,
/tmp/blocked-et-real-regression.log, /tmp/blocked-et-app-tests.log.
Full configure command and artifact hashes are retained in the downstream
validation packet.

The person.jpg CLI pose gate renders a skeleton with score 0.850784 on both
ORT and ExecuTorch. Raw parity on dog.jpg is supplemental; the downstream
evidence records the person.jpg tensor comparison separately as the primary
pose gate. Explicit portable-delegate configure retains only portable_ops_lib
and portable_kernels, without combined optimized registration.

Primary person.jpg pose raw maxima: boxes 0.000091553, logits 0.000027180,
keypoints 0.000644684; these are distinct from the supplemental dog.jpg errors
listed above.
