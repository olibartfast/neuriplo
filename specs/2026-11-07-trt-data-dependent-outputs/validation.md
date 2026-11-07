# Validation — PASS

Jetson Orin, installed TensorRT 10.16.2, CUDA 13.2, native ARM build.
Final Slurm job 65 dispatches via the Slurm 25.11.2 Docker worker and host SSH.

- Build PASS; CTest 76/76 PASS, NEURIPLO_REQUIRE_TENSORRT_TESTS=1.
- NonZero fixture output [M,1], M=0/2/4/1/0: typed and raw shapes/values PASS.
  Metadata preserves [-1,1]. Public context recreation and transactional
  successful/failed reload PASS. Fixed-shape tests use a synthetic
  ReduceMean+broadcast graph with [1,3,224,224] -> [1,1000].
- Downstream neuriplo-infer CTest 149/149 PASS.
- Actual neuriplo-infer SuperPoint+LightGlue: Capitol74 and Sacré-Cœur49
  matches at score >0.2, both inspected app PNGs. Blank pair succeeds with0.
- clang-format18 full-tree and cppcheck full-tree PASS.
- GPT-6.1-sol review: no remaining concrete findings.

Backend patch changes buffer allocation, ownership and shape propagation.
Model values, backend selection and fallback policy are preserved. No runtime
version or release pin changed. Upstream task snapshot1331639 plus zero-match
fix is required for the downstream test. Logs and renders retained under
/tmp/keypoint-matching/app-e2e/trt/ on the controlling notebook.
