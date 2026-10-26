# OpenCV 5 compatibility gates

- R1: Exercise pinned OpenCV 4.6.0 and 5.0.0 with mandatory real ONNX inference in CI.
- R2: A supplied classifier fixture that cannot load must fail rather than use a mock.
- R3: Explicitly select AUTO on OpenCV 5. Reject every OpenCV 5 GPU request before loading, including CPU-only builds, rather than silently falling back. Preserve OpenCV 4 behavior.
- R4: Validate output values, shapes, and raw FP32 buffers with a tiny deterministic ONNX fixture; test Darknet and GPU rejection without GPU hardware.

No runtime pin bump, new public engine API, or GPU support expansion is included.
