# OpenCV graph tensor wiring

The maintainer requests resolving blocked coverage cells with parallel Slurm checks.
R1: Use OpenCV 5 public graph metadata to bind each named input with its declared dtype and shape; preserve INT64 values and typed integer outputs. This is mechanical backend wiring, with no model math, task preprocessing or kernel changes.
R2: Preserve the OpenCV 4 legacy path and explicit CPU/GPU policy. Reject unsupported types, unresolved dimensions and malformed buffers rather than guessing or casting integer bytes to floats.
R3: Exercise RT-DETR and OWLv2 with identical ORT inputs; retain upstream failures honestly. Add small real multi-input regression fixtures.

Scope classification: the initial approval question was based on inference-logic risk. Native graph metadata establishes that no inference semantics change is needed; backend-config-fix is allowed by REPO_META.yaml. No runtime upgrade or task contract change is authorized here.
