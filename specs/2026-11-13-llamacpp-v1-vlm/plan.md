# Plan

## Group 0 — Specify

- [T-1] Record the dependency, compatibility contract, and validation commands before implementation.

## Group 1 — Upgrade API integration

- [T-2] Pin `LLAMACPP_VERSION` to `v1.0.0`, regenerate dependency documentation, disable the upstream CLI app target in setup/Docker SDK builds, validate required mtmd install artifacts and rebuild mismatched SDK stamps, and update the mtmd input initializer to explicitly pass text length and both flags.
- [T-3] Add an opt-in strict real VLM test driven by model/projector environment paths, using a generated RGB image. Validate the image and prompt are actually submitted to mtmd; test invalid image and projector failures; expose the existing GPU request option for hardware validation.

## Group 2 — Validate and record

- [T-4] Build upstream llama.cpp v1.0.0 and the LLAMACPP backend, run its test suite, and run formatting and generated-doc checks.
- [T-5] Run strict real VLM integration on the authorized Brev instance with the compatible Gemma 4 model/projector; use the available CPU or NVIDIA CUDA SDK, record selected-device evidence, and record instance cleanup status.
- [T-6] Update roadmap and validation evidence to reflect the resulting compatibility status.
