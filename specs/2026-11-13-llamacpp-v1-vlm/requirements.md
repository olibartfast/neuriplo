# llama.cpp v1.0.0 VLM compatibility

## Requirements

- [R-1] Neuriplo pins and documents upstream llama.cpp `v1.0.0` without changing the GGML backend pin or backend selection behavior.
- [R-2] The LLAMACPP backend compiles against the v1.0.0 C/C++ API and retains its text inference path.
- [R-3] Gemma 4 multimodal inference passes the complete prompt and correctly sets the v1.0.0 `mtmd_input_text` length, `add_special`, and `parse_special` fields; model, projector, and image-conditioned decode remain functional.
- [R-4] An opt-in real VLM integration test requires model and projector fixtures and fails on load or inference errors; fixture absence must not silently pass as mock coverage.
- [R-5] Invalid image payloads and an invalid projector path fail clearly.
- [R-6] The checked-in source setup script and LLAMACPP Docker build both configure a v1.0.0 SDK without requesting unsupported auxiliary app targets; source setup rebuilds mismatched or incomplete installations and preserves the prior installation when staging is incomplete.
- [R-7] The strict real VLM test supports the backend's existing CPU/GPU selection; validation records the selected device and runs on the requested Brev instance.

The text-only check proves that the text call completes and returns shape-consistent output. It does not require non-empty generated text because this Gemma 4 fixture can emit EOG immediately on the existing raw-prompt fallback; its chat template is not handled by llama.cpp's minimal template engine (see `docs/TROUBLESHOOTING.md`).

## Decisions

- [D-1] Keep the existing Gemma 4-specific multimodal prompt construction. This upgrade does not introduce a generic model-family prompt system.
- [D-2] Define the opt-in real fixture through `NEURIPLO_LLAMACPP_VLM_MODEL` and `NEURIPLO_LLAMACPP_VLM_MMPROJ`; construct a deterministic packed RGB image in the test so no new image decoder dependency is needed.
- [D-3] `NEURIPLO_LLAMACPP_VLM_USE_GPU=1` requests the backend's existing GPU path for the real VLM integration test; unset or other values select CPU.

## Assumptions

- [A-1] The v1.0.0 mtmd API accepts a pointer plus explicit byte length for the input text and retains the current tokenization/evaluation helpers.

## Traceability

- [R-1] → [V-1]
- [R-2], [R-3] → [V-2], [V-3]
- [R-4] → [V-3]
- [R-5] → [V-4]
- [R-6] → [V-2]
- [R-7] → [V-3]
