# Validation

## Automated checks

- [x] [V-1] → [R-1]: `python3 scripts/gen_backend_docs.py --check` reports generated dependency docs synchronized with the llama.cpp pin; GGML pin remains unchanged.
- [x] [V-2] → [R-2], [R-3], [R-6]: build llama.cpp at the exact `v1.0.0` tag using the checked-in setup configuration, configure Neuriplo with `DEFAULT_BACKEND=LLAMACPP`, and build the backend against that installation.
- [x] [V-2a] → [R-2], [R-3]: configure/build Neuriplo against the installed SDK without `LD_LIBRARY_PATH` set to prove compile/link discovery; use the documented SDK runtime library path when executing tests.
- [ ] [V-3] → [R-3], [R-4], [R-7]: on the authorized Brev instance, set `NEURIPLO_LLAMACPP_VLM_MODEL` and `NEURIPLO_LLAMACPP_VLM_MMPROJ`, select CPU by default or set `NEURIPLO_LLAMACPP_VLM_USE_GPU=1` if CUDA is available, then run strict VLM integration. Require the text-only call to return shape-consistent output without error, and image-conditioned inference to return non-empty finite output; reject bad projector/image and capture evidence of the selected runtime device. No mock fallback is allowed.
- [x] [V-4] → [R-5]: strict tests reject malformed packed image bytes and a nonexistent projector.
- [x] [V-5] → [R-2]: with the configured SDK library directory in `LD_LIBRARY_PATH`, `ctest --test-dir <llamacpp-build> --output-on-failure` passes the backend tests.
- [x] [V-6] → [R-2], [R-3]: `./scripts/quality/format.sh --check` passes.
- [x] [V-3a] → [R-2], [R-3], [R-4]: run the strict integration on CPU with real Gemma 4 model/projector fixtures; prove text-only call completion/shape consistency, real image-conditioned non-empty finite output, and no mock fallback.
- [x] [V-7] → [R-6]: run the setup script with mocked `git`/`cmake` and an isolated prefix; matching stamps skip, a mismatched stamp and a missing mtmd header rebuild, and an incomplete staged install fails while retaining the prior installation.

## Evidence

| Check | Result | Evidence |
| --- | --- | --- |
| V-1 | Pass | `python3 scripts/gen_backend_docs.py --check`; LLAMACPP is v1.0.0 and GGML remains v0.11.0. |
| V-2 | Pass | llama.cpp exact `v1.0.0` CPU SDK configure/build/install; Neuriplo LLAMACPP configure/build; final Docker `llamacpp_install` target build completed with `LLAMA_BUILD_APP=OFF`, matching stamp, and all mtmd headers/library present. |
| V-3 | Blocked | All three attempts were unable to execute commands over Brev SSH; details and IDs are recorded below. No remote test result is claimed. |
| V-4 | Pass | Strict CPU fixture test rejected bad projector and malformed packed image. |
| V-5 | Pass | `LD_LIBRARY_PATH=/tmp/neuriplo-llamacpp-v1-install/lib ctest --test-dir /tmp/neuriplo-llamacpp-v1-build --output-on-failure` — 77/77 passed. |
| V-6 | Pass | `./scripts/quality/format.sh --check`. |
| V-2a | Pass | `env -u LD_LIBRARY_PATH cmake -S . -B /tmp/neuriplo-llamacpp-v1-build-linkcheck -DDEFAULT_BACKEND=LLAMACPP -DLLAMACPP_DIR=/tmp/neuriplo-llamacpp-v1-install -DBUILD_INFERENCE_ENGINE_TESTS=ON -DCMAKE_GTEST_DISCOVER_TESTS_DISCOVERY_MODE=PRE_TEST`, followed by `env -u LD_LIBRARY_PATH cmake --build /tmp/neuriplo-llamacpp-v1-build-linkcheck --parallel 4`; all test executables linked successfully with the propagated `rpath-link` option. |
| V-3a | Pass | Strict CPU real-fixture test passed in 7.8s. It logged image embedding (81 image tokens) and returned non-empty finite image-conditioned output. The text-only call completed with matching output/shape counts; this fixture can emit EOG immediately under the existing raw-prompt fallback. |
| V-7 | Pass | Mocked setup harness covered matching-version skip, mismatched-version rebuild, missing `mtmd-helper.h` rebuild, and failed incomplete-stage preservation. |

The required Brev VLM check could not run because none of three provisioned instances accepted remote commands. `neuriplo-llamacpp-v1-vlm` (`bq83nm1hq`) and `neuriplo-llamacpp-v1-vlm-retry` (`13wv0hfyx`) remained unavailable over SSH during provisioning and were deleted. `neuriplo-llamacpp-v1-cpu` (`7omu1adbk`) reached RUNNING/COMPLETED/READY, but the container shell did not return commands and `brev exec --host` timed out connecting to port 21293. Deletion was issued and the instance remained `DELETING` at the last `brev ls` check. No Brev inference result is claimed. `act push --job build-and-test-cpu --matrix backend:LLAMACPP --dryrun` reached the changed-backends job but failed to evaluate the workflow's dynamic `fromJSON(needs.changes.outputs.cpu_matrix)` include expression. The verbose attempt stalled while pulling the act runner image and was stopped; the Docker SDK install target was validated directly with buildx instead.
