# Release 0.9.0 Preparation

## Candidate Status

Prepared on 2026-09-10 on `release/0.9.0`, based on
`511b766a04474c7ec8a2d9ca1ab8fca9e3b4f498` from `develop`.
Local validation below was performed before the preparation commit. Subsequent
candidate commit, PR, and CI identifiers are recorded in the release PR as
publishing proceeds. Publishing the candidate branch is authorized; merging,
tagging, and accepting coverage exclusions are not. The changelog date must be
updated if release publication happens on another day. This record is not final
release approval or a claim that every backend was exercised.

Scope: release metadata, compatibility documentation, and two stale integration
test expectations. No production inference, public API, plugin ABI, dependency
pin, Dockerfile, or workflow code changed during preparation. This is release
stabilization, not completion of roadmap Phase 6.

## Compatibility Notes

- TensorRT metadata includes the batch dimension. Consumers must not prepend
  it again; constructor `input_sizes` still exclude it.
- Consumers using OpenCV must include and link it themselves rather than rely
  on Neuriplo's public headers to supply it transitively.
- OpenCV 5 has importer and GPU restrictions described in `CHANGELOG.md`.
- Windows evidence covers OPENCV_DNN compilation, ONNX_RUNTIME compilation and
  tests, and public-header compilation without OpenCV. It does not establish
  Windows support for every backend.
- DALI's generated source-shape output is INT64 `(height, width)`. The byte
  interface copies outputs to host memory; GPU stages are not zero-copy.

## Observed Validation

Host: Linux x86_64, GCC 13.3, CMake 3.28.3, OpenCV 4.6.0, glog 0.6.0,
GTest 1.14.0. GoogleMock was built from the existing distro sources with
`NEURIPLO_REQUIRE_GMOCK=ON`. GPU: RTX 3060 Laptop, 6 GiB, driver 610.43.02.
TensorRT: `/home/oli/dependencies/TensorRT-10.14.1.48`; local CUDA toolkit:
12.0.140. This host run does not validate the pinned CUDA 13 container stack.
DALI libraries came from `/home/oli/dependencies/dali`; the configured pin is
1.50.0, but the installed library version was not independently established.

| Check | Result | Scope / Limitation |
| --- | --- | --- |
| `./scripts/quality/run.sh` | Pass | Format, include audit, cppcheck. Cppcheck reports its 12-configuration limit; this is not exhaustive preprocessing coverage. |
| `python3 scripts/gen_backend_docs.py --check` | Pass | Generated dependency documentation matches unchanged metadata. |
| OpenCV release build and CTest | 3/3 CTest entries pass | Common tests, template tests, and backend tests. Mock-only case intentionally skips when a real model is supplied. |
| OpenCV real-model gate | 2/2 GoogleTests pass, no skips | BasicInference and IntegrationTest with explicit fixture dimensions. |
| TensorRT release build and CTest | 3/3 CTest entries pass | All six TensorRT GPU tests ran, including batch-inclusive metadata, rejected static-batch request reporting, and repeated inference. |
| DALI GPU tests | 6/6 local GoogleTests pass, no skips | Cover the host-pinned DALI 1.50.0 build only; not postprocessing or a complete ensemble. |
| DALI in-image validation | 23/23 GoogleTests pass, no skips | Run inside the NVIDIA Triton image with that image's DALI 1.51.2 against the repaired backend. Covers unit8/int32/int64/float raw outputs, undeclared metadata, changing shapes, batch-size rejection, and float16 rejection. This run follows the three 0.9.0 DALI repairs and is the authoritative DALI gate. |
| Downstream neuriplo-infer | Build and 90/90 CTest entries pass | Local OPENCV_DNN configuration with KServe disabled; not HTTP serving or model-accuracy validation. |
| Markdown links | Existing targets resolve / HTTP 200 | Two new `v0.9.0` comparison URLs return 404 until the tag is published. Recheck them afterward. |

Base-commit CI, inspected during preparation:

- [Linux CI](https://github.com/olibartfast/neuriplo/actions/runs/33346143977):
  success at `511b766`. TensorRT and ExecuTorch jobs are build-only; ordinary
  CI does not exercise DALI GPU inference.
- [Windows Build](https://github.com/olibartfast/neuriplo/actions/runs/33346143882):
  success at the same SHA.

These runs predate the preparation edits. They are not candidate-commit CI.
No new local sanitizer run or full backend-matrix run was performed.

## Reproduction

Run from the Neuriplo root unless another working directory is specified.
All `build-release-0.9.0-*` directories were new; pre-existing builds and SDK
installations were not replaced. The commands below require the stated local
dependencies and fixtures and do not install them.

### CPU Gate

```bash
cmake -S . -B build-release-0.9.0-opencv -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DDEFAULT_BACKEND=OPENCV_DNN \
  -DBUILD_INFERENCE_ENGINE_TESTS=ON -DNEURIPLO_REQUIRE_GMOCK=ON \
  -DENABLE_CCACHE=OFF
./scripts/quality/run.sh && \
python3 scripts/gen_backend_docs.py --check && \
cmake --build build-release-0.9.0-opencv --parallel 2 && \
ctest --test-dir build-release-0.9.0-opencv --no-tests=error --output-on-failure
```

The local fixture was exported offline using system Python packages torch
2.12.0+cu130, torchvision 0.27.0+cu130, and ONNX 1.22.0. It is an **untrained**
ResNet-18, `torch.manual_seed(0)`, `weights=None`, evaluation mode, with static
FP32 input `[1,3,224,224]` and output `[1,1000]`, exported with opset 17 and
`dynamo=False`. No pretrained weights or packages were downloaded. This proves
execution and shape handling, not predictive accuracy.

For OpenCV 4.6, 16 constant Identity nodes from the exporter were replaced with
equivalent copied initializers; ONNX checker accepted the result. Temporary
scripts are `/tmp/opencode/export_release_resnet.py` and
`/tmp/opencode/fold_fixture_identities.py`. The unmodified ONNX was used for
TensorRT. The folded ONNX was named `resnet18-opencv.onnx` in
`build-release-0.9.0-opencv/backends/opencv-dnn/test/`, with its absolute path
in that directory's `model_path.txt`.

Run there for the real-model gate:

```bash
./OCVDNNInferTest \
  --gtest_filter=OCVDNNInferTest.BasicInference:OCVDNNInferTest.IntegrationTest \
  --gtest_output=xml:release-integration-fixed.xml
```

Require both tests to run and pass, with zero skips. A zero exit code alone
is insufficient: the existing harness falls back to mocks after load failures.

### TensorRT Gate

```bash
cmake -S . -B build-release-0.9.0-trt -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DDEFAULT_BACKEND=TENSORRT \
  -DTENSORRT_DIR=/home/oli/dependencies/TensorRT-10.14.1.48 \
  -DCUDAToolkit_ROOT=/usr -DBUILD_INFERENCE_ENGINE_TESTS=ON \
  -DNEURIPLO_REQUIRE_GMOCK=ON -DENABLE_CCACHE=OFF
cmake --build build-release-0.9.0-trt --parallel 2 \
  --target TensorRTInferTest PatternsTest TestTemplateCompileTest
# The build does not stage the model; supply the unmodified ONNX fixture
# yourself at the path the test harness reads.
cp /path/to/resnet18.onnx build-release-0.9.0-trt/backends/tensorrt/test/
env LD_LIBRARY_PATH=/home/oli/dependencies/TensorRT-10.14.1.48/lib \
  /home/oli/dependencies/TensorRT-10.14.1.48/bin/trtexec \
  --onnx=build-release-0.9.0-trt/backends/tensorrt/test/resnet18.onnx \
  --saveEngine=build-release-0.9.0-trt/backends/tensorrt/test/resnet18.engine \
  --fp16 --memPoolSize=workspace:1024M --skipInference
env LD_LIBRARY_PATH=/home/oli/dependencies/TensorRT-10.14.1.48/lib \
  NEURIPLO_REQUIRE_TENSORRT_TESTS=1 \
  ctest --test-dir build-release-0.9.0-trt --no-tests=error -V
```

Engine build took approximately 39 seconds. Subsequent six-test execution took
less than one second because the engine already existed; explicit executed
test counts, not duration alone, establish that inference occurred.

### DALI Gate

```bash
cmake -S . -B build-release-0.9.0-dali -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DDEFAULT_BACKEND=DALI \
  -DDALI_DIR=/home/oli/dependencies/dali \
  -DBUILD_INFERENCE_ENGINE_TESTS=ON -DNEURIPLO_REQUIRE_GMOCK=ON \
  -DENABLE_CCACHE=OFF
cmake --build build-release-0.9.0-dali --parallel 2 --target DALIInferTest
```

In `build-release-0.9.0-dali/backends/dali/test/`, `dali_pipeline_path.txt`
contains `/home/oli/repos/neuriplo/data/dali/yolo_preprocess_640.dali` and
`dali_image_path.txt` contains `/home/oli/repos/neuriplo/data/neuriplo.png`.
Run from that directory and require all six tests to execute without skips:

```bash
env LD_LIBRARY_PATH=/home/oli/dependencies/dali:/home/oli/dependencies/dali/.libs \
  ./DALIInferTest --gtest_output=xml:release-results-fixed.xml
```

### DALI In-Image Gate

The local backend tests compile against the repository-pinned DALI 1.50.0.
The authoritative gate compiles the same backend in the NVIDIA container the
fixtures ship from and exercises it against that image's DALI 1.51.2:

```bash
bash backends/dali/test/run_container_tests.sh
```

The script generates typed fixtures plus a fresh preprocessing pipeline inside
`nvcr.io/nvidia/tritonserver:25.12-py3` (DALI 1.51.2), compiles the backend in
that image with host GoogleTest sources, and runs 23 tests. The verification
stage asserts every test ran with zero skips, failures, or errors. The
generated pipeline embeds explicit `output_dtype` declarations, matching the
new metadata-first guidance in `CHANGELOG.md`; its SHA-256 is below.

### Consumer Gate

The clean neuriplo-infer checkout was at
`bf42832b289905d587079606a205c2af2765407e`. Its local neuriplo-tasks dependency
was at `6d088bcc288d14bece247543ae69cc64f34c256b`, with pre-existing edits to
`AGENTS.md` and `docs/Versioning.md`. Those changes were not touched. Local
source overrides, rather than published sibling pins, were deliberately tested.

```bash
cmake -S /home/oli/repos/neuriplo-infer -B build-release-0.9.0-consumer -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DDEFAULT_BACKEND=OPENCV_DNN \
  -DENABLE_APP_TESTS=ON -DNEURIPLO_INFER_ENABLE_KSERVE=OFF \
  -DFETCHCONTENT_FULLY_DISCONNECTED=ON \
  -DFETCHCONTENT_SOURCE_DIR_NEURIPLO=/home/oli/repos/neuriplo \
  -DFETCHCONTENT_SOURCE_DIR_VIDEOCAPTURE=/home/oli/repos/neuriplo-infer/build/_deps/videocapture-src \
  -DFETCHCONTENT_SOURCE_DIR_NLOHMANN_JSON=/home/oli/repos/neuriplo-infer/build/_deps/nlohmann_json-src \
  -DENABLE_CCACHE=OFF
cmake --build build-release-0.9.0-consumer --parallel 2
ctest --test-dir build-release-0.9.0-consumer --no-tests=error --output-on-failure
```

The first build exceeded the command's 180-second limit; resuming the same
build completed, followed by the successful test run. No consumer source edits
or dependency installation were needed.

### Fixture SHA-256

| Fixture | SHA-256 |
| --- | --- |
| Original ONNX | `cdaae3c3a3930ebf8bdf625865e1dc05ecdf96ef098b65a76730a9daa4610827` |
| OpenCV-folded ONNX | `ad3a7baae3711c5605a39dfc271fe3e85e10ec8e456928f595515b383ec8befa` |
| TensorRT engine | `15d73377735ecb0417f70f8f8fae3717d806c021f4f13f294d97ebc2c152205e` |
| DALI pipeline (host-pinned, 1.50.0) | `5567596e4cbc064d4d4e3756047cc4e4e8146041f0333a6c276daedf05cca029` |
| DALI pipeline (generated in image, 1.51.2) | `38d84960ae1333129c1bef1ace0e07bd690333151ecaf4ad2010863183f2f88b` |
| PNG | `07d2d65b0849a22b09dd3deda95759fa02aea9098e7785061c5e26c94fc3891a` |

## Attempt Record

| Attempt | Owner | Outcome / Intervention |
| --- | --- | --- |
| Metadata packet | Implementer subagent | Edits delivered. The brief ambiguously appended a period after the command; worker ran `--check .`, failed, then reran without authorization and passed. Not a valid first-pass score. Direct review corrected inaccurate draft claims. |
| Initial DALI GPU check | Orchestrator | 5 pass, 1 fail: obsolete INT32/3 expectation. Existing generator establishes INT64/2; only the test assertions changed. |
| DALI repair check | Orchestrator | 6 pass, zero skips. |
| DALI fix implementation | Orchestrator | Batch-size gate, int decode, per-output metadata, and constructor cleanup landed; 23/23 in-image acceptance tests pass with zero skips. |
| Initial CPU gate | Orchestrator | Exit 0, but OpenCV real-model test skipped. Exporter constant identities, then missing constructor dimensions, prevented real-model validation. |
| OpenCV repair and final CPU gate | Orchestrator | Two real-model tests pass with no skips; quality/docs/build/CTest gate passes. One exploratory invocation from the repo root found no fixture; the recorded real-model gate uses the test working directory. |
| TensorRT gate | Orchestrator | Fresh engine built; all six GPU tests pass with required-tests mode. |
| Consumer gate | Orchestrator | Initial build timed out; resumed build and 90/90 CTest entries pass. |
| Independent reviewer dispatch | Reviewer subagent | Blocked by provider credit exhaustion. |
| Fallback preparation review | General subagent, read-only packet | Confirmed the test fixes and documentation claims; requested an explicit manual Windows candidate-CI gate, now recorded below. This was not a full production-diff audit. |

The orchestrator performed repair work explicitly after delegation became
unavailable. This is a workflow deviation, not a successful delegated attempt;
the later fallback review independently examined the preparation changes.
No model-cost or token-savings estimates were invented. Existing Claude/Codex
agent definitions do exist; their host-specific permission settings must not be
assumed to enforce path restrictions in this OpenCode session.

## Remaining Release Gates

- [ ] Independent review of the complete production-target diff, including the
  release-preparation working changes. No weakened acceptance or unreviewed
  inference changes may be accepted.
- [ ] Commit, push, and create the release PR after explicit authorization.
  This includes C++ test changes: do not use a docs-only `[skip ci]` commit.
- [ ] Required CI succeeds on the exact final candidate SHA; record that SHA
  and run URLs here or in the release PR. Base-commit CI is not a substitute.
- [ ] After publication of the candidate branch is authorized, explicitly run
  `gh workflow run windows-build.yml --ref release/0.9.0`. Windows CI does not
  trigger automatically for `release/**` pushes or PRs into `master`. Check
  the resulting run's `headSha` against the final candidate and record its
  successful URL, or obtain an explicit maintainer exclusion. A later candidate
  change requires a new matching run.
- [ ] Validate LibTorch setup's driver/variant selection or obtain a maintainer
  risk decision. The installed SDK is `2.0.1+cu118`, not the pinned `2.3.0`;
  it was neither replaced nor presented as validation of the pinned release.
- [ ] Decide whether additional DALI postprocessing, dynamic-shape TensorRT,
  performance, or KServe integration evidence is required. KServe's checkout
  contains unrelated active work and was left untouched. No exclusions have
  been approved merely by listing these gaps.
- [ ] Confirm migration notes, final date, and any explicit exclusions.
- [ ] Merge the approved release PR into `master`; validate the resulting
  commit, then create and publish `v0.9.0` on that exact commit with approval.
- [ ] Verify the new comparison links and publish release notes if requested.
- [ ] Back-merge `master` into `develop`, validate and push the integration,
  set the approved next development version (proposed `0.10.0-dev`), and
  remove the finished release branch locally and remotely after verifying both
  destinations. Follow `docs/Versioning.md` and the release cleanup policy.

Rollback planning: retain the existing `v0.8.0` artifact and its matching
consumer/configuration set. Reverting only the library while retaining new
TensorRT metadata assumptions is not a safe rollback. Never move an existing
release tag; release repairs follow the approved hotfix process.
