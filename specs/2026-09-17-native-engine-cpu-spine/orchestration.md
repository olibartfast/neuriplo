# Orchestration — Native Engine CPU Reference Spine (Phase N0)

How Phase N0 is executed: roles, routing, packets, boundaries, and the run
ledger. Every delegation below is traceable to a packet in this file and a
ledger row. Packets are committed **before** their worker runs; the ledger is
updated when the handback lands.

## Roles and routing

Routed by capability tier and role name, per `orchestrate-ai-coding-workflows`.
Model assignments live in `~/.config/opencode/opencode.json` (global
OpenCode config; repo has no `.opencode/` override):

| Role | Tier | Model (as configured) | Owns |
| --- | --- | --- | --- |
| Specifier | Strongest | session driver | This packet, [Q-n]/[D-n] decisions, accepting or rejecting output |
| Architect | Strongest | `deepinfra/tencent/Hy3` | Ambiguous/expensive decisions (escalation only) |
| Planner | Mid | `meta/muse-spark-1.3-contributor` | Handoff packets, routing defects to fresh workers |
| Implementer | Cheap | `deepinfra/zai-org/GLM-5.3-Flash` | One packet, named paths only |
| Reviewer | Strongest, read-only | `deepseek/deepseek-flash` ("DeepSeek V4.1 Flash", DeepSeek API) | Diff review against the packet; may reject, never writes |

Note (2026-09-30): the reviewer mapping was broken (`deepseek/deepseek-v4.1-flash`,
not a real endpoint) and fixed to `deepseek/deepseek-flash`. Group 1 ran
without a reviewer gate for that reason; Groups 2+ reinstate it.

## Acceptance

Owned by the specifier, read-only to anything it scores — `validation.md`
commands. Per-group filters score implementation groups until parity lands;
the full command applies from Group 7. The acceptance command runs **once**
per attempt, verbatim, as the worker's final action; targeted checks may run
freely before it.

## Enforced vs advisory

| Rule | Enforcement |
| --- | --- |
| Reviewer writes nothing (`permission: edit: deny` in `agents/reviewer.md`) | Harness-enforced |
| Worker writable paths (per packet) | Advisory (prompt + orchestrator diff review before commit) |
| No commits/pushes by workers (orchestrator commits between packets) | Advisory + verified (`git status` before commit) |
| Acceptance run once, verbatim, no repair-and-rerun | Advisory + verified (handback must paste the tail) |
| Spec files (`specs/**`) writable by specifier/orchestrator only | Advisory + verified |
| Clean tree before dispatch; never stash/revert чужое work | Orchestrator-verified |

## Handoff packets

### Packet T-3 — fixture node dump (probe, read-only)

- **Worker:** implementer. **Writable:** nothing (read-only probe).
- **Task:** read `export_torchvision_classifier.py`, run the export into a temp
  dir if torch/torchvision/onnx exist (never write to the hard-coded
  `/workspace/`), dump node op_types + attributes + opset + IO shapes.
- **Result:** opset 18, 49 nodes, no BatchNorm → decisions [D-8], [R-5]/[V-4a]
  updates. No repo files touched.

### Packet Group 1 — skeleton and build seam (T-4..T-7, amended)

Branch synced with `origin/develop`; baseline default build green 35/35.
Q-1 resolved as [D-7] before dispatch; packet touches no parsing code.

- **Writable:** new `engine/` (`engine/CMakeLists.txt` → static lib
  `neuriplo_engine`, `engine/include/` headers, stub source); new
  `cmake/Native.cmake` (module `Native`); `cmake/BackendRegistry.cmake`
  (NATIVE entry only); `cmake/versions.cmake` (declared-null validator case
  only); `CMakeLists.txt` (engine hook only); `docs/backends.yaml` (NATIVE
  entry only) + GEN-regenerated files; T-6 guard inside the writable set;
  `backends/native/test/CMakeLists.txt` as an **empty stub** (orchestrator
  amendment, plan.md Notes — `neuriplo_add_backend_tests` runs an unguarded
  `add_subdirectory` over every enabled backend's `TEST_DIR`, so
  `-DDEFAULT_BACKEND=NATIVE` with tests on cannot configure without it).
  Never: `specs/**`, rest of `backends/native/` (Group 6), `versions.env`,
  Docker, workflows, `src/**`, `include/**`, `backends/src/**`.
- **Read (1 turn, parallel):** plan.md:24-51 + Notes tail; BackendRegistry
  :1-25,:177-197; CMakeLists.txt:29-70; versions.cmake:155-230;
  OpenCVdnn.cmake; backends.yaml:1-45,:193-218; requirements [R-1]/[D-1]/
  [D-5]/[D-7]; validation [V-1]/[V-9]/[V-10].
- **Final state (T-5 order load-bearing):** (1) versions.cmake FIRST branch
  `VERSION_VAR_NAME STREQUAL "NEURIPLO_NO_EXTERNAL_SDK"` passing validation;
  (2) `cmake/Native.cmake`; (3) registry `NATIVE` + `Native` /
  `backends/native/test` / `NEURIPLO_NO_EXTERNAL_SDK`. T-4: standalone-capable
  `engine/` (CMake ≥3.10, CXX 17, `$<BUILD_INTERFACE:>` includes), default
  build compiles no engine sources. T-6: configure-time guard over engine TUs;
  prove it fires on a temp offender, then revert; engine `*.hpp/*.cpp` must
  not contain `neuriplo`/`InferenceInterface` (validation grep). T-7: yaml
  entry (`version_var`/`setup_script`/`dir_var`/`dir_default: null`,
  `dockerfile` omitted, `test_exe: null`, ONNX, x86_64+arm64, gpu false);
  regen + `--check` clean. Accepted: null version renders a cosmetic `` `None` ``
  GEN row (generator out of scope).
- **Checks:** default `OPENCV_DNN` configure/build/ctest green with NATIVE
  registered; `-DDEFAULT_BACKEND=NATIVE` configures+builds empty engine.
- **Acceptance (once, verbatim):**
  ```bash
  cmake -S . -B build-ocv -DDEFAULT_BACKEND=OPENCV_DNN -DBUILD_INFERENCE_ENGINE_TESTS=ON && cmake --build build-ocv && ctest --test-dir build-ocv --output-on-failure && cmake -S . -B build-native -DDEFAULT_BACKEND=NATIVE -DBUILD_INFERENCE_ENGINE_TESTS=ON && cmake --build build-native && python3 scripts/gen_backend_docs.py --check && ./scripts/quality/format.sh --check
  ```
- **Budget:** 12 turns. **Handback:** `GROUP 1 HANDACK pass|fail`, one line per
  T-n with evidence, acceptance tail, deviations, NO-GO, `git status` list.

### Packet Group 2 — ONNX loading and graph IR (T-8..T-10)

*Planner writes the full packet here before dispatch. Constraints it must
carry: [D-7] hand-written wire reader behind one interface, opset 18 schemas
([D-8]), [R-5] op set (`Conv Gemm MatMul Add Relu MaxPool ReduceMean
Reshape`), load-time rejection with node name + op type ([V-3]), loader unit
tests ([V-2]), no parsing dependency in any build closure.*

## Run ledger

One row per attempt. Metrics the harness did not report are marked `—`
(Phase 7 precedent). Tokens are not re-estimated.

| Attempt | Group | Role | Model | Turns | Wall clock | First-pass acceptance | Interventions | Outcome |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1 (packet) | Planner | muse-spark-1.3 | — | — | n/a (writes packet) | 0 | Packet delivered, committed here |
| 2 | 1 (implement) | Implementer | GLM-5.3-Flash | 12 (budget exhausted in investigation) | — | n/a (blocked pre-implementation) | 1 (orchestrator ruled on SetupTests fork, amended packet + plan.md) | Blocked, no files changed |
| 3 | 1 (implement) | Implementer | GLM-5.3-Flash | — | — | Pass (self-reported slip: acceptance-before-regen, re-ran verbatim) | 0 code; orchestrator re-scored (77/77 + NATIVE build + docs/format) | Pass → commit `f9f48eb`, 9 files, +134/−1 |
| 4 | 0 (T-3 probe) | Implementer | GLM-5.3-Flash | — | — | Pass with finding (opset 18 vs pinned 12) | 1 (maintainer decision [D-8]) | Evidence recorded in [A-1]; [R-5]/[V-4a]/plan updated |

## Open questions

- [Q-1] Resolved → [D-7] (hand-written reader), 2026-09-30.
- [T-3]/[A-1] Resolved → [D-8] (explicit opset-18 pin), 2026-09-30.
- [Q-2..Q-4] Open, non-blocking for N0 (owners: maintainer).
