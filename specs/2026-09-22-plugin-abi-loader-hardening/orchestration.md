# Orchestration — Plugin ABI and Loader Hardening

How Phase 2 gets executed by coding agents. `requirements.md` owns *what* and
`plan.md` owns *in what order*; this file owns delegation, permission
boundaries, model routing, and measurement. It exists because the packet is
implementable by more than one model and the engineering contract should not
change when the model does.

## 1. Harness inventory

Taken from the tree rather than assumed:

| Concern | Current state | Enforcement |
| --- | --- | --- |
| Committed agent definitions | **None on `develop`.** `ead4f20` deliberately removed the four Claude Code role agents and their Codex counterparts: "AGENTS.md and REPO_META.yaml carry the agent contract on their own." | — |
| Agent contract | `AGENTS.md` (branch rules, quality gates, commit conventions) and `REPO_META.yaml` (`entrypoints`, `owned_paths`, `allowed_change_classes`) | Advisory — prose and YAML, not a harness allowlist |
| Build/test/quality entry points | `REPO_META.yaml: entrypoints` — configure, build, test, backend_matrix, quality | Advisory |
| Pre-commit / pre-push | `.githooks/` via `./scripts/quality/setup_hooks.sh`; pre-push runs full-tree cppcheck and backend-docs sync | **Enforced** once installed locally |
| CI | `.github/workflows/ci.yml`, `windows-build.yml`, `cache-prune.yml`; `[skip ci]` honored for documentation-only commits | **Enforced** on push |
| Branch policy | `.github/workflows/branch-policy.yml`; features branch off `develop`, never `master` (`AGENTS.md:16`) | **Enforced** |

Two consequences worth stating plainly.

**This work is not an allowed autonomous change class.** `REPO_META.yaml`
lists `ci-triage`, `dependency-sync`, `backend-config-fix`, and `docs-sync`.
Hardening the plugin loader is none of those: it changes inference-path code
behind a public boundary. So every delegated group here runs under human-approved
delegation with a named specifier, not as routine automated maintenance.

**Prompt rules are persuasion, not enforcement.** Nothing in this repository
stops a worker from editing the acceptance suite — the hooks check formatting
and docs sync, not file ownership. Every ownership rule below is therefore
marked *advisory* unless a harness allowlist backs it, and the specifier
verifies ownership after the fact via [M-4] in `validation.md` rather than
trusting the packet text. If these groups are delegated to a harness that
supports per-path write rules, encode the writable/read-only lists as
allowlists there; that is the only way they become real.

## 2. Engineering contract

The four artifacts the workflow routes on, so swapping a model is a
configuration change rather than a rewrite:

| Artifact | Lives in |
| --- | --- |
| Intent | `requirements.md` — Goal, In Scope [R-1..R-9] |
| Constraints | `requirements.md` — Constraints, Out of Scope, Decisions; `specs/tech-stack.md` non-choices |
| Phased roadmap | `plan.md` — Groups 0–6 |
| Acceptance | `validation.md` — the single acceptance command |

**Acceptance is immutable and specifier-owned.** One command decides whether a
group is done ([D-6]):

```bash
cmake -S . -B build-plugin -DDEFAULT_BACKEND=OPENCV_DNN -DBUILD_INFERENCE_ENGINE_TESTS=ON \
  && cmake --build build-plugin \
  && ctest --test-dir build-plugin -R "PluginAbi|PluginLoader" --output-on-failure \
  && ./scripts/quality/format.sh --check
```

`backends/src/test/PluginAbiContractTest.cpp` and
`backends/src/test/plugin_fixtures/` are the scoreboard. A worker may not write
them. Worker-authored tests are welcome as *output* — they go anywhere else —
but they never count as acceptance.

## 3. Roles and routing

Routed by capability tier and role, never by model name, so the table survives
model releases:

| Role | Tier | Owns |
| --- | --- | --- |
| Specifier | Strongest | This packet, Group 0, the acceptance suite, [Q-1..Q-3] decisions, accepting or rejecting worker output |
| Planner | Mid | Group sequencing, writing handoff packets, routing defects back |
| Implementer | Cheap or local | Groups 2, 3, 4, 5 — each fully specified, mechanical against a fixed suite |
| Reviewer | Strongest | Read-only diff review against the packet; may reject |

Two corrections against instinct, both from the skill and both relevant here:
delegation on a single model costs more than it saves, so if only one tier is
available, do not split roles — just implement it; and context pressure does not
vanish under delegation, it relocates to the planner, so the planner's context
is the budget to watch, not the worker's.

**Group 0 is not delegable.** It writes the scoreboard. A worker that authors
both the guard and the test proving the guard has scored its own exam.
**Group 6 is not delegable either** — [R-8]'s documentation is judgement work
about what a newcomer needs, which is exactly where cheap tiers do worst.

## 4. Handoff packets

Template, then the delegable groups filled in. A packet states obligations and
boundaries; it does not paste the spec, the plan, or finished code.

```
Writable paths:     <nothing beyond these>
Read-only paths:    <interfaces + the acceptance suite>
Required final state: <checkable obligations>
Exact identifiers:  <names that are contracts, spelled exactly>
Permitted commands: <matching the permission block>
Acceptance:         <the one command, run exactly once, as the final action>
Stop condition:     <when to stop and report>
```

### Group 2 — Metadata validation

- **Writable:** `backends/src/plugin/PluginLoader.cpp`,
  `backends/src/plugin/PluginLoader.hpp`
- **Read-only:** `include/neuriplo/plugin_abi.h`,
  `backends/src/test/PluginAbiContractTest.cpp`,
  `backends/src/test/plugin_fixtures/`, `backends/src/InferenceInterface.hpp`
- **Required final state:** every `neuriplo_layer_info_t` from a plugin is
  validated before any container is constructed from it; a null `shape` with
  `ndim > 0`, an `ndim` past a stated bound, or a null `name` is rejected with
  `ModelLoadException`; the message contains the plugin path, the layer index,
  and the field at fault; inputs and outputs are both covered; no read of an
  unvalidated pointer under ASan or UBSan.
- **Exact identifiers:** `neuriplo_layer_info_t`, `neuriplo_metadata_t`,
  `populate_metadata`, `ModelLoadException`
- **Permitted commands:** `cmake`, `cmake --build`, `ctest`,
  `./scripts/quality/format.sh`, `git diff`, `git status`
- **Acceptance:** the §2 command, once, as the final action
- **Stop condition:** acceptance reported pass or fail. Do not repair a failure
  and rerun — report it.

### Group 3 — Output validation and release ownership

- **Writable:** `backends/src/plugin/PluginLoader.cpp`, `PluginLoader.hpp`
- **Read-only:** as Group 2
- **Required final state:** `release_outputs` is called exactly once for every
  `infer` that returned 0 — on normal return, on rejection, and on an exception
  thrown mid-copy — via an RAII guard introduced *before* the new rejection
  paths; out-parameters are validated per [R-2] and [Q-3]; unknown dtype is
  handled per [R-4] and [Q-2], with `metadata_dtype_from_abi` and `to_elements`
  agreeing rather than each guessing; no out-of-bounds read reachable from any
  fixture; LSan reports no leak.
- **Exact identifiers:** `release_outputs`, `neuriplo_output_tensor_t`,
  `get_infer_results_raw`, `metadata_dtype_from_abi`, `to_elements`,
  `TensorDtype`, `InferenceExecutionException`
- **Prerequisite:** [T-1] decided. Do not start otherwise — guessing [Q-2] or
  [Q-3] produces a diff the reviewer must reject.
- **Acceptance / stop:** as Group 2

### Group 4 — Concurrency

- **Writable:** `backends/src/plugin/PluginLoader.cpp`, `PluginLoader.hpp`
- **Read-only:** as Group 2, plus `src/InferenceBackendSetup.cpp` (a caller
  whose compilation must not break)
- **Required final state:** a concurrent `load_backend_plugin` and
  `find_plugin_backend` / `get_plugin_backends` cannot race; existing callers
  compile unchanged; clean under TSan where available, ASan otherwise; the
  approach chosen is recorded in one line in `plan.md` under [T-13].
- **Exact identifiers:** `get_plugin_backends`, `find_plugin_backend`,
  `load_backend_plugin`, `load_backend_plugins`, `PluginState`
- **Acceptance / stop:** as Group 2

### Group 5 — Isolation

- **Writable:** worker-authored test files other than the acceptance suite
- **Read-only:** all of `backends/src/plugin/`, the acceptance suite, fixtures
- **Required final state:** with one conforming and several broken fixtures in
  one directory, the conforming plugin loads and serves correct results,
  `load_backend_plugins` counts accepted plugins only, and the compiled-in
  `OPENCV_DNN` path is unaffected in the same process.
- **Acceptance / stop:** as Group 2

## 5. Permissions — deny by default

Mark every rule enforced or advisory; in this repository almost all are
advisory until encoded in a harness that supports per-path rules (§1).

| Rule | Status here |
| --- | --- |
| Deny all writes except the packet's writable paths | Advisory — encode as an allowlist if the harness supports it |
| Acceptance suite and fixtures read-only to workers | Advisory; verified after the fact by [M-4] |
| `include/neuriplo/plugin_abi.h` read-only to everyone this phase ([D-4]) | Advisory; verified by [V-10] (`git diff` must be empty) |
| No network, no package installs, no SDK downloads | Advisory; [V-1] fails if a dependency appeared |
| Reviewer: broad read, zero write | Advisory |
| Worker: step ceiling, whole-file writes, one acceptance run, no repair loop | Advisory |
| Feature branches off `develop` | **Enforced** by branch-policy workflow |
| Format / cppcheck / docs-sync before push | **Enforced** by pre-push hook once installed |

Egress note: this packet needs no network at any point. Fixtures are
first-party C, there is no model download, and no SDK is fetched — which is
also what makes the suite a candidate for a local or offline worker.

## 6. Execution discipline

- **Start clean.** Each attempt starts from a known revision with fresh
  context. If the tree is dirty, do not stash or revert user work — use an
  isolated worktree, or hand the dirty state back to a human. Cleanup touches
  only paths the run itself created.
- **Iterate freely, score once.** The worker compiles and runs targeted checks
  as often as it likes — the compiler is its fastest feedback — but runs the
  acceptance command exactly once, as its final action, and reports the result
  pass or fail. A worker that edits and reruns acceptance in a loop turns a
  cheap failure into an expensive spiral.
- **No planner self-repair.** When a worker returns a defective diff, the
  planner does not quietly rewrite it: that restores single-model cost while
  still looking delegated. Send a fresh, corrected packet and confirm the
  implementation tokens were spent by the implementer role.
- **Replanning is expected.** If Group 3 reveals that the RAII guard forces a
  different shape on Group 4, update `plan.md` on this branch with the reason.
  The spec is living; silent divergence is the failure mode.

## 7. Measurement

One row per attempt, planner and worker split, so configurations can be
compared under control rather than by impression:

| Attempt | Group | Role | Model / tier | Total ctx | Active ctx | Tool calls | Turns | Wall clock | Cost | First-pass acceptance | Interventions |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0 | Specifier | strongest (this session) | — | — | — | — | — | — | n/a (writes the suite) | 0 |
| 2 | 2 | Implementer | cheap (Sonnet) | 70.4k | — | 13 | — | 63 s | — | pass (17/17) | 0 |
| 2r | 2 | Reviewer | strongest (Opus, read-only) | — | — | — | — | — | — | ACCEPT | 0 |
| 3 | 3 | Implementer | cheap (Sonnet) | 92.0k | — | 24 | — | 190 s | — | pass (29/29) | 0 |
| 3r | 3 | Reviewer | strongest (Opus, read-only) | — | — | — | — | — | — | REJECT: NULL data on empty tensor | 1 |
| 3b | 3 | Specifier | strongest (this session) | — | — | — | — | — | — | suite amended (c510760) | 1 |
| 4 | 3 | Implementer (fresh) | cheap (Sonnet) | 77.0k | — | 14 | — | 65 s | — | pass (30/30) | 0 |
| 4r | 3 | Reviewer | strongest (Opus, read-only) | 56.6k | — | 3 | — | 22 s | — | ACCEPT | 0 |
| 5 | 4 | Implementer | cheap (Sonnet) | 78.2k | — | 17 | — | 99 s | — | pass (32/32, full acceptance) | 0 |
| 5r | 4 | Reviewer | strongest (Opus, read-only) | 41.4k | — | 6 | — | 35 s | — | ACCEPT | 0 |

Run 2026-09-25 in one Claude Code cloud session. Totals are the subagent
token counts the harness reported; "—" means the harness did not report it.
Planner context was not metered separately, which is the gap §7 warns about:
the planner/specifier also wrote Group 0, Group 6 and the evidence, so its
share is not comparable to the workers'. No planner self-repair: the one
rejected diff went back to a fresh implementer with a corrected packet; the
specifier's own change in 3b was to the acceptance suite, which it owns.

Comparisons change one variable at a time, same task and packet shape,
reasoning effort pinned on both sides, repeated more than once. A single run of
each proves nothing. `outcome` is the first-pass acceptance result — recorded
whether or not a later attempt passed.

## 8. Open questions

- [O-1] Should the role definitions removed in `ead4f20` come back as committed
  harness configs (`.claude/agents/`, `.codex/agents/`) now that a phase is
  actually being delegated? That removal was deliberate, so this is the
  maintainer's call — this packet does not re-add them. Without them, every
  ownership rule in §5 stays advisory.
- [O-2] Should `REPO_META.yaml: allowed_change_classes` gain a class covering
  specified feature work under an approved packet, or does such work stay
  outside autonomous maintenance by design? Current reading: outside by design,
  which is why §1 requires a named specifier.
