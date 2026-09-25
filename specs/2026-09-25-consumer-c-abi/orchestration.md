# Orchestration — Stable Consumer C ABI

Same workflow as Phase 2 (`specs/2026-09-22-plugin-abi-loader-hardening/
orchestration.md`), which is the reference for the harness inventory and the
enforced-vs-advisory table; only what differs is stated here.

## Roles and routing

| Role | Tier | Owns |
| --- | --- | --- |
| Specifier | Strongest | This packet, Group 0 (header contract, acceptance suite, symbol list), [Q-n] decisions, accepting or rejecting output, Group 6 docs |
| Planner | Mid (may be the specifier's session) | Sequencing, packets, routing defects to fresh workers |
| Implementer | Cheap | Groups 1, 2, 4, 5; Group 3 after [Q-1] |
| Reviewer | Strongest, read-only | Diff review against the packet; may reject |

Group 0 is not delegable: it writes the header every group implements against
and the suite that scores them. Lesson carried from Phase 2: the suite must
include the conforming edge cases (empty outputs, trailing `struct_size`
fields), not only the malformed ones — a missing conforming case is how a
worker's over-strict guard went unnoticed there.

## Acceptance

The command in `validation.md`, run once per attempt as the worker's final
action. Implementation groups score a subset (`ctest -R` limited to their
suites) until Group 5 lands the symbol check; the full command then applies.

Read-only to every worker: `include/neuriplo/neuriplo_c.h` (the contract —
a worker that needs it changed stops and reports), `CApiContractTest.c`,
`CApiWrapperTest.cpp`, `CApiHeaderCheck.c`, `CApiHeaderCheck.cpp`,
`CheckHeaderIncludes.cmake`, `backends/src/test/CMakeLists.txt`,
`scripts/abi/neuriplo_c.symbols`, the fixtures
(`backends/src/test/plugin_fixtures/`), `plugin_abi.h`, and the existing C++
API headers. The public declarations in `include/neuriplo/neuriplo.hpp` are
read-only too; Group 4 writes only their bodies and private parts.

Targeted checks per group are the anchored `ctest -R` filters in `plan.md`
(Group 0 section). Per [O-1], Groups 1–2 run their filter in both a gcc
(`build-capi`) and a clang (`build-capi-clang`, `CC=clang-18
CXX=clang++-18`, same flags) tree, both with `-DWERROR=ON`; the acceptance
command itself stays gcc.

## Packet sketches

Filled in fully when each group is delegated, from the template in the
Phase 2 orchestration file.

Corrected in Group 0 (2026-09-26): the stub file and its `SOURCES` entry
already exist, so no group but 3 and 5 touches the root `CMakeLists.txt`.

- **Group 1:** writable `src/neuriplo_c.cpp` only. Final state: lifecycle,
  version, status strings, backend list, entry-point guard, thread-local
  error, `struct_size` rules, status choice per [D-11]/[D-12]/[D-15]. 15
  cases. Largest group; `plan.md` Notes gives a 1a/1b split by function.
- **Group 2:** writable `src/neuriplo_c.cpp`. Final state: metadata views
  owned by the engine; result handle owning moved `RawOutputTensor`s; exactly
  one release; result independent of engine ([D-13]). 11 cases, plus ASan.
- **Group 3:** writable `src/neuriplo_c.cpp`, a new `src/neuriplo_c_log.cpp`,
  and the root `CMakeLists.txt` limited to adding that one source. Final
  state per [D-7] and [D-16] and the [A-2] outcome. 7 cases, under TSan.
  Prerequisite: Groups 1–2 accepted.
- **Group 4:** writable `include/neuriplo/neuriplo.hpp` — bodies and private
  members only; public declarations as in the Group 0 skeleton; required
  behaviour in `plan.md` [T-12]. Must not include any neuriplo header but
  `neuriplo_c.h` (the build's include audit enforces it). 9 cases.
  Prerequisite: Groups 1–2 accepted (independent of Group 3).
- **Group 5:** writable `CMakeLists.txt`, `cmake/neuriplo-config.cmake.in`,
  `neuriplo.pc.in`, `test/consumer/**`, `scripts/abi/check_symbols.sh`.
  Must not alter what existing targets build or link. `check_symbols.sh`
  reads `scripts/abi/neuriplo_c.symbols` (ignoring `#` lines) and fails both
  on a listed symbol missing from the library and on an exported
  `neuriplo_*` symbol not listed.

## Measurement

Run ledger in the same format as Phase 2 §7, one row per attempt, planner and
worker split. Record when the harness does not report a metric rather than
leaving the cell looking measured.

## Open questions

- [O-1] **Resolved (2026-09-26): yes.** Implementers of Groups 1–2 run their
  targeted filter in a gcc 13 tree and a clang 18 tree, both
  `-DWERROR=ON` (commands in Acceptance above). Both compilers are installed
  locally, the Group 0 tree builds warning-free under both, and Phase 2's
  Werror failure was compiler-specific. Groups 3–5 may do the same; it is
  required only where the most new C/C++ lands.
