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
`CApiWrapperTest.cpp`, `scripts/abi/neuriplo_c.symbols`, the Phase 2 fixtures,
`plugin_abi.h`, and the existing C++ API headers.

## Packet sketches

Filled in fully when each group is delegated, from the template in the
Phase 2 orchestration file.

- **Group 1:** writable `src/neuriplo_c.cpp` (new), `CMakeLists.txt` (add the
  source). Final state: lifecycle, version, backend listing, entry-point
  exception guard, thread-local error, `struct_size` rules.
- **Group 2:** writable `src/neuriplo_c.cpp`. Final state: metadata views
  owned by the engine; result handle owning moved `RawOutputTensor`s; exactly
  one release.
- **Group 3:** writable `src/neuriplo_c.cpp`, a new `src/neuriplo_c_log.cpp`.
  Final state per [Q-1] decision and [A-2] outcome.
- **Group 4:** writable `include/neuriplo/neuriplo.hpp` (new). Must not
  include any neuriplo header but `neuriplo_c.h`.
- **Group 5:** writable `CMakeLists.txt`, `cmake/neuriplo-config.cmake.in`,
  `neuriplo.pc.in`, `test/consumer/**`, `scripts/abi/check_symbols.sh`.
  Must not alter what existing targets build or link.

## Measurement

Run ledger in the same format as Phase 2 §7, one row per attempt, planner and
worker split. Record when the harness does not report a metric rather than
leaving the cell looking measured.

## Open questions

- [O-1] Should implementers run against a pinned `-DWERROR=ON` gcc *and* a
  clang build in their targeted checks? Phase 2's Werror failure was
  gcc-only. Recommendation: yes for Groups 1–2, which add the most new C/C++.
