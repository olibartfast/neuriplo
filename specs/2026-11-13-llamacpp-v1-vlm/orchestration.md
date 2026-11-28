# Orchestration

## Roles

- Orchestrator: scoped the compatibility upgrade, verified the official v1.0.0 release tag, and integrated worker/reviewer results.
- Planner: inspected the current llama.cpp adapter, setup/test paths, and upstream v1.0.0 CMake/mtmd API changes; supplied implementation and validation criteria.
- Worker: wrote the packet before code changes; updated the pin, adapter, SDK setup, CMake validation/linking, tests, and docs; ran local and Docker checks.
- Reviewer: independently reviewed API compatibility, packet traceability, SDK setup, CMake link behavior, and test evidence; no actionable findings remain in the reviewed local changes.

## Handoffs

- [O-1] Verify the canonical v1.0.0 tag and upstream mtmd API before pinning — complete; tag resolves to commit `b1b770d90d94cad9e9ea5b993bdbc383a522fb28`.
- [O-2] Build and test the CPU SDK/Neuriplo integration, then obtain real Gemma 4 fixture evidence — local checks complete; the Brev check is blocked by SSH/command transport after three instance attempts.
- [O-3] Independently review the code/spec/evidence and resolve actionable findings — local review complete; final GPU evidence review remains pending.

## Run ledger

| Check | Result | Evidence |
| --- | --- | --- |
| Local API and CPU VLM | Pass | `validation.md` V-2, V-3a, V-4, V-5, V-6 |
| Brev VLM | Blocked | Three instances could not execute remote commands; deletion of the last instance was issued and it remained DELETING at the last check |
