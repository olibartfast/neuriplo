# Validation — XSE backend candidate

Defined before implementation. This packet has no code, so all checks are document/probe review. A check not performed is recorded as not executed, never as a pass.

## Automated

- [ ] [V-1] → [R-1]: bundle contract note cites upstream paths/URLs read (docs export/serve pages, `xsm-cli`, example bundle) with access dates.
- [ ] [V-2] → [R-2], [R-4]: fit memo traces each claim to `specs/mission.md`, `specs/tech-stack.md`, `specs/roadmap.md`, and the TF-packet precedent — no untraced assertion.
- [ ] [V-3] → [R-3]: inventory gap list names every registry/versions/docs/setup/Docker/CI/test slot a future backend would touch, without modifying any of them in this packet (`git status` shows only `specs/2026-11-18-xse-backend-candidate/` as new).
- [ ] [V-4] → [R-5]: spike shape paragraph exists and asserts invariance: `DEFAULT_BACKEND`, legacy overload, C++ API, consumer C ABI symbols/layouts, plugin ABI — all unchanged in this packet.
- [ ] [V-5] → [R-6]: go/no-go recorded with reasons; on go, follow-up packet name(s) named; on no-go, blocker [Q-n] and revisit trigger named.

## Manual

- [ ] [M-1] Maintainer reads the packet and calls [Q-2] (in-process TPU in scope or remote-only precedent holds). Record the call + date.
- [ ] [M-2] A second reader checks the upstream links resolve and the bundle-contract summary matches the cited docs pages.
- [ ] [M-3] Confirm no backend landing, version bump, Dockerfile/workflow change, or `Readme.md` backend-table change leaked into this branch.

## Acceptance Commands

```bash
git status --short
git diff --stat
```

Expected: only `specs/2026-11-18-xse-backend-candidate/{requirements,plan,validation}.md` (plus any pre-existing working-tree entries unrelated to this packet, listed separately). No `backends/`, `cmake/`, `docs/`, `versions.env` modifications from this packet.

## Evidence

| ID | Command/check | Result | Date | Notes |
|----|---------------|--------|------|-------|
| V-1 | bundle contract note | not executed | - | |
| V-2 | constitution trace | not executed | - | |
| V-3 | inventory gap + git status | not executed | - | pre-existing dirty entries if any must be listed here, not attributed to this packet |
| V-4 | invariance assertion | not executed | - | |
| V-5 | go/no-go | not executed | - | |
| M-1 | [Q-2] maintainer call | not executed | - | |
| M-2 | link/contract read-through | not executed | - | |
| M-3 | no-leak review | not executed | - | |

## Deviations

None yet. Pre-existing working-tree modifications on this branch at creation time (`REPO_META.yaml`, `scripts/convert_onnx_to_gguf.py` per `git status` on 2026-11-18) are **not** part of this packet and must not be committed with it.
