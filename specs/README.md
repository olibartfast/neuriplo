# specs/ — layout and ID conventions

This folder holds the project constitution and the dated feature packets written
against it. The short version: `R-` is a requirement, `T-` is a task, `V-` is a
check that proves a requirement, and the letters exist so a spec can be argued
about precisely — "V-5 doesn't actually prove R-3" is a reviewable claim in a way
that "the memory test looks weak" is not.

## What is in here

| File | Role |
| --- | --- |
| `mission.md` | What the product is, who it serves, what it must never do. |
| `tech-stack.md` | Languages, dependencies, architectural boundaries, explicit non-choices, validation entrypoints. |
| `roadmap.md` | Ordered phases with status. Its **Status Key** section defines the status values — read it there rather than here, so the two cannot drift. |
| `YYYY-MM-DD-feature-name/` | One feature packet per increment of active work. The date is when the spec was written, not when it shipped. |

`roadmap.md`'s **Specification Rule** decides when a packet is required at all:
multi-phase work, public-behavior or architecture changes, or low reversibility.
Trivial fixes need no packet; small contained ones may get a PR-level note.

## Packet files

| File | Answers | Owns IDs |
| --- | --- | --- |
| `requirements.md` | What must be true, and what is deliberately excluded | `R-`, `D-`, `A-`, `Q-` |
| `plan.md` | In what order, in thin phases that each end runnable | `T-` |
| `validation.md` | How each requirement is proven — **written before implementation** | `V-`, `M-` |
| `orchestration.md` | How the phase is delegated to coding agents, when it is | `O-` |

`orchestration.md` is optional and appears only where a phase is actually being
delegated: roles, handoff packets, permission boundaries, the single acceptance
command, and the run ledger.

## The prefixes

| Prefix | Means | Lives in | Example |
| --- | --- | --- | --- |
| `R-n` | **Requirement.** Something that must be true when the phase is done. In scope, testable, and traceable to a `V-n`. | `requirements.md` | `[R-3]` release ownership is exception-safe |
| `D-n` | **Decision.** A choice already made, with its rationale, so it is not silently relitigated later. | `requirements.md` | `[D-4]` no ABI version bump |
| `A-n` | **Assumption.** Believed true but unverified, with its basis and how it will be confirmed. If an `A-` turns out false, something downstream is wrong — that is the point of naming it. | `requirements.md` | `[A-1]` fixtures build with no vendor SDK |
| `Q-n` | **Open question.** Not yet decided, with a recommendation and who owns the call. A `Q-` that blocks work says so explicitly. | `requirements.md` | `[Q-2]` reject the call or drop the tensor? |
| `T-n` | **Task.** A concrete unit of implementation work, grouped into thin phases. | `plan.md` | `[T-10]` introduce the RAII release guard |
| `V-n` | **Validation check.** An automated, observable proof of one or more requirements — a command, a test, an assertion. | `validation.md` | `[V-5]` release count is exactly one |
| `M-n` | **Manual check.** A check that needs human judgement and cannot be automated honestly, such as reading docs as a newcomer would. | `validation.md` | `[M-1]` docs walkthrough on a clean machine |
| `O-n` | **Orchestration question.** An open question about delegation, permissions, or harness configuration rather than about the feature. | `orchestration.md` | `[O-1]` re-add committed agent definitions? |

## How they connect

The traceability rule is that every requirement has at least one check, and
every check names what it proves:

```
[R-3] release_outputs is called exactly once on every exit path
  ↳ [T-10] introduce an RAII guard owning the infer out-parameters
  ↳ [V-5] → [R-3]: fixture counts release calls; 0 or 2 fails
```

Written in the files, a check states its targets with an arrow:

```
- [ ] [V-6] → [R-10], [A-2]: parity against ONNX_RUNTIME within 1e-4
```

So `V-6` proves requirement `R-10` and simultaneously tests assumption `A-2`.
A requirement with no `V-` is unprovable and a `V-` with no `R-` is
unmotivated; both are review findings.

## Rules that keep the IDs useful

- **IDs are stable and append-only.** Never renumber. `R-7` means the same thing
  for the life of the packet, because commits, PR comments, and review threads
  cite it. A new requirement gets the next free number even if that leaves the
  list in a non-obvious order.
- **Split with a letter suffix.** When one check turns out to be two, add
  `V-4a` beside `V-4` rather than renumbering everything after it. Same for
  tasks (`T-27a`).
- **Retired IDs stay retired.** If a requirement is dropped, say so where it was
  and do not reuse the number.
- **Numbers are per packet.** `R-1` in one feature directory is unrelated to
  `R-1` in another. Cross-packet references name the packet.
- **Phases vs groups.** `roadmap.md` numbers repository phases (`Phase 1`…`6`,
  and `N0`…`N3` for the parallel Native Engine Track). Inside a packet,
  `plan.md` numbers its own delivery slices as `Group 0`, `Group 1`, … Each
  group ends in something runnable, so `develop` stays green mid-phase.
- **Validation before implementation.** `validation.md` is written and reviewed
  before code exists. A check invented after the fact tends to describe what the
  code happens to do rather than what the requirement demanded.
- **Evidence is recorded, not assumed.** `validation.md` carries an evidence
  table filled in with real results and dates. A checked box with no evidence
  row is not validated.

## Where this comes from

The packet structure and the validation-before-code rule follow the
`apply-spec-driven-development` workflow; the delegation and permission material
in `orchestration.md` follows `orchestrate-ai-coding-workflows`. The conventions
above are what those produce in this repository — they are recorded here so the
packets are readable without the skills at hand.

Two packets are in flight on their own branches as of 2026-09-22 and arrive with
them: `2026-09-17-native-engine-cpu-spine` (Native Engine Track, Phase N0) and
`2026-09-22-plugin-abi-loader-hardening` (Phase 2).
