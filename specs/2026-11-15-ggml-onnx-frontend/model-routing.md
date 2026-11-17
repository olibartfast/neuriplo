# Model routing — GGML ONNX frontend

Snapshot: 2026-11-15. Applies to the `feature/ggml-yolo-family` work and the
OpenCode agent pyramid.

## Primary (open-weight, paid/provider-hosted)

| Role | Model | Variant |
|---|---|---|
| Orchestrator | `meta/muse-spark-1.3` | high |
| Architect | `meta/muse-spark-1.3` | high |
| Planner | `meta/muse-ember-1.0` | high |
| Worker / shared frontend | `deepseek/deepseek-flash` | high |
| Worker / family packet | `deepseek/deepseek-flash` | low/medium |
| Reviewer / verifier | `meta/muse-spark-1.3` | high |
| Explorer / app gate | `meta/muse-ember-1.0` | low |
| Integrator | `deepseek/deepseek-flash` | low |

## Free fallback (OpenCode Console)

Probe date: 2026-11-15. Probe command:

```sh
~/.config/opencode/skills/select-opencode-free-models/scripts/probe-free-models.sh \
  'deepseek-v4.1-flash-free#high' \
  'glm-5.3-flash-free#high' \
  'minimax-m3.1-flash-free#high'
```

All three returned `reachable`. `longcat-2.5-free` also returned `reachable`,
but it is China-hosted and capped at 100 requests/day; use it only as a last
resort for low-volume exploration.

| Role | Primary fallback | Secondary fallback | Privacy |
|---|---|---|---|
| Orchestrator | `opencode/glm-5.3-flash-free#high` | `opencode/minimax-m3.1-flash-free#high` | zero-retention |
| Architect | `opencode/glm-5.3-flash-free#high` | `opencode/minimax-m3.1-flash-free#high` | zero-retention |
| Planner | `opencode/deepseek-v4.1-flash-free#high` | `opencode/glm-5.3-flash-free#high` | zero-retention |
| Worker / shared frontend | `opencode/deepseek-v4.1-flash-free#high` | `opencode/glm-5.3-flash-free#high` | zero-retention |
| Worker / family packet | `opencode/deepseek-v4.1-flash-free#low` | `opencode/glm-5.3-flash-free#low` | zero-retention |
| Reviewer / verifier | `opencode/glm-5.3-flash-free#high` | `opencode/minimax-m3.1-flash-free#high` | zero-retention |
| Explorer / app gate | `opencode/glm-5.3-flash-free#low` | `opencode/deepseek-v4.1-flash-free#low` | zero-retention |
| Integrator | `opencode/deepseek-v4.1-flash-free#low` | `opencode/glm-5.3-flash-free#low` | zero-retention |

## Expanded fallback (may-use-for-training approved 2026-11-15)

The maintainer approved models that may use inputs for training for this repo.
These are lower-priority than the zero-retention set but are valid backups:

| Model | Privacy | Probe |
|---|---|---|
| `opencode/flint-40b-free` | may improve, 30-day retention | reachable |
| `opencode/mimo-v2.6-flash-free` | may improve | reachable |
| `opencode/iceman-free` | retained and used for improvement | reachable |
| `opencode/big-pickle` | may improve | reachable |
| `opencode/beam-free` | no training, 30-day retention, geo-restricted | reachable |

Use them in this order after the zero-retention chain:

| Role | Third fallback | Fourth fallback |
|---|---|---|
| Orchestrator | `opencode/flint-40b-free#high` | `opencode/big-pickle` |
| Architect | `opencode/flint-40b-free#high` | `opencode/big-pickle` |
| Planner | `opencode/mimo-v2.6-flash-free` | `opencode/flint-40b-free#high` |
| Worker / shared frontend | `opencode/flint-40b-free#high` | `opencode/mimo-v2.6-flash-free` |
| Worker / family packet | `opencode/mimo-v2.6-flash-free` | `opencode/iceman-free` |
| Reviewer / verifier | `opencode/flint-40b-free#high` | `opencode/big-pickle` |
| Explorer / app gate | `opencode/beam-free#low` | `opencode/mimo-v2.6-flash-free` |
| Integrator | `opencode/mimo-v2.6-flash-free` | `opencode/iceman-free` |

## Rules

- Keep the reviewer on a different family from the worker. A DeepSeek worker
  gets GLM or MiniMax review.
- Prefer the zero-retention chain. The expanded set is approved for this repo,
  not a reason to skip the zero-retention options.
- Do not use trial-logged models (`nemotron-*`) without separate approval.
- Do not use `inkling-2-small-free`; the probe still returns geo-blocked.
- Re-probe before a long run, after any model/provider error, or after a
  catalog/deprecation change.
- If a free model is unavailable, use the next row in the same privacy class.
- `longcat-2.5-free` is a last-resort explorer only; it is China-hosted and
  limited to 100 requests/day.
- Only the orchestrator delegates. Only workers edit. Reviewer findings go to
  a worker for repair and re-verification.

## Sources

- `~/.config/opencode/skills/select-opencode-free-models/`
- `~/.config/opencode/skills/choose-agent-workflow/`
- `agentic-ai-playground/specs/2026-11-11-orchestration-pyramid/requirements.md`
