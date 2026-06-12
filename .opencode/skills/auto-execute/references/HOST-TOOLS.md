# Host Tools

Host: `opencode`

Use this file when an Automaton skill asks for host-native collaboration or coordination tools.

## Automaton Subagents

`auto-execute` dispatches three host-native subagents that `installHost()` wrote into this host's agent directory:

- `automaton-implementer` — Implements exactly one approved Automaton plan slice from coordinator-provided context and returns evidence.
- `automaton-spec-reviewer` — Reviews spec compliance for one approved Automaton plan slice. Verdict only; no edits.
- `automaton-quality-reviewer` — Reviews maintainability and regression risk for one approved Automaton plan slice. Verdict only; no edits.

Their static role bodies are baked into the host agent files. The coordinator fills per-call slots in `auto-execute/references/*-prompt.md` (slice, constraints, acceptance criteria, implementation summary) and hands the packet to the named agent.

## Dispatch

- availability: available
- dispatch: Use the Task tool (or `@mention` where supported) to invoke `automaton-implementer`, `automaton-spec-reviewer`, or `automaton-quality-reviewer` by name. Pass the per-call dispatch packet (slice, constraints, acceptance criteria, implementation summary) as the task body; the role body is in the markdown file under `.opencode/agents/` and every Automaton subagent denies `permission.task` so it cannot fan out to another subagent.
- wait: Wait for the OpenCode subagent response before dispatching dependent reviews.
- cleanup: No Automaton cleanup step is required; follow OpenCode session conventions.
- tracking: Use todowrite for session-local progress tracking when useful.
- precondition: The primary agent's `permission.task` configuration must allow `automaton-implementer`, `automaton-spec-reviewer`, and `automaton-quality-reviewer` for Task-tool named-agent dispatch to work. If any of those three names is denied or filtered out, treat dispatch as unavailable and stop under SUBAGENT-PROTOCOL.md's "Host does not expose subagent support" condition rather than pasting a role body into a generic agent.

## Rules

- Follow the skill protocol first; this file only maps host tool names.
- Dispatch only by named agent (`automaton-implementer`, `automaton-spec-reviewer`, `automaton-quality-reviewer`). Do not paste a role body into a generic worker, explorer, or other host agent at runtime.
- If the host cannot expose one of the named agents (configuration disabled, permission denied, capability missing), stop under SUBAGENT-PROTOCOL.md's "Host does not expose subagent support" condition. Do not fall back to runtime-curated prompt injection.
- Do not invent a universal SDK or CLI when the host has native subagent tools.
