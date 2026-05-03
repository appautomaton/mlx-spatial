# Host Tools

Host: `opencode`

Use this file when an Automaton skill asks for host-native collaboration or coordination tools.

## Subagents

- availability: available
- dispatch: Use OpenCode native subagent routing, including @mention-style dispatch where available. Provide the complete curated prompt to the selected subagent.
- wait: Wait for the OpenCode subagent response before dispatching dependent reviews.
- cleanup: No Automaton cleanup step is required; follow OpenCode session conventions.
- tracking: Use todowrite for session-local progress tracking when useful.

## Rules

- Follow the skill protocol first; this file only maps host tool names.
- Do not invent a universal SDK or CLI when the host has native subagent tools.
- If a required host capability is unavailable, stop and recommend the non-subagent fallback skill.
