# Host Tools

Host: `claude`

Use this file when an Automaton skill asks for host-native collaboration or coordination tools.

## Subagents

- availability: available
- dispatch: Use the Agent tool to dispatch host-native subagents. Task remains a documented alias in existing Claude Code configurations. Provide a short description, the full curated prompt, and the most specific available subagent type.
- wait: Agent tool calls return their result to the coordinator when complete; no separate wait command is needed.
- cleanup: No explicit close step is needed after an Agent result is returned.
- tracking: Use TodoWrite for session-local progress tracking when useful.

## Rules

- Follow the skill protocol first; this file only maps host tool names.
- Do not invent a universal SDK or CLI when the host has native subagent tools.
- If a required host capability is unavailable, stop and recommend the non-subagent fallback skill.
