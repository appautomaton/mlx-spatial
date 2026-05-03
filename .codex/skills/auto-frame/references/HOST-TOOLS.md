# Host Tools

Host: `codex`

Use this file when an Automaton skill asks for host-native collaboration or coordination tools.

## Subagents

- availability: available
- dispatch: Use spawn_agent with a complete task message. Prefer built-in agent_type="worker" for implementation, agent_type="explorer" for read-only discovery, or a project custom agent defined as TOML with name, description, and developer_instructions.
- wait: Use wait to collect subagent results before continuing review or integration.
- cleanup: Use close_agent after each completed subagent to free the slot.
- tracking: Use update_plan for session-local progress tracking when useful.
- configuration: Requires [features].multi_agent = true in .codex/config.toml.

## Rules

- Follow the skill protocol first; this file only maps host tool names.
- Do not invent a universal SDK or CLI when the host has native subagent tools.
- If a required host capability is unavailable, stop and recommend the non-subagent fallback skill.
