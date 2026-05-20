import { existsSync } from 'node:fs'
import { join, resolve } from 'node:path'
import { buildSessionContext } from '../../.agent/.automaton/lib/context.mjs'
import { syncStatusPointerFromCurrentState } from '../../.agent/.automaton/bin/sync-status-pointer.mjs'

function resolveProjectRoot(worktree, directory) {
  const candidates = [worktree, directory, process.cwd()]
  for (const candidate of candidates) {
    if (candidate && existsSync(join(candidate, '.agent'))) {
      return candidate
    }
  }
  let current = process.cwd()
  while (current !== "/") {
    if (existsSync(join(current, '.agent'))) {
      return current
    }
    const parent = resolve(current, '..')
    if (parent === current) break
    current = parent
  }
  return process.cwd()
}

// Flag-and-clear: session.compacted handler does not receive output.messages,
// so it sets this flag; the next message-transform reads and clears it.
let needsCompactedInject = false

export const AutomatonPlugin = async ({ project, client, $, directory, worktree }) => {
  const projectRoot = resolveProjectRoot(worktree, directory)

  return {
    event: async ({ event }) => {
      if (event.type === 'session.idle') {
        syncStatusPointerFromCurrentState({
          currentTarget: join(projectRoot, '.agent', '.automaton', 'state', 'current.json'),
          statusTarget: join(projectRoot, '.agent', 'steering', 'STATUS.md')
        })
        return
      }
      if (event.type === 'session.compacted') {
        needsCompactedInject = true
        return
      }
    },
    'experimental.chat.messages.transform': async (_input, output) => {
      if (!output || !Array.isArray(output.messages) || output.messages.length === 0) return
      const firstUser = output.messages.find((m) => m && m.info && m.info.role === 'user')
      if (!firstUser || !Array.isArray(firstUser.parts) || firstUser.parts.length === 0) return

      // Dedup: every Automaton context line starts with "Automaton:" — covers both
      // active-state and no-state cases emitted by buildSessionContext.
      if (firstUser.parts.some((p) => p && p.type === 'text' && typeof p.text === 'string' && p.text.startsWith('Automaton:'))) return

      const compacted = needsCompactedInject
      needsCompactedInject = false

      const bootstrap = buildSessionContext(projectRoot, { compacted })
      if (!bootstrap) return

      const ref = firstUser.parts[0]
      firstUser.parts.unshift({ ...ref, type: 'text', text: bootstrap })
    }
  }
}
