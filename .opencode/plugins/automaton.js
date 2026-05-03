import { existsSync } from 'node:fs'
import { join, resolve } from 'node:path'
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

export const AutomatonPlugin = async ({ client, directory, worktree }) => {
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
    }
  }
}
