import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'
import { syncStatusPointerFromCurrentState } from '../../.agent/.automaton/bin/sync-status-pointer.mjs'

const projectRoot = join(dirname(fileURLToPath(import.meta.url)), '..', '..')
syncStatusPointerFromCurrentState({
  currentTarget: join(projectRoot, '.agent', '.automaton', 'state', 'current.json'),
  statusTarget: join(projectRoot, '.agent', 'steering', 'STATUS.md')
})

process.stdout.write('')
