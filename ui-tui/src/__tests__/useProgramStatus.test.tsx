import { PassThrough } from 'node:stream'

import { renderSync } from '@hermes/ink'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { patchOverlayState, resetOverlayState } from '../app/overlayStore.js'
import { turnController } from '../app/turnController.js'
import { patchUiState, resetUiState } from '../app/uiStore.js'
import { applyDisplay, hydrateFullConfig } from '../app/useConfigSync.js'
import { useProgramStatus } from '../app/useProgramStatus.js'
import { $programStatusEnabled, clearProgramStatus, encodeProgramStatus, PROGRAM_STATUS_CLEAR } from '../lib/programStatus.js'

const pause = () => new Promise(resolve => setTimeout(resolve, 220))
const mounted: ReturnType<typeof renderSync>[] = []

function mount(isTTY = true) {
  const reports: string[] = []
  const stdout = { isTTY, write: (seq: string) => reports.push(seq) } as unknown as NodeJS.WriteStream

  function Harness() {
    useProgramStatus(stdout)

    return null
  }

  mounted.push(renderSync(<Harness />, {
    patchConsole: false,
    stdin: new PassThrough() as never,
    stdout: Object.assign(new PassThrough(), { columns: 80, rows: 24, isTTY: false }) as never,
    stderr: new PassThrough() as never
  }))

  return reports
}

const configure = (display: Record<string, unknown>) => applyDisplay({ config: { display } }, () => {})

beforeEach(() => {
  resetUiState()
  resetOverlayState()
  turnController.fullReset()
  $programStatusEnabled.set(null)
  clearProgramStatus(() => {})
})

afterEach(() => {
  for (const instance of mounted.splice(0)) {
    instance.unmount()
  }

  turnController.fullReset()
  resetOverlayState()
  clearProgramStatus(() => {})
})

describe('terminal-owned program status', () => {
  it('waits for config, defaults on, and clears an existing report when opted out live', async () => {
    const reports = mount()
    await pause()
    expect(reports).toEqual([])
    configure({})
    await expect.poll(() => reports).toEqual([encodeProgramStatus('idle')])
    configure({ program_status: false })
    await expect.poll(() => reports).toEqual([encodeProgramStatus('idle'), PROGRAM_STATUS_CLEAR])
    turnController.startMessage()
    await pause()
    expect(reports).toHaveLength(2)
    configure({})
    await expect.poll(() => reports.at(-1)).toBe(encodeProgramStatus('working'))
  })

  it('does not report to pipes, including cleanup', async () => {
    const reports = mount(false)
    configure({})
    turnController.startMessage()
    await pause()
    expect(reports).toEqual([])
    expect(clearProgramStatus(seq => reports.push(seq))).toBe(false)
  })

  it('honors opt-out from config.get before the first report and preserves it on RPC failure', async () => {
    const reports = mount()
    const request = vi.fn(async () => ({ config: { display: { program_status: false } } }))
    await hydrateFullConfig({ request } as never, () => {})
    expect($programStatusEnabled.get()).toBe(false)
    request.mockRejectedValueOnce(new Error('offline'))
    await hydrateFullConfig({ request } as never, () => {})
    expect($programStatusEnabled.get()).toBe(false)
    await pause()
    expect(reports).toEqual([])
  })

  it('reports real turn outcomes without sending assistant text to the terminal protocol', async () => {
    const reports = mount()
    configure({})
    await expect.poll(() => reports.at(-1)).toBe(encodeProgramStatus('idle'))
    turnController.startMessage()
    await expect.poll(() => reports.at(-1)).toBe(encodeProgramStatus('working'))
    turnController.recordMessageComplete({ text: 'Private assistant reply' })
    await expect.poll(() => reports.at(-1)).toBe(encodeProgramStatus('done'))
    turnController.startMessage()
    await expect.poll(() => reports.at(-1)).toBe(encodeProgramStatus('working'))
    turnController.recordMessageComplete({ text: 'Partial private answer', partial: true, status: 'error' })
    await expect.poll(() => reports.at(-1)).toBe(encodeProgramStatus('error'))
    turnController.startMessage()
    await expect.poll(() => reports.at(-1)).toBe(encodeProgramStatus('working'))
    turnController.recordError()
    await expect.poll(() => reports.at(-1)).toBe(encodeProgramStatus('error'))
    turnController.startMessage()
    await expect.poll(() => reports.at(-1)).toBe(encodeProgramStatus('working'))
    turnController.recordMessageComplete({ status: 'interrupted' })
    await expect.poll(() => reports.at(-1)).toBe(encodeProgramStatus('idle'))
    turnController.recordMessageComplete({ text: 'Finished' })
    await expect.poll(() => reports.at(-1)).toBe(encodeProgramStatus('done'))
    turnController.reset()
    await expect.poll(() => reports.at(-1)).toBe(encodeProgramStatus('idle'))
    expect(reports.every(seq => !seq.includes('msg='))).toBe(true)
  })

  it('reports confirmation then resumes working, deduplicating unrelated UI updates', async () => {
    const reports = mount()
    configure({})
    turnController.startMessage()
    await expect.poll(() => reports).toEqual([encodeProgramStatus('working')])
    patchOverlayState({ confirm: { title: 'Continue?', onConfirm: () => {} } })
    await expect.poll(() => reports.at(-1)).toBe(encodeProgramStatus('blocked', 'permission'))
    patchOverlayState({ confirm: null })
    await expect.poll(() => reports.at(-1)).toBe(encodeProgramStatus('working'))
    patchUiState({ status: 'A cosmetic update' })
    await pause()
    expect(reports).toHaveLength(3)
  })
})
