import { PassThrough } from 'node:stream'

import { renderSync } from '@hermes/ink'
import React from 'react'
import { expect, it, vi } from 'vitest'

import { ActiveSessionSwitcher } from '../components/activeSessionSwitcher.js'
import { DEFAULT_THEME } from '../theme.js'

it.each([
  { model: null, label: 'model?' },
  { model: undefined, label: 'model?' },
  { model: '', label: 'model?' },
  { model: 'provider/example-model', label: 'example-model' },
  { model: 'plain-model', label: 'plain-model' }
])('canonical sessions render and select owner rows with model $model', async ({ model, label }) => {
  const request = vi.fn(async (method: string) => {
    if (method !== 'session.list') {
      throw new Error('invalid_params')
    }

    return {
      scope: 'live',
      sessions: [
        {
          id: 'owner',
          session_id: 'owner',
          title: 'Owned conversation',
          model,
          started_at: 1,
          message_count: 2,
          running: true
        }
      ]
    }
  })

  const stdin = Object.assign(new PassThrough(), { isTTY: true, setRawMode: vi.fn(), ref: vi.fn(), unref: vi.fn() })
  let output = ''
  const stdout = Object.assign(new PassThrough(), { columns: 100, rows: 30, isTTY: false })
  stdout.on('data', chunk => {
    output += chunk.toString()
  })
  const onSelect = vi.fn()

  const instance = renderSync(
    <ActiveSessionSwitcher
      currentSessionId="owner"
      gw={{ isCanonical: true, request } as any}
      onCancel={vi.fn()}
      onClose={vi.fn()}
      onNew={vi.fn()}
      onNewPrompt={vi.fn()}
      onResume={vi.fn()}
      onSelect={onSelect}
      t={DEFAULT_THEME}
    />,
    { stdin: stdin as any, stdout: stdout as any, stderr: new PassThrough() as any, patchConsole: false }
  )

  try {
    await new Promise(resolve => setImmediate(resolve))
    instance.rerender(
      <ActiveSessionSwitcher
        currentSessionId="owner"
        gw={{ isCanonical: true, request } as any}
        onCancel={vi.fn()}
        onClose={vi.fn()}
        onNew={vi.fn()}
        onNewPrompt={vi.fn()}
        onResume={vi.fn()}
        onSelect={onSelect}
        t={DEFAULT_THEME}
      />
    )
    await new Promise(resolve => setImmediate(resolve))
    expect(request.mock.calls.every(([method]) => method === 'session.list')).toBe(true)
    expect(output).toContain('Owned conversation')
    expect(output).toContain('working')
    expect(output).toContain(label)
    stdin.write('\r')
    await new Promise(resolve => setImmediate(resolve))
    expect(onSelect).toHaveBeenCalledWith('owner')
  } finally {
    instance.unmount()
  }
})
