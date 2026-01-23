import { describe, it, expect, beforeEach, vi } from 'vitest'
import { copyToClipboard } from '@/utils/clipboard'

describe('copyToClipboard', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it('写入剪贴板内容', async () => {
    await copyToClipboard('hello')
    expect(navigator.clipboard.writeText).toHaveBeenCalledWith('hello')
  })
})
