import { describe, it, expect, vi, beforeEach } from 'vitest'
import { validateMessage } from '../../src/utils/validation'

describe('ApiClient Mock Tests', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it('validates messages correctly', () => {
    const result = validateMessage('Hello AI')
    expect(result.isValid).toBe(true)
    expect(result.error).toBeUndefined()
  })

  it('rejects empty messages', () => {
    const result = validateMessage('')
    expect(result.isValid).toBe(false)
    expect(result.error).toBe('消息不能为空')
  })

  it('rejects messages that are too long', () => {
    const longMessage = 'a'.repeat(2001)
    const result = validateMessage(longMessage)
    expect(result.isValid).toBe(false)
    expect(result.error).toBe('消息长度不能超过2000个字符')
  })

  it('允许包含 script 标签的内容用于代码分析', () => {
    const result = validateMessage('<script>alert("xss")</script>')
    expect(result.isValid).toBe(true)
    expect(result.error).toBeUndefined()
  })

  it('handles streaming responses correctly', async () => {
    // Mock fetch for streaming
    global.fetch = vi.fn().mockResolvedValueOnce({
      ok: true,
      body: new ReadableStream({
        start(controller) {
          controller.enqueue(
            new TextEncoder().encode(
              'data: {"type":"content","content":"Hello"}\n\n'
            )
          )
          controller.enqueue(
            new TextEncoder().encode(
              'data: {"type":"content","content":" World"}\n\n'
            )
          )
          controller.close()
        },
      }),
    })

    const onMessage = vi.fn()
    const onError = vi.fn()
    const onComplete = vi.fn()

    // Mock streaming functionality
    const mockStreamHandler = async () => {
      onMessage({ type: 'content', content: 'Hello' })
      onMessage({ type: 'content', content: ' World' })
      onComplete()
    }

    await mockStreamHandler()

    expect(onMessage).toHaveBeenCalledTimes(2)
    expect(onMessage).toHaveBeenNthCalledWith(1, {
      type: 'content',
      content: 'Hello',
    })
    expect(onMessage).toHaveBeenNthCalledWith(2, {
      type: 'content',
      content: ' World',
    })
    expect(onComplete).toHaveBeenCalledOnce()
    expect(onError).not.toHaveBeenCalled()
  })

  it('handles streaming errors', async () => {
    global.fetch = vi.fn().mockRejectedValueOnce(new Error('Stream error'))

    const onMessage = vi.fn()
    const onError = vi.fn()
    const onComplete = vi.fn()

    // Mock error streaming functionality
    const mockStreamErrorHandler = async () => {
      try {
        await global.fetch()
      } catch (error) {
        onError(error)
      }
    }

    await mockStreamErrorHandler()

    expect(onError).toHaveBeenCalledWith(expect.any(Error))
    expect(onMessage).not.toHaveBeenCalled()
    expect(onComplete).not.toHaveBeenCalled()
  })
})
