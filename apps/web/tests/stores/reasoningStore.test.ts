import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { act, renderHook } from '@testing-library/react'
import { useReasoningStore } from '../../src/stores/reasoningStore'
import * as reasoningService from '../../src/services/reasoningService'

// Mock axios和服务
vi.mock('../../src/services/reasoningService')

const mockReasoningService = reasoningService as any

const mockReasoningChain = {
  id: 'chain-123',
  problem: '测试推理问题',
  strategy: 'ZERO_SHOT',
  steps: [
    {
      id: 'step-1',
      step_number: 1,
      step_type: 'observation',
      content: '观察内容',
      reasoning: '推理过程',
      confidence: 0.8,
      duration_ms: 1200,
    },
  ],
  conclusion: '推理结论',
  confidence_score: 0.85,
  total_duration_ms: 1200,
  created_at: '2025-01-15T10:00:00Z',
  completed_at: '2025-01-15T10:01:00Z',
}

const mockReasoningRequest = {
  problem: '测试问题',
  strategy: 'ZERO_SHOT' as const,
  context: '',
  max_steps: 10,
  stream: false,
  enable_branching: true,
  examples: [],
}

describe('reasoningStore测试', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    // 重置store状态
    const { result } = renderHook(() => useReasoningStore())
    act(() => {
      result.current.setCurrentChain(null)
      result.current.clearStreamingSteps()
    })
  })

  afterEach(() => {
    vi.restoreAllMocks()
  })

  it('初始状态正确', () => {
    const { result } = renderHook(() => useReasoningStore())

    expect(result.current.currentChain).toBeNull()
    expect(result.current.streamingSteps).toEqual([])
    expect(result.current.reasoningHistory).toEqual([])
    expect(result.current.isExecuting).toBe(false)
    expect(result.current.error).toBeNull()
  })

  it('executeReasoning成功执行', async () => {
    mockReasoningService.executeReasoning.mockResolvedValue(mockReasoningChain)

    const { result } = renderHook(() => useReasoningStore())

    let chainResult
    await act(async () => {
      chainResult = await result.current.executeReasoning(mockReasoningRequest)
    })

    expect(mockReasoningService.executeReasoning).toHaveBeenCalledWith(
      mockReasoningRequest
    )
    expect(result.current.currentChain).toEqual(mockReasoningChain)
    expect(result.current.isExecuting).toBe(false)
    expect(result.current.error).toBeNull()
    expect(chainResult).toEqual(mockReasoningChain)
  })

  it('executeReasoning处理错误', async () => {
    const error = new Error('推理执行失败')
    mockReasoningService.executeReasoning.mockRejectedValue(error)

    const { result } = renderHook(() => useReasoningStore())

    await act(async () => {
      try {
        await result.current.executeReasoning(mockReasoningRequest)
      } catch (e) {
        // 预期的错误
      }
    })

    expect(result.current.isExecuting).toBe(false)
    expect(result.current.error).toBe('推理执行失败')
    expect(result.current.currentChain).toBeNull()
  })

  it('streamReasoning正确处理流式数据', async () => {
    const mockEventSource = {
      addEventListener: vi.fn(),
      close: vi.fn(),
      readyState: 1,
    }

    mockReasoningService.streamReasoning.mockReturnValue(mockEventSource)

    const { result } = renderHook(() => useReasoningStore())

    await act(async () => {
      result.current.streamReasoning(mockReasoningRequest)
    })

    expect(mockReasoningService.streamReasoning).toHaveBeenCalledWith(
      mockReasoningRequest
    )
    expect(result.current.isExecuting).toBe(true)

    // 模拟接收到流式数据
    const stepData = {
      id: 'step-1',
      step_number: 1,
      step_type: 'observation',
      content: '流式步骤',
      reasoning: '流式推理',
      confidence: 0.8,
    }

    act(() => {
      const messageCallback = mockEventSource.addEventListener.mock.calls.find(
        call => call[0] === 'step'
      )[1]
      messageCallback({ data: JSON.stringify(stepData) })
    })

    expect(result.current.streamingSteps).toContainEqual(stepData)
  })

  it('流式推理完成时正确处理', async () => {
    const mockEventSource = {
      addEventListener: vi.fn(),
      close: vi.fn(),
      readyState: 1,
    }

    mockReasoningService.streamReasoning.mockReturnValue(mockEventSource)

    const { result } = renderHook(() => useReasoningStore())

    await act(async () => {
      result.current.streamReasoning(mockReasoningRequest)
    })

    // 模拟完成事件
    act(() => {
      const completeCallback = mockEventSource.addEventListener.mock.calls.find(
        call => call[0] === 'complete'
      )[1]
      completeCallback({ data: JSON.stringify(mockReasoningChain) })
    })

    expect(result.current.currentChain).toEqual(mockReasoningChain)
    expect(result.current.isExecuting).toBe(false)
    expect(mockEventSource.close).toHaveBeenCalled()
  })

  it('流式推理错误处理', async () => {
    const mockEventSource = {
      addEventListener: vi.fn(),
      close: vi.fn(),
      readyState: 1,
    }

    mockReasoningService.streamReasoning.mockReturnValue(mockEventSource)

    const { result } = renderHook(() => useReasoningStore())

    await act(async () => {
      result.current.streamReasoning(mockReasoningRequest)
    })

    // 模拟错误事件
    const errorMessage = '流式推理失败'
    act(() => {
      const errorCallback = mockEventSource.addEventListener.mock.calls.find(
        call => call[0] === 'error'
      )[1]
      errorCallback({ data: errorMessage })
    })

    expect(result.current.error).toBe(errorMessage)
    expect(result.current.isExecuting).toBe(false)
    expect(mockEventSource.close).toHaveBeenCalled()
  })

  it('getReasoningHistory正确获取历史记录', async () => {
    const mockHistory = [mockReasoningChain]
    mockReasoningService.getReasoningHistory.mockResolvedValue(mockHistory)

    const { result } = renderHook(() => useReasoningStore())

    await act(async () => {
      await result.current.getReasoningHistory()
    })

    expect(mockReasoningService.getReasoningHistory).toHaveBeenCalled()
    expect(result.current.reasoningHistory).toEqual(mockHistory)
  })

  it('getReasoningChain正确获取特定推理链', async () => {
    mockReasoningService.getReasoningChain.mockResolvedValue(mockReasoningChain)

    const { result } = renderHook(() => useReasoningStore())

    let chainResult
    await act(async () => {
      chainResult = await result.current.getReasoningChain('chain-123')
    })

    expect(mockReasoningService.getReasoningChain).toHaveBeenCalledWith(
      'chain-123'
    )
    expect(chainResult).toEqual(mockReasoningChain)
  })

  it('deleteReasoningChain正确删除推理链', async () => {
    mockReasoningService.deleteReasoningChain.mockResolvedValue(undefined)

    const { result } = renderHook(() => useReasoningStore())

    await act(async () => {
      await result.current.deleteReasoningChain('chain-123')
    })

    expect(mockReasoningService.deleteReasoningChain).toHaveBeenCalledWith(
      'chain-123'
    )
  })

  it('validateChain正确验证推理链', async () => {
    const mockValidation = {
      step_id: 'step-123',
      is_valid: true,
      consistency_score: 0.85,
      issues: [],
      suggestions: [],
    }

    mockReasoningService.validateChain.mockResolvedValue(mockValidation)

    const { result } = renderHook(() => useReasoningStore())

    let validationResult
    await act(async () => {
      validationResult = await result.current.validateChain('chain-123')
    })

    expect(mockReasoningService.validateChain).toHaveBeenCalledWith('chain-123')
    expect(validationResult).toEqual(mockValidation)
  })

  it('recoverChain正确执行恢复', async () => {
    mockReasoningService.recoverChain.mockResolvedValue(true)

    const { result } = renderHook(() => useReasoningStore())

    let recoveryResult
    await act(async () => {
      recoveryResult = await result.current.recoverChain(
        'chain-123',
        'backtrack'
      )
    })

    expect(mockReasoningService.recoverChain).toHaveBeenCalledWith(
      'chain-123',
      'backtrack'
    )
    expect(recoveryResult).toBe(true)
  })

  it('getReasoningStats正确获取统计数据', async () => {
    const mockStats = {
      totalChains: 10,
      avgConfidence: 0.8,
      completionRate: 0.9,
    }

    mockReasoningService.getReasoningStats.mockResolvedValue(mockStats)

    const { result } = renderHook(() => useReasoningStore())

    await act(async () => {
      await result.current.getReasoningStats()
    })

    expect(mockReasoningService.getReasoningStats).toHaveBeenCalled()
    expect(result.current.reasoningStats).toEqual(mockStats)
  })

  it('setCurrentChain正确设置当前链', () => {
    const { result } = renderHook(() => useReasoningStore())

    // 先设置当前链
    act(() => {
      result.current.currentChain = mockReasoningChain
    })

    expect(result.current.currentChain).toEqual(mockReasoningChain)

    act(() => {
      result.current.setCurrentChain(null)
    })

    expect(result.current.currentChain).toBeNull()
  })

  it('clearStreamingSteps正确清除流式步骤', () => {
    const { result } = renderHook(() => useReasoningStore())

    const mockSteps = [
      {
        chain_id: 'chain-123',
        step_number: 1,
        step_type: 'observation',
        content: '测试',
        reasoning: '测试',
        confidence: 0.8,
        is_final: false,
      },
    ]

    // 先设置流式步骤
    act(() => {
      result.current.streamingSteps = mockSteps
    })

    expect(result.current.streamingSteps).toEqual(mockSteps)

    act(() => {
      result.current.clearStreamingSteps()
    })

    expect(result.current.streamingSteps).toEqual([])
  })

  it('并发请求正确处理', async () => {
    mockReasoningService.executeReasoning.mockResolvedValue(mockReasoningChain)

    const { result } = renderHook(() => useReasoningStore())

    // 同时发起两个请求
    const promise1 = act(async () => {
      return result.current.executeReasoning(mockReasoningRequest)
    })

    const promise2 = act(async () => {
      return result.current.executeReasoning(mockReasoningRequest)
    })

    await Promise.all([promise1, promise2])

    // 只应该有一个请求被执行（第二个应该被忽略）
    expect(mockReasoningService.executeReasoning).toHaveBeenCalledTimes(1)
  })
})
