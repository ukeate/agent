import { useCallback, useEffect, useRef } from 'react'
import {
  streamAgentChat,
  useConversationStore,
  AgentStreamStep,
} from '../stores/conversationStore'
import { useAgentStore } from '../stores/agentStore'
import { Message, ReasoningStep, ToolCall } from '../types'
import { HttpError } from '../utils/apiBase'

import { logger } from '../utils/logger'

const buildErrorMessage = (error: unknown, fallback: string) => {
  if (error instanceof HttpError) {
    if (error.status === 401) return '身份验证失败，请重新登录'
    if (error.status === 403) return '权限不足，无法执行此操作'
    if (error.status === 404) return '请求资源不存在'
    if (error.status >= 500) return '服务器错误，请稍后重试'
  }
  let errorMessage = fallback
  if (typeof error === 'string' && error.trim()) {
    errorMessage = error
  } else if (error instanceof Error && error.message) {
    errorMessage = error.message
  }
  const lowerCase = errorMessage.toLowerCase()
  if (lowerCase.includes('network') || lowerCase.includes('fetch')) {
    return '网络连接异常，请检查网络连接后重试'
  }
  if (lowerCase.includes('timeout')) {
    return '请求超时，请稍后重试'
  }
  if (lowerCase.includes('401')) {
    return '身份验证失败，请重新登录'
  }
  if (lowerCase.includes('403')) {
    return '权限不足，无法执行此操作'
  }
  if (lowerCase.includes('500')) {
    return '服务器错误，请稍后重试'
  }
  return errorMessage
}

const buildConversationTitle = (content: string) => {
  const normalized = content.replace(/\s+/g, ' ').trim()
  if (!normalized) return '新对话'
  const limit = 24
  if (normalized.length <= limit) return normalized
  return `${normalized.slice(0, limit)}...`
}

const isAbortError = (error: unknown) => {
  if (!error || typeof error !== 'object') return false
  if ('name' in error && error.name === 'AbortError') return true
  return false
}

const getNow = () => {
  if (typeof performance !== 'undefined' && typeof performance.now === 'function') {
    return performance.now()
  }
  return Date.now()
}

export const useChat = () => {
  const lastUserMessageRef = useRef<string | null>(null)
  const abortControllerRef = useRef<AbortController | null>(null)
  const {
    messages,
    loading,
    error,
    addMessage,
    updateLastMessage,
    clearMessages,
    setLoading,
    setError,
    createNewConversation,
    refreshConversations,
    closeCurrentConversation,
    restoreLastConversation,
    currentConversation,
  } = useConversationStore()

  const {
    incrementMessageCount,
    incrementToolCount,
    setStatus,
    setError: setAgentError,
  } = useAgentStore()

  const stopStreaming = useCallback(() => {
    abortControllerRef.current?.abort()
  }, [])

  useEffect(() => {
    return () => {
      abortControllerRef.current?.abort()
    }
  }, [])

  useEffect(() => {
    if (currentConversation || messages.length > 0) return
    restoreLastConversation().catch(error => {
      logger.warn('恢复对话失败', error)
    })
  }, [currentConversation, messages.length, restoreLastConversation])

  useEffect(() => {
    if (messages.length === 0) {
      lastUserMessageRef.current = null
      return
    }
    for (let i = messages.length - 1; i >= 0; i -= 1) {
      const message = messages[i]
      if (message.role === 'user' && message.content?.trim()) {
        lastUserMessageRef.current = message.content
        return
      }
    }
    lastUserMessageRef.current = null
  }, [messages])

  const sendMessage = useCallback(
    async (content: string) => {
      if (loading) return
      const trimmed = content.trim()
      if (!trimmed) return
      lastUserMessageRef.current = trimmed
      const requestStart = getNow()

      let conversationId = currentConversation?.id
      if (!conversationId) {
        try {
          conversationId = await createNewConversation(
            buildConversationTitle(trimmed)
          )
        } catch (error) {
          const errorMessage = buildErrorMessage(error, '创建对话失败')
          setError(errorMessage)
          setAgentError(errorMessage)
          setStatus({
            id: 'agent-1',
            name: 'AI助手',
            status: 'error',
          })
          return
        }
      }

      // 创建用户消息
      const userMessage: Message = {
        id: `user-${Date.now()}`,
        content: trimmed,
        role: 'user',
        timestamp: new Date().toISOString(),
      }

      addMessage(userMessage)
      setLoading(true)
      setError(null)
      setAgentError(null)

      // 更新智能体状态
      setStatus({
        id: 'agent-1',
        name: 'AI助手',
        status: 'thinking',
        currentTask: '处理用户请求',
      })

      const agentMessage: Message = {
        id: `agent-${Date.now()}`,
        content: '',
        role: 'agent',
        timestamp: new Date().toISOString(),
        toolCalls: [],
        reasoningSteps: [],
      }

      addMessage(agentMessage)

      let toolCalls: ToolCall[] = []
      let reasoningSteps: ReasoningStep[] = []
      let hasStreamedAnswer = false
      let finished = false

      const updateAgentMessage = (
        contentUpdate: string,
        options?: { replace?: boolean }
      ) => {
        const update =
          toolCalls.length > 0 || reasoningSteps.length > 0 || options?.replace
            ? {
                ...(toolCalls.length > 0 ? { toolCalls } : {}),
                ...(reasoningSteps.length > 0 ? { reasoningSteps } : {}),
                ...(options?.replace ? { replace: true } : {}),
              }
            : undefined
        updateLastMessage(contentUpdate, update)
      }

      const finalize = (status: 'idle' | 'error') => {
        if (finished) return
        finished = true
        setLoading(false)
        setStatus({
          id: 'agent-1',
          name: 'AI助手',
          status,
        })
      }

      const complete = () => {
        if (finished) return
        if (toolCalls.some(toolCall => toolCall.status === 'pending')) {
          toolCalls = toolCalls.map(toolCall =>
            toolCall.status === 'pending'
              ? {
                  ...toolCall,
                  status: 'error',
                  result: toolCall.result || '未返回工具结果',
                }
              : toolCall
          )
          updateAgentMessage('')
        }
        finalize('idle')
        const durationMs = Math.max(0, getNow() - requestStart)
        incrementMessageCount(durationMs)
        refreshConversations()
      }

      const handleError = (rawError: unknown) => {
        const errorMessage = buildErrorMessage(rawError, '发送消息失败')
        const displayMessage = `抱歉，${errorMessage}。你可以稍后重试或检查网络连接。`
        toolCalls = toolCalls.map(toolCall =>
          toolCall.status === 'pending'
            ? { ...toolCall, status: 'error', result: displayMessage }
            : toolCall
        )
        if (hasStreamedAnswer) {
          updateAgentMessage('')
        } else {
          updateAgentMessage(displayMessage, { replace: true })
        }
        setError(errorMessage)
        setAgentError(errorMessage)
        finalize('error')
        refreshConversations()
      }

      const handleAbort = () => {
        if (finished) return
        const hasPending = toolCalls.some(
          toolCall => toolCall.status === 'pending'
        )
        if (hasPending) {
          toolCalls = toolCalls.map(toolCall =>
            toolCall.status === 'pending'
              ? {
                  ...toolCall,
                  status: 'error',
                  result: toolCall.result || '用户已停止生成',
                }
              : toolCall
          )
          updateAgentMessage('')
        }
        finalize('idle')
        refreshConversations()
      }

      const handleStreamStep = (step: AgentStreamStep) => {
        if (!step) return
        if (step.step_type === 'error' || step.error) {
          handleError(step.error || step.content || '发送消息失败')
          return
        }

        if (step.step_type === 'streaming_token') {
          hasStreamedAnswer = true
          updateAgentMessage(step.content || '')
          return
        }

        if (step.step_type === 'final_answer') {
          if (step.content) {
            updateAgentMessage(step.content, { replace: true })
            return
          }
          updateAgentMessage('')
          return
        }

        if (
          step.step_type === 'thought' ||
          step.step_type === 'action' ||
          step.step_type === 'observation'
        ) {
          const stepTimestamp = new Date().toISOString()
          const nextStep: ReasoningStep = {
            id: step.step_id || `step-${Date.now()}`,
            type: step.step_type,
            content: step.content || '',
            timestamp: stepTimestamp,
          }
          reasoningSteps = [...reasoningSteps, nextStep]
          if (
            step.step_type === 'thought' ||
            step.step_type === 'observation'
          ) {
            setStatus({
              id: 'agent-1',
              name: 'AI助手',
              status: 'thinking',
              currentTask:
                step.step_type === 'observation' ? '分析工具结果' : '推理中',
            })
          }
        }

        if (step.step_type === 'action') {
          const toolTimestamp = new Date().toISOString()
          const toolCall: ToolCall = {
            id: step.step_id || `tool-${Date.now()}`,
            name: step.tool_name || step.content || '工具调用',
            args: step.tool_args || {},
            status: 'pending',
            timestamp: toolTimestamp,
          }
          toolCalls = [...toolCalls, toolCall]
          incrementToolCount()
          setStatus({
            id: 'agent-1',
            name: 'AI助手',
            status: 'acting',
            currentTask: toolCall.name,
          })
        }

        if (step.step_type === 'observation' && toolCalls.length > 0) {
          const lastIndex = toolCalls.length - 1
          toolCalls = toolCalls.map((toolCall, index) => {
            if (index !== lastIndex || toolCall.status !== 'pending')
              return toolCall
            return {
              ...toolCall,
              status: 'success',
              result: step.content || toolCall.result,
            }
          })
        }

        updateAgentMessage('')
      }

      if (abortControllerRef.current) {
        abortControllerRef.current.abort()
      }
      const abortController = new AbortController()
      abortControllerRef.current = abortController

      try {
        await streamAgentChat(
          conversationId,
          trimmed,
          handleStreamStep,
          complete,
          abortController.signal
        )
        complete()
      } catch (error) {
        if (isAbortError(error)) {
          handleAbort()
          return
        }
        logger.error('发送消息失败:', error)
        handleError(error)
      } finally {
        if (abortControllerRef.current === abortController) {
          abortControllerRef.current = null
        }
      }
    },
    [
      loading,
      currentConversation,
      addMessage,
      updateLastMessage,
      setLoading,
      setError,
      setAgentError,
      setStatus,
      incrementMessageCount,
      incrementToolCount,
      createNewConversation,
      refreshConversations,
      streamAgentChat,
    ]
  )

  const clearChat = useCallback(async () => {
    stopStreaming()
    try {
      await closeCurrentConversation()
      setError(null)
      setAgentError(null)
      clearMessages()
      setStatus(null)
      lastUserMessageRef.current = null
    } catch (error) {
      const errorMessage = buildErrorMessage(error, '关闭对话失败')
      setError(errorMessage)
      setAgentError(errorMessage)
    }
  }, [
    stopStreaming,
    closeCurrentConversation,
    clearMessages,
    setError,
    setAgentError,
    setStatus,
  ])

  const startNewConversation = useCallback(() => {
    stopStreaming()
    clearMessages()
    setLoading(false)
    setError(null)
    setAgentError(null)
    setStatus(null)
    lastUserMessageRef.current = null
  }, [
    clearMessages,
    setLoading,
    setError,
    setAgentError,
    setStatus,
    stopStreaming,
  ])

  const retryLastMessage = useCallback(() => {
    if (!lastUserMessageRef.current) return
    sendMessage(lastUserMessageRef.current)
  }, [sendMessage])

  const dismissError = useCallback(() => {
    setError(null)
    setAgentError(null)
  }, [setError, setAgentError])

  return {
    currentConversation,
    messages,
    loading,
    error,
    sendMessage,
    stopStreaming,
    retryLastMessage,
    dismissError,
    clearChat,
    startNewConversation,
  }
}
