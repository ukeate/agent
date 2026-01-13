import { useCallback } from 'react'
import { createAgentChat, useConversationStore } from '../stores/conversationStore'
import { useAgentStore } from '../stores/agentStore'
import { Message } from '../types'

import { logger } from '../utils/logger'
export const useChat = () => {
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
    currentConversation,
  } = useConversationStore()

  const {
    incrementMessageCount,
    incrementToolCount,
    setStatus,
    setError: setAgentError,
  } = useAgentStore()

  const sendMessage = useCallback(async (content: string) => {
    if (loading) return

    let conversationId = currentConversation?.id
    if (!conversationId) {
      conversationId = await createNewConversation()
    }

    // 创建用户消息
    const userMessage: Message = {
      id: `user-${Date.now()}`,
      content,
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

    try {
      const agentMessage: Message = {
        id: `agent-${Date.now()}`,
        content: '',
        role: 'agent',
        timestamp: new Date().toISOString(),
        toolCalls: [],
        reasoningSteps: [],
      }

      addMessage(agentMessage)

      const response = await createAgentChat(conversationId, content)
      updateLastMessage(response.response)

      setLoading(false)
      setStatus({
        id: 'agent-1',
        name: 'AI助手',
        status: 'idle',
      })
      incrementMessageCount()
      refreshConversations()
    } catch (error) {
      logger.error('发送消息失败:', error)
      
      let errorMessage = '发送消息失败'
      if (error instanceof Error) {
        errorMessage = error.message
      }
      
      // 根据错误类型提供更友好的提示
      if (errorMessage.toLowerCase().includes('network') || 
          errorMessage.toLowerCase().includes('fetch')) {
        errorMessage = '网络连接异常，请检查网络连接后重试'
      } else if (errorMessage.toLowerCase().includes('timeout')) {
        errorMessage = '请求超时，请稍后重试'
      } else if (errorMessage.toLowerCase().includes('401')) {
        errorMessage = '身份验证失败，请重新登录'
      } else if (errorMessage.toLowerCase().includes('403')) {
        errorMessage = '权限不足，无法执行此操作'
      } else if (errorMessage.toLowerCase().includes('500')) {
        errorMessage = '服务器错误，请稍后重试'
      }
      
      setError(errorMessage)
      setAgentError(errorMessage)
      setLoading(false)
      setStatus({
        id: 'agent-1',
        name: 'AI助手',
        status: 'error',
      })

      // 添加错误消息
      const errorAgentMessage: Message = {
        id: `agent-error-${Date.now()}`,
        content: `抱歉，${errorMessage}。你可以稍后重试或检查网络连接。`,
        role: 'agent',
        timestamp: new Date().toISOString(),
      }
      addMessage(errorAgentMessage)
    }
  }, [
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
  ])

  const clearChat = useCallback(async () => {
    try {
      await closeCurrentConversation()
    } catch {}
    clearMessages()
    setLoading(false)
    setError(null)
    setAgentError(null)
    setStatus(null)
  }, [closeCurrentConversation, clearMessages, setLoading, setError, setAgentError, setStatus])

  const startNewConversation = useCallback(async () => {
    await createNewConversation()
    setError(null)
    setAgentError(null)
    setStatus(null)
  }, [createNewConversation, setError, setAgentError, setStatus])

  return {
    messages,
    loading,
    error,
    sendMessage,
    clearChat,
    startNewConversation,
  }
}
