import { useCallback } from 'react'
import { useConversationStore } from '../stores/conversationStore'
import { useAgentStore } from '../stores/agentStore'
import { apiClient } from '../services/apiClient'
import { Message } from '../types'

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
    saveConversation,
    createNewConversation,
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

    // 如果没有当前对话，先创建一个新对话
    if (!currentConversation) {
      createNewConversation()
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
      // 使用流式API
      const agentMessage: Message = {
        id: `agent-${Date.now()}`,
        content: '',
        role: 'agent',
        timestamp: new Date().toISOString(),
        toolCalls: [],
        reasoningSteps: [],
      }

      addMessage(agentMessage)

      await apiClient.sendMessageStream(
        {
          message: content,
          stream: true,
        },
        // onMessage callback - 处理OpenAI标准格式
        (data) => {
          // 处理OpenAI标准格式的流式响应
          if (data.object === 'chat.completion.chunk') {
            const choice = data.choices?.[0]
            if (choice?.delta?.content) {
              // 追加内容到最后一条消息
              updateLastMessage(choice.delta.content)
            }
            
            // 检查是否完成
            if (choice?.finish_reason === 'stop') {
              // 流式传输已完成
              setLoading(false)
              setStatus({
                id: 'agent-1',
                name: 'AI助手',
                status: 'idle',
              })
              incrementMessageCount()
              saveConversation()
            }
          }
          // 错误处理
          else if (data.error) {
            console.error('Stream error:', data.error)
            setError(data.error.message || '发生未知错误')
            setAgentError(data.error.message || '发生未知错误')
            setLoading(false)
            setStatus({
              id: 'agent-1',
              name: 'AI助手',
              status: 'error', 
            })
          }
        },
        // onError callback
        (error) => {
          console.error('Chat stream error:', error)
          setError(error.message)
          setAgentError(error.message)
          setLoading(false)
          setStatus({
            id: 'agent-1',
            name: 'AI助手',
            status: 'error',
          })
        },
        // onComplete callback - 仅处理流结束清理
        () => {
          // 如果还在加载状态，说明可能没有正常收到完成信号
          if (loading) {
            setLoading(false)
            setStatus({
              id: 'agent-1',
              name: 'AI助手', 
              status: 'idle',
            })
            incrementMessageCount()
            saveConversation()
          }
        }
      )
    } catch (error) {
      console.error('Failed to send message:', error)
      
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
    saveConversation,
    createNewConversation,
  ])

  const clearChat = useCallback(() => {
    clearMessages()
    setError(null)
    setAgentError(null)
    setStatus(null)
  }, [clearMessages, setError, setAgentError, setStatus])

  const startNewConversation = useCallback(() => {
    createNewConversation()
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