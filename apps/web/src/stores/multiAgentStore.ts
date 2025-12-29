import { buildApiUrl, apiFetch } from '../utils/apiBase'
import { create } from 'zustand'
import { persist } from 'zustand/middleware'
import { CONVERSATION_CONSTANTS } from '../constants/timeout'
import { multiAgentService } from '../services/multiAgentService'

import { logger } from '../utils/logger'
// 多智能体相关类型定义
export interface Agent {
  id: string
  name: string
  role: 'code_expert' | 'architect' | 'doc_expert' | 'supervisor'
  status: 'active' | 'idle' | 'busy' | 'offline'
  capabilities: string[]
  configuration: {
    model: string
    temperature: number
    max_tokens: number
    tools: string[]
    system_prompt: string
  }
  created_at: string
  updated_at: string
}

export interface MultiAgentMessage {
  id: string
  role: string
  sender: string
  content: string
  timestamp: string
  round: number
  isStreaming?: boolean
  streamingComplete?: boolean
}

export interface StreamingMessage {
  id: string
  agentName: string
  content: string
  round: number
  isComplete: boolean
  hasError?: boolean
  error?: string
}

export interface MultiAgentConversation {
  id: string
  title: string
  type: 'multi_agent'
  participants: string[] // Agent IDs
  status: 'created' | 'active' | 'paused' | 'completed' | 'terminated' | 'error'
  metadata: {
    user_context?: string
    task_complexity?: number
    workflow_type?: string
  }
  created_at: string
  updated_at: string
  messages: MultiAgentMessage[]
  round_count: number
  current_speaker_index: number
}

export interface ConversationSession {
  session_id: string
  conversation_id?: string
  status: string
  created_at: string
  updated_at: string
  message_count: number
  round_count: number
  participants: Array<{
    name: string
    role: string
    status: string
  }>
  config: {
    max_rounds: number
    timeout_seconds: number
    auto_reply: boolean
  }
}

interface MultiAgentState {
  // 智能体管理
  agents: Agent[]
  
  // 多智能体对话
  currentSession: ConversationSession | null
  sessions: ConversationSession[]
  
  // 当前对话消息
  currentMessages: MultiAgentMessage[]
  
  // 流式消息状态
  streamingMessages: Record<string, StreamingMessage>
  
  // 当前发言者状态
  currentSpeaker: string | null
  
  // 状态管理
  loading: boolean
  error: string | null
  
  // WebSocket连接状态
  websocketConnected: boolean
  
  // Actions
  setAgents: (agents: Agent[]) => void
  updateAgentStatus: (agentId: string, status: Agent['status']) => void
  
  // 会话管理
  setCurrentSession: (session: ConversationSession | null) => void
  addSession: (session: ConversationSession) => void
  updateSessionStatus: (sessionId: string, status: string) => void
  updateSessionId: (oldSessionId: string, newSessionId: string) => void
  
  // 消息管理
  addMessage: (message: MultiAgentMessage) => void
  clearMessages: () => void
  
  // 流式消息管理
  addStreamingToken: (messageId: string, tokenData: {
    agentName: string
    token: string
    fullContent: string
    round: number
    isComplete: boolean
  }) => void
  completeStreamingMessage: (messageId: string, messageData: {
    agentName: string
    fullContent: string
    round: number
  }) => void
  handleStreamingError: (messageId: string, errorData: {
    agentName: string
    error: string
    fullContent: string
    round: number
  }) => void
  
  // 状态管理
  setLoading: (loading: boolean) => void
  setError: (error: string | null) => void
  setWebsocketConnected: (connected: boolean) => void
  setCurrentSpeaker: (speaker: string | null) => void
  
  // 历史记录管理
  deleteSession: (sessionId: string) => void
  loadSessionHistory: (sessionId: string) => Promise<void>
  getSessionSummary: (session: ConversationSession) => { title: string; preview: string; messageCount: number }
  
  // 智能体操作
  createConversation: (participants: string[], initial_topic?: string) => Promise<void>
  startConversation: (sessionId: string, message: string) => Promise<void>
  pauseConversation: (sessionId: string) => Promise<void>
  resumeConversation: (sessionId: string) => Promise<void>
  terminateConversation: (sessionId: string, reason?: string) => Promise<void>
}

export const useMultiAgentStore = create<MultiAgentState>()(
  persist(
    (set, get) => ({
      // 初始状态
      agents: [],
      currentSession: null,
      sessions: [],
      currentMessages: [],
      streamingMessages: {},
      currentSpeaker: null,
      loading: false,
      error: null,
      websocketConnected: false,

      // 智能体管理
      setAgents: (agents) => set({ agents }),
      
      updateAgentStatus: (agentId, status) => {
        set((state) => ({
          agents: state.agents.map(agent =>
            agent.id === agentId
              ? { ...agent, status, updated_at: new Date().toISOString() }
              : agent
          )
        }))
      },

      // 会话管理
      setCurrentSession: (session) => {
        set({ 
          currentSession: session,
          currentMessages: [],
          error: null 
        })
      },

      addSession: (session) => {
        set((state) => ({
          sessions: [session, ...state.sessions]
        }))
      },

      updateSessionStatus: (sessionId, status) => {
        set((state) => ({
          sessions: state.sessions.map(session =>
            session.session_id === sessionId
              ? { ...session, status, updated_at: new Date().toISOString() }
              : session
          ),
          currentSession: state.currentSession?.session_id === sessionId
            ? { ...state.currentSession, status, updated_at: new Date().toISOString() }
            : state.currentSession
        }))
      },

      // 消息管理
      addMessage: (message) => {
        set((state) => ({
          currentMessages: [...state.currentMessages, message]
        }))
      },

      clearMessages: () => {
        set({ currentMessages: [], streamingMessages: {} })
      },

      // 流式消息管理
      addStreamingToken: (messageId, tokenData) => {
        set((state) => {
          const existingMessage = state.streamingMessages[messageId]
          
          if (existingMessage) {
            // 更新现有流式消息
            return {
              streamingMessages: {
                ...state.streamingMessages,
                [messageId]: {
                  ...existingMessage,
                  content: tokenData.fullContent,
                  isComplete: tokenData.isComplete
                }
              }
            }
          } else {
            // 创建新的流式消息
            const newStreamingMessage: StreamingMessage = {
              id: messageId,
              agentName: tokenData.agentName,
              content: tokenData.fullContent,
              round: tokenData.round,
              isComplete: tokenData.isComplete
            }
            
            // 同时在消息列表中添加一个占位消息用于实时显示
            const placeholderMessage: MultiAgentMessage = {
              id: messageId,
              role: 'assistant',
              sender: tokenData.agentName,
              content: tokenData.fullContent,
              timestamp: new Date().toISOString(),
              round: tokenData.round,
              isStreaming: true,
              streamingComplete: false
            }
            
            return {
              streamingMessages: {
                ...state.streamingMessages,
                [messageId]: newStreamingMessage
              },
              currentMessages: [...state.currentMessages, placeholderMessage]
            }
          }
        })
        
        // 更新消息列表中对应的占位消息
        set((state) => ({
          currentMessages: state.currentMessages.map(msg => 
            msg.id === messageId 
              ? { ...msg, content: tokenData.fullContent }
              : msg
          )
        }))
      },

      completeStreamingMessage: (messageId, messageData) => {
        set((state) => {
          // 完成流式消息
          const updatedStreamingMessages = { ...state.streamingMessages }
          if (updatedStreamingMessages[messageId]) {
            updatedStreamingMessages[messageId] = {
              ...updatedStreamingMessages[messageId],
              content: messageData.fullContent,
              isComplete: true
            }
          }
          
          // 更新消息列表中的消息状态
          const updatedMessages = state.currentMessages.map(msg => 
            msg.id === messageId 
              ? { 
                  ...msg, 
                  content: messageData.fullContent,
                  isStreaming: false,
                  streamingComplete: true
                }
              : msg
          )
          
          return {
            streamingMessages: updatedStreamingMessages,
            currentMessages: updatedMessages
          }
        })
      },

      handleStreamingError: (messageId, errorData) => {
        set((state) => {
          const errorContent = errorData.fullContent || `${errorData.agentName}: 遇到错误 - ${errorData.error}`
          
          const updatedStreamingMessages = {
            ...state.streamingMessages,
            [messageId]: {
              id: messageId,
              agentName: errorData.agentName,
              content: errorContent,
              round: errorData.round,
              isComplete: true,
              hasError: true,
              error: errorData.error
            }
          }
          
          const existing = state.currentMessages.find(msg => msg.id === messageId)
          const updatedMessages = existing
            ? state.currentMessages.map(msg => 
                msg.id === messageId 
                  ? { 
                      ...msg, 
                      content: errorContent,
                      isStreaming: false,
                      streamingComplete: true
                    }
                  : msg
              )
            : [
                ...state.currentMessages,
                {
                  id: messageId,
                  role: 'assistant',
                  sender: errorData.agentName,
                  content: errorContent,
                  timestamp: new Date().toISOString(),
                  round: errorData.round,
                  isStreaming: false,
                  streamingComplete: true
                }
              ]
          
          return {
            streamingMessages: updatedStreamingMessages,
            currentMessages: updatedMessages
          }
        })
      },

      // 状态管理
      setLoading: (loading) => set({ loading }),
      setError: (error) => set({ error }),
      setWebsocketConnected: (connected) => set({ websocketConnected: connected }),
      setCurrentSpeaker: (speaker) => set({ currentSpeaker: speaker }),

      // 历史记录管理
      deleteSession: (sessionId) => {
        set((state) => ({
          sessions: state.sessions.filter(session => session.session_id !== sessionId),
          // 如果删除的是当前会话，清空当前会话
          currentSession: state.currentSession?.session_id === sessionId ? null : state.currentSession,
          currentMessages: state.currentSession?.session_id === sessionId ? [] : state.currentMessages
        }))
      },

      loadSessionHistory: async (sessionId) => {
        set({ loading: true, error: null })
        try {
          // 从本地sessions中找到对应会话
          const { sessions } = get()
          const session = sessions.find(s => s.session_id === sessionId)
          
          if (!session) {
            throw new Error('会话不存在')
          }

          // 尝试从API加载会话的消息历史
          try {
            const messagesData = await multiAgentService.getMessages(sessionId)
            if (messagesData.messages && messagesData.messages.length > 0) {
              // 设置当前会话和消息
              set({ 
                currentSession: session,
                currentMessages: messagesData.messages 
              })
              logger.log(`已加载会话 ${sessionId} 的历史消息`)
              return
            }
          } catch (apiError) {
            logger.warn('从API加载消息失败，使用本地数据:', apiError)
          }

          // 如果API调用失败，使用本地存储的基本会话信息
          set({ 
            currentSession: session,
            currentMessages: [] // 本地没有存储消息，只显示会话信息
          })
          
        } catch (error) {
          logger.error('加载会话历史失败:', error)
          set({ error: error instanceof Error ? error.message : '加载会话历史失败' })
        } finally {
          set({ loading: false })
        }
      },

      getSessionSummary: (session) => {
        // 生成会话标题
        const participantNames = session.participants.map(p => p.name).join(', ')
        const title = `${participantNames} 协作对话`
        
        // 生成预览内容
        const preview = session.status === 'completed' 
          ? `已完成 ${session.round_count} 轮讨论`
          : session.status === 'terminated'
          ? '对话已终止'
          : session.status === 'active'
          ? '对话进行中'
          : '等待开始'
        
        return {
          title,
          preview,
          messageCount: session.message_count || 0
        }
      },

      // 智能体操作 (通过WebSocket创建和管理会话)
      createConversation: async (participants, initial_topic) => {
        set({ loading: true, error: null })
        
        try {
          logger.log('创建对话，参与者:', participants, '主题:', initial_topic)
          
          // 生成临时的session ID用于WebSocket连接
          const tempSessionId = `session-${crypto.randomUUID()}`
          
          // 创建session对象（临时）
          const session: ConversationSession = {
            session_id: tempSessionId,
            status: 'created',
            created_at: new Date().toISOString(),
            updated_at: new Date().toISOString(),
            message_count: 1,
            round_count: 0,
            participants: participants.map(id => {
              // 根据ID找到对应的智能体信息
              const agents = get().agents
              const agent = agents.find(a => a.id === id)
              return {
                name: agent?.name || '未知智能体',
                role: agent?.role || 'assistant',
                status: 'active'
              }
            }),
            config: {
              max_rounds: CONVERSATION_CONSTANTS.DEFAULT_MAX_ROUNDS,
              timeout_seconds: CONVERSATION_CONSTANTS.DEFAULT_TIMEOUT_SECONDS,
              auto_reply: CONVERSATION_CONSTANTS.DEFAULT_AUTO_REPLY
            }
          }
          
          set({ currentSession: session })
          get().addSession(session)
          
          // 清空现有消息，准备接收实时消息
          set({ currentMessages: [] })
          
          // 添加用户初始消息
          get().addMessage({
            id: `msg-${Date.now()}-1`,
            role: 'user',
            sender: '用户',
            content: initial_topic || '开始多智能体协作讨论',
            timestamp: new Date().toISOString(),
            round: 0,
          })
          
          logger.log('临时会话已创建，等待WebSocket连接并启动实际对话...')
          
          logger.log('会话创建完成，WebSocket连接建立后将自动启动对话')
          
        } catch (error) {
          logger.error('创建对话失败:', error)
          let errorMessage = '创建对话失败'
          
          if (error instanceof Error) {
            errorMessage = error.message
          }
          
          set({ error: errorMessage })
        } finally {
          set({ loading: false })
        }
      },

      // 更新会话ID（从WebSocket接收真实conversation_id后调用）
      updateSessionId: (oldSessionId: string, newSessionId: string) => {
        set((state) => {
          let updatedCurrentSession = state.currentSession
          
          // 更新当前会话的ID并同时设置状态为active
          if (state.currentSession?.session_id === oldSessionId) {
            updatedCurrentSession = {
              ...state.currentSession,
              session_id: newSessionId,
              status: 'active',  // 同时更新状态为active
              updated_at: new Date().toISOString()
            }
          }
          
          // 更新会话列表中的ID并同时设置状态为active
          const updatedSessions = state.sessions.map(session =>
            session.session_id === oldSessionId
              ? { 
                  ...session, 
                  session_id: newSessionId, 
                  status: 'active',  // 同时更新状态为active
                  updated_at: new Date().toISOString() 
                }
              : session
          )
          
          logger.log(`会话ID已更新并设置为active: ${oldSessionId} -> ${newSessionId}`)
          
          return {
            currentSession: updatedCurrentSession,
            sessions: updatedSessions
          }
        })
      },

      startConversation: async (sessionId, message) => {
        set({ loading: true, error: null })
        try {
          logger.log('启动对话:', sessionId, '消息:', message)
          
          // 多智能体对话通常在创建时就开始，这里可能不需要额外的启动API
          // 但我们可以通过WebSocket发送消息来启动对话
          const { currentSession } = get()
          if (currentSession) {
            // 更新会话状态为活跃
            get().updateSessionStatus(sessionId, 'active')
            
            // 添加用户消息到对话历史
            get().addMessage({
              id: `msg-${Date.now()}`,
              role: 'user',
              sender: '用户',
              content: message,
              timestamp: new Date().toISOString(),
              round: 1,
            })
          }
        } catch (error) {
          logger.error('启动对话失败:', error)
          set({ error: error instanceof Error ? error.message : '启动对话失败' })
        } finally {
          set({ loading: false })
        }
      },

      pauseConversation: async (sessionId) => {
        set({ loading: true, error: null })
        get().updateSessionStatus(sessionId, 'paused')
        try {
          logger.log('暂停对话，sessionId:', sessionId)
          logger.log('当前会话状态:', get().currentSession)
          
          // 通过REST API调用后端暂停功能
          const conversationId = get().currentSession?.conversation_id || sessionId
          const url = buildApiUrl(`/multi-agent/conversation/${conversationId}/pause`)
          logger.log('暂停请求URL:', url)
          
          const response = await apiFetch(url, {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
            },
          })
          
          logger.log('暂停响应状态:', response.status)
          
          const result = await response.json()
          logger.log('后端暂停响应:', result)
          
          // 更新本地状态
          logger.log('对话已暂停 (已通知后端停止token生成)')
        } catch (error) {
          logger.error('暂停对话失败:', error)
          get().updateSessionStatus(sessionId, 'active')
          set({ error: error instanceof Error ? error.message : '暂停对话失败' })
        } finally {
          set({ loading: false })
        }
      },

      resumeConversation: async (sessionId) => {
        set({ loading: true, error: null })
        get().updateSessionStatus(sessionId, 'active')
        try {
          logger.log('恢复对话:', sessionId)
          
          // 通过REST API调用后端恢复功能
          const conversationId = get().currentSession?.conversation_id || sessionId
          const response = await apiFetch(buildApiUrl(`/multi-agent/conversation/${conversationId}/resume`), {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
            },
          })
          
          const result = await response.json()
          logger.log('后端恢复响应:', result)
          
          // 更新本地状态
          logger.log('对话已恢复 (已通知后端继续token生成)')
        } catch (error) {
          logger.error('恢复对话失败:', error)
          get().updateSessionStatus(sessionId, 'paused')
          set({ error: error instanceof Error ? error.message : '恢复对话失败' })
        } finally {
          set({ loading: false })
        }
      },

      terminateConversation: async (sessionId, reason = '用户终止') => {
        set({ loading: true, error: null })
        try {
          logger.log('终止对话:', sessionId, '原因:', reason)
          
          // 通过REST API调用后端终止功能
          const conversationId = get().currentSession?.conversation_id || sessionId
          const response = await apiFetch(buildApiUrl(`/multi-agent/conversation/${conversationId}/terminate`), {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
            },
            body: JSON.stringify({ reason }),
          })
          
          const result = await response.json()
          logger.log('后端终止响应:', result)
          
          // 更新本地状态
          get().updateSessionStatus(sessionId, 'terminated')
          // 清理当前会话
          set({ currentSession: null, currentMessages: [] })
          
          logger.log('对话已终止 (已通知后端完全停止token生成)')
        } catch (error) {
          logger.error('终止对话失败:', error)
          set({ error: error instanceof Error ? error.message : '终止对话失败' })
        } finally {
          set({ loading: false })
        }
      },
    }),
    {
      name: 'multi-agent-store',
      // 只持久化基本状态，不持久化连接状态
      partialize: (state) => ({
        agents: state.agents,
        sessions: state.sessions,
        currentSessionId: state.currentSession?.session_id || null,
      }),
      onRehydrateStorage: () => (state) => {
        if (state) {
          const currentSessionId = (state as any).currentSessionId
          if (currentSessionId) {
            const session = state.sessions.find(s => s.session_id === currentSessionId)
            if (session) {
              state.currentSession = session
            }
          }
          // 重置连接状态和流式消息状态
          state.websocketConnected = false
          state.loading = false
          state.error = null
          state.streamingMessages = {}
        }
      },
    }
  )
)
