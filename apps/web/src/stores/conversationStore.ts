import { create } from 'zustand'
import { apiClient } from '../services/apiClient'
import { Conversation, Message } from '../types'

type AgentSessionResponse = {
  conversation_id: string
}

type AgentChatResponse = {
  conversation_id: string
  response: string
  steps: number
  tool_calls: any[]
  completed: boolean
  session_summary?: Record<string, any>
}

type ConversationSummary = {
  conversation_id: string
  title: string
  created_at: string
  updated_at: string
  message_stats?: {
    total?: number
    user?: number
  }
}

type ListConversationsResponse = {
  conversations: ConversationSummary[]
}

type ConversationHistoryResponse = {
  conversation_id: string
  messages: Array<{
    id: string
    content: string
    sender_type: string
    created_at: string
    tool_calls?: any[]
  }>
  summary: any
}

const mapConversationSummary = (s: ConversationSummary): Conversation => ({
  id: s.conversation_id,
  title: s.title || '对话',
  messages: [],
  createdAt: s.created_at,
  updatedAt: s.updated_at,
  messageCount: s.message_stats?.total || 0,
  userMessageCount: s.message_stats?.user || 0,
})

const mapHistoryToConversation = (history: ConversationHistoryResponse): Conversation => {
  const summary = history.summary || {}
  const messages: Message[] = (history.messages || []).map((m) => ({
    id: m.id,
    content: m.content,
    role: m.sender_type === 'user' ? 'user' : 'agent',
    timestamp: m.created_at,
    toolCalls: m.tool_calls,
  }))

  return {
    id: history.conversation_id,
    title: summary.title || '对话',
    messages,
    createdAt: summary.created_at,
    updatedAt: summary.updated_at,
    messageCount: summary.message_stats?.total,
    userMessageCount: summary.message_stats?.user,
  }
}

interface ConversationState {
  currentConversation: Conversation | null
  conversations: Conversation[]
  messages: Message[]
  loading: boolean
  error: string | null

  addMessage: (message: Message) => void
  updateLastMessage: (content: string) => void
  clearMessages: () => void
  setLoading: (loading: boolean) => void
  setError: (error: string | null) => void

  refreshConversations: () => Promise<void>
  createNewConversation: () => Promise<string>
  loadConversation: (conversationId: string) => Promise<void>
  deleteConversation: (conversationId: string) => Promise<void>
  closeCurrentConversation: () => Promise<void>
}

export const useConversationStore = create<ConversationState>((set, get) => ({
  currentConversation: null,
  conversations: [],
  messages: [],
  loading: false,
  error: null,

  addMessage: (message) => {
    set((state) => {
      const messages = [...state.messages, message]
      const currentConversation = state.currentConversation
        ? {
            ...state.currentConversation,
            messages,
            updatedAt: new Date().toISOString(),
          }
        : null
      return { messages, currentConversation }
    })
  },

  updateLastMessage: (content) => {
    set((state) => {
      if (state.messages.length === 0) return state
      const messages = [...state.messages]
      const last = messages[messages.length - 1]
      messages[messages.length - 1] = { ...last, content: last.content + content }
      const currentConversation = state.currentConversation
        ? { ...state.currentConversation, messages, updatedAt: new Date().toISOString() }
        : null
      return { messages, currentConversation }
    })
  },

  clearMessages: () => set({ messages: [], currentConversation: null, error: null }),

  setLoading: (loading) => set({ loading }),
  setError: (error) => set({ error }),

  refreshConversations: async () => {
    try {
      const response = await apiClient.get<ListConversationsResponse>('/agents/conversations')
      const list = response.data?.conversations || []
      set({ conversations: list.map(mapConversationSummary) })
    } catch (e: any) {
      set({ error: e?.message || '加载对话列表失败' })
    }
  },

  createNewConversation: async () => {
    try {
      const response = await apiClient.post<AgentSessionResponse>('/agents/sessions', {
        agent_type: 'react',
      })
      const conversationId = response.data.conversation_id
      const now = new Date().toISOString()
      set({
        currentConversation: {
          id: conversationId,
          title: '新对话',
          messages: [],
          createdAt: now,
          updatedAt: now,
        },
        messages: [],
        error: null,
      })
      await get().refreshConversations()
      return conversationId
    } catch (e: any) {
      set({ error: e?.message || '创建对话失败' })
      throw e
    }
  },

  loadConversation: async (conversationId) => {
    try {
      const response = await apiClient.get<ConversationHistoryResponse>(
        `/agents/conversations/${conversationId}/history`
      )
      const conversation = mapHistoryToConversation(response.data)
      set({
        currentConversation: conversation,
        messages: conversation.messages,
        error: null,
      })
    } catch (e: any) {
      set({ error: e?.message || '加载对话失败' })
      throw e
    }
  },

  deleteConversation: async (conversationId) => {
    try {
      await apiClient.delete(`/agents/conversations/${conversationId}`)
      await get().refreshConversations()
      if (get().currentConversation?.id === conversationId) {
        set({ currentConversation: null, messages: [] })
      }
    } catch (e: any) {
      set({ error: e?.message || '关闭对话失败' })
      throw e
    }
  },

  closeCurrentConversation: async () => {
    const id = get().currentConversation?.id
    if (!id) {
      set({ currentConversation: null, messages: [] })
      return
    }
    await get().deleteConversation(id)
    set({ currentConversation: null, messages: [] })
  },
}))

export const createAgentChat = async (conversationId: string, message: string) => {
  const response = await apiClient.post<AgentChatResponse>(`/agents/react/chat/${conversationId}`, {
    message,
    stream: false,
  })
  return response.data
}
