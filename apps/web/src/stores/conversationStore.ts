import { create } from 'zustand'
import { apiClient } from '../services/apiClient'
import { apiFetch } from '../utils/apiBase'
import { consumeSseJson } from '../utils/sse'
import { Conversation, Message, ToolCall } from '../types'
import { logger } from '../utils/logger'

const isCanceledError = (error: unknown) => {
  if (!error || typeof error !== 'object') return false
  const err = error as { code?: string; name?: string; message?: string }
  if (err.code === 'ERR_CANCELED' || err.name === 'CanceledError') return true
  if (typeof err.message === 'string' && err.message.toLowerCase() === 'canceled')
    return true
  return false
}

let conversationListRequestId = 0
let conversationListController: AbortController | null = null
let conversationDetailRequestId = 0
let conversationDetailController: AbortController | null = null

const LAST_CONVERSATION_STORAGE_KEY = 'ai-agent-last-conversation'

const hasStorage = () =>
  typeof window !== 'undefined' && typeof window.localStorage !== 'undefined'

const readStoredConversationId = (): string | null => {
  if (!hasStorage()) return null
  const stored = window.localStorage.getItem(LAST_CONVERSATION_STORAGE_KEY)
  if (!stored) return null
  const normalized = stored.trim()
  return normalized ? normalized : null
}

const writeStoredConversationId = (conversationId?: string | null) => {
  if (!hasStorage()) return
  if (conversationId) {
    window.localStorage.setItem(LAST_CONVERSATION_STORAGE_KEY, conversationId)
    return
  }
  window.localStorage.removeItem(LAST_CONVERSATION_STORAGE_KEY)
}

const clearStoredConversationId = () => {
  writeStoredConversationId(null)
}

type AgentSessionResponse = {
  conversation_id: string
}

export type AgentStreamStep = {
  conversation_id: string
  step_id?: string
  step_type:
    | 'thought'
    | 'action'
    | 'observation'
    | 'final_answer'
    | 'streaming_token'
    | 'error'
  content?: string
  timestamp?: number | string
  tool_name?: string
  tool_args?: Record<string, unknown>
  error?: string
}

type ConversationSummary = {
  conversation_id: string
  title: string
  created_at: string
  updated_at: string
  status?: string
  message_stats?: {
    total?: number
    user?: number
  }
  last_message?: {
    id?: string
    content?: string
    sender_type?: string
    created_at?: string
  }
}

type ListConversationsResponse = {
  conversations: ConversationSummary[]
  total: number
  limit?: number
  offset?: number
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

export const normalizeToolCalls = (
  toolCalls?: Array<Record<string, any>>
): ToolCall[] => {
  if (!toolCalls || toolCalls.length === 0) return []
  const now = new Date().toISOString()
  return toolCalls.map((toolCall, index) => {
    const status =
      toolCall.status === 'pending' || toolCall.status === 'error'
        ? toolCall.status
        : toolCall.error
          ? 'error'
          : 'success'
    const rawTimestamp = toolCall.timestamp
    const timestamp = rawTimestamp
      ? new Date(
          typeof rawTimestamp === 'number' ? rawTimestamp * 1000 : rawTimestamp
        ).toISOString()
      : now
    return {
      id: String(toolCall.step_id ?? toolCall.id ?? `${Date.now()}-${index}`),
      name: toolCall.tool_name ?? toolCall.name ?? '工具调用',
      args: toolCall.tool_args ?? toolCall.args ?? {},
      result: toolCall.result ?? toolCall.tool_result ?? toolCall.output,
      status,
      timestamp,
    }
  })
}

const mapConversationSummary = (s: ConversationSummary): Conversation => ({
  id: s.conversation_id,
  title: s.title || '对话',
  messages: [],
  createdAt: s.created_at,
  updatedAt: s.updated_at,
  messageCount: s.message_stats?.total || 0,
  userMessageCount: s.message_stats?.user || 0,
  status: s.status,
  lastMessage: s.last_message
    ? {
        id: s.last_message.id || `last-${s.conversation_id}`,
        content: s.last_message.content || '',
        role: s.last_message.sender_type === 'user' ? 'user' : 'agent',
        timestamp: s.last_message.created_at || s.updated_at,
      }
    : undefined,
})

const mapHistoryToConversation = (
  history: ConversationHistoryResponse
): Conversation => {
  const summary = history.summary || {}
  const messages: Message[] = (history.messages || []).map(m => ({
    id: m.id,
    content: m.content,
    role: m.sender_type === 'user' ? 'user' : 'agent',
    timestamp: m.created_at,
    toolCalls: normalizeToolCalls(m.tool_calls),
  }))

  return {
    id: history.conversation_id,
    title: summary.title || '对话',
    messages,
    createdAt: summary.created_at,
    updatedAt: summary.updated_at,
    messageCount: summary.message_stats?.total,
    userMessageCount: summary.message_stats?.user,
    status: summary.status,
    lastMessage: summary.last_message
      ? {
          id: summary.last_message.id || `last-${history.conversation_id}`,
          content: summary.last_message.content || '',
          role: summary.last_message.sender_type === 'user' ? 'user' : 'agent',
          timestamp: summary.last_message.created_at || summary.updated_at,
        }
      : undefined,
  }
}

interface ConversationState {
  currentConversation: Conversation | null
  conversations: Conversation[]
  conversationTotal: number
  messages: Message[]
  loading: boolean
  error: string | null
  historyLoading: boolean
  historyError: string | null

  addMessage: (message: Message) => void
  updateLastMessage: (
    content: string,
    update?: Pick<Message, 'toolCalls' | 'reasoningSteps'> & {
      replace?: boolean
    }
  ) => void
  clearMessages: () => void
  setLoading: (loading: boolean) => void
  setError: (error: string | null) => void
  setHistoryError: (error: string | null) => void

  refreshConversations: (options?: {
    limit?: number
    offset?: number
    append?: boolean
    query?: string
  }) => Promise<{ items: Conversation[]; hasMore: boolean }>
  createNewConversation: (title?: string) => Promise<string>
  loadConversation: (conversationId: string) => Promise<void>
  renameConversation: (conversationId: string, title: string) => Promise<void>
  deleteConversation: (
    conversationId: string,
    options?: { refresh?: boolean }
  ) => Promise<void>
  closeCurrentConversation: () => Promise<void>
  restoreLastConversation: () => Promise<void>
}

export const useConversationStore = create<ConversationState>((set, get) => ({
  currentConversation: null,
  conversations: [],
  conversationTotal: 0,
  messages: [],
  loading: false,
  error: null,
  historyLoading: false,
  historyError: null,

  addMessage: message => {
    set(state => {
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

  updateLastMessage: (content, update) => {
    set(state => {
      if (state.messages.length === 0) return state
      const messages = [...state.messages]
      const last = messages[messages.length - 1]
      const next: Message = {
        ...last,
        content: update?.replace ? content : last.content + content,
      }
      if (update?.toolCalls !== undefined) {
        next.toolCalls = update.toolCalls
      }
      if (update?.reasoningSteps !== undefined) {
        next.reasoningSteps = update.reasoningSteps
      }
      messages[messages.length - 1] = next
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

  clearMessages: () => {
    clearStoredConversationId()
    set({ messages: [], currentConversation: null, error: null })
  },

  setLoading: loading => set({ loading }),
  setError: error => set({ error }),
  setHistoryError: error => set({ historyError: error }),

  refreshConversations: async options => {
    const limit = options?.limit ?? 20
    const offset = options?.offset ?? 0
    const append = options?.append ?? false
    const query = options?.query?.trim()
    const requestId = (conversationListRequestId += 1)
    if (conversationListController) {
      conversationListController.abort()
    }
    const controller = new AbortController()
    conversationListController = controller
    const mergeConversations = (
      existing: Conversation[],
      incoming: Conversation[]
    ) => {
      if (existing.length === 0) return incoming
      const map = new Map(existing.map(item => [item.id, item]))
      incoming.forEach(item => map.set(item.id, item))
      return Array.from(map.values())
    }
    try {
      if (!append) {
        set({ historyLoading: true })
      }
      const response = await apiClient.get<ListConversationsResponse>(
        '/agents/conversations',
        {
          params: { limit, offset, ...(query ? { query } : {}) },
          signal: controller.signal,
        }
      )
      if (requestId !== conversationListRequestId) {
        return { items: [], hasMore: false }
      }
      const list = response.data?.conversations || []
      const total =
        typeof response.data?.total === 'number' ? response.data.total : null
      const next = list.map(mapConversationSummary)
      const merged = append
        ? mergeConversations(get().conversations, next)
        : next
      merged.sort((a, b) => {
        const timeA = Date.parse(a.updatedAt || a.createdAt)
        const timeB = Date.parse(b.updatedAt || b.createdAt)
        return (Number.isNaN(timeB) ? 0 : timeB) -
          (Number.isNaN(timeA) ? 0 : timeA)
      })
      const totalForState = Math.max(total ?? 0, merged.length)
      set({
        conversations: merged,
        historyError: null,
        conversationTotal: totalForState,
      })
      const hasMore =
        total !== null ? offset + list.length < total : list.length === limit
      return { items: next, hasMore }
    } catch (e: any) {
      if (requestId !== conversationListRequestId || isCanceledError(e)) {
        return { items: [], hasMore: false }
      }
      set({ historyError: e?.message || '加载对话列表失败' })
      return { items: [], hasMore: false }
    } finally {
      if (!append && requestId === conversationListRequestId) {
        set({ historyLoading: false })
      }
      if (conversationListController === controller) {
        conversationListController = null
      }
    }
  },

  createNewConversation: async title => {
    try {
      const conversationTitle = title?.trim()
      const response = await apiClient.post<AgentSessionResponse>(
        '/agents/sessions',
        {
          agent_type: 'react',
          ...(conversationTitle
            ? { conversation_title: conversationTitle }
            : {}),
        }
      )
      const conversationId = response.data.conversation_id
      const now = new Date().toISOString()
      set({
        currentConversation: {
          id: conversationId,
          title: conversationTitle || '新对话',
          messages: [],
          createdAt: now,
          updatedAt: now,
        },
        messages: [],
        error: null,
      })
      writeStoredConversationId(conversationId)
      await get().refreshConversations()
      return conversationId
    } catch (e: any) {
      set({ error: e?.message || '创建对话失败' })
      throw e
    }
  },

  loadConversation: async conversationId => {
    const requestId = (conversationDetailRequestId += 1)
    if (conversationDetailController) {
      conversationDetailController.abort()
    }
    const controller = new AbortController()
    conversationDetailController = controller
    try {
      const response = await apiClient.get<ConversationHistoryResponse>(
        `/agents/conversations/${conversationId}/history`,
        { signal: controller.signal }
      )
      if (requestId !== conversationDetailRequestId) return
      const conversation = mapHistoryToConversation(response.data)
      set({
        currentConversation: conversation,
        messages: conversation.messages,
        historyError: null,
      })
      writeStoredConversationId(conversationId)
    } catch (e: any) {
      if (requestId !== conversationDetailRequestId || isCanceledError(e)) {
        return
      }
      set({ historyError: e?.message || '加载对话失败' })
      throw e
    } finally {
      if (conversationDetailController === controller) {
        conversationDetailController = null
      }
    }
  },

  renameConversation: async (conversationId, title) => {
    const normalized = title.trim()
    if (!normalized) {
      throw new Error('对话标题不能为空')
    }
    const response = await apiClient.put<{
      conversation_id: string
      title: string
      updated_at: string
    }>(`/agents/conversations/${conversationId}/title`, { title: normalized })
    const updatedAt = response.data?.updated_at
    set(state => ({
      conversations: state.conversations.map(conversation =>
        conversation.id === conversationId
          ? {
              ...conversation,
              title: response.data?.title || normalized,
              updatedAt: updatedAt || conversation.updatedAt,
            }
          : conversation
      ),
      currentConversation:
        state.currentConversation?.id === conversationId
          ? {
              ...state.currentConversation,
              title: response.data?.title || normalized,
              updatedAt: updatedAt || state.currentConversation.updatedAt,
            }
          : state.currentConversation,
    }))
  },

  deleteConversation: async (conversationId, options) => {
    await apiClient.delete(`/agents/conversations/${conversationId}`)
    if (readStoredConversationId() === conversationId) {
      clearStoredConversationId()
    }
    if (get().currentConversation?.id === conversationId) {
      set({ currentConversation: null, messages: [] })
    }
    if (options?.refresh === false) return
    await get().refreshConversations()
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
  restoreLastConversation: async () => {
    const storedId = readStoredConversationId()
    if (!storedId) return
    if (get().currentConversation?.id === storedId) return
    try {
      await get().loadConversation(storedId)
    } catch {
      clearStoredConversationId()
    }
  },
}))

export const streamAgentChat = async (
  conversationId: string,
  message: string,
  onStep: (step: AgentStreamStep) => void,
  onComplete?: () => void,
  signal?: AbortSignal
) => {
  const response = await apiFetch(`/agents/react/chat/${conversationId}`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Accept: 'text/event-stream',
    },
    signal,
    body: JSON.stringify({ message, stream: true }),
  })
  await consumeSseJson<AgentStreamStep>(response, onStep, {
    onDone: onComplete,
    onParseError: error => {
      logger.warn('解析流式消息失败', error)
    },
  })
}
