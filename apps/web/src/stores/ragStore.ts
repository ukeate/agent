/**
 * RAG系统状态管理
 *
 * 统一管理基础RAG和Agentic RAG的状态
 */

import { create } from 'zustand'
import { persist } from 'zustand/middleware'
import {
  QueryRequest,
  QueryResponse,
  // AgenticQueryRequest,
  AgenticQueryResponse,
  KnowledgeItem,
  QueryAnalysisInfo,
  StatsResponse,
} from '../services/ragService'

// ==================== 基础类型定义 ====================

export interface QueryHistory {
  id: string
  query: string
  timestamp: Date
  type: 'basic' | 'agentic'
  results_count: number
  processing_time: number
}

export interface SearchPreferences {
  default_search_type: 'semantic' | 'keyword' | 'hybrid'
  default_limit: number
  default_score_threshold: number
  save_history: boolean
  auto_expand_queries: boolean
}

export interface IndexStatus {
  status: 'healthy' | 'unhealthy' | 'unknown'
  total_documents: number
  total_vectors: number
  index_size: number
  last_updated: string
  is_loading: boolean
}

export interface AgenticSession {
  id: string
  name: string
  created_at: Date
  last_active: Date
  query_count: number
  context_history: string[]
}

// ==================== 状态接口定义 ====================

interface RagState {
  // ==================== 基础RAG状态 ====================

  // 查询状态
  currentQuery: string
  isQuerying: boolean
  queryResults: KnowledgeItem[]
  queryHistory: QueryHistory[]
  lastQuery: QueryRequest | null
  lastResponse: QueryResponse | null

  // 搜索偏好
  searchPreferences: SearchPreferences

  // 索引状态
  indexStatus: IndexStatus

  // 错误处理
  error: string | null

  // ==================== Agentic RAG状态 ====================

  // 智能查询状态
  isAgenticQuerying: boolean
  agenticResults: AgenticQueryResponse | null
  currentSession: AgenticSession | null
  sessions: AgenticSession[]

  // 查询分析
  queryAnalysis: QueryAnalysisInfo | null
  expandedQueries: string[]

  // 检索过程
  retrievalProgress: {
    current_step: string
    progress: number
    message: string
    stage:
      | 'analysis'
      | 'expansion'
      | 'retrieval'
      | 'validation'
      | 'composition'
      | 'explanation'
      | 'complete'
  } | null

  // 解释和反馈
  showExplanation: boolean
  explanationData: any | null
  feedbackData: {
    query_id: string
    ratings: Record<string, number>
    comments: string
  } | null
}

// ==================== 状态行为接口 ====================

interface RagActions {
  // ==================== 基础RAG操作 ====================

  setCurrentQuery: (query: string) => void
  setQueryResults: (results: KnowledgeItem[]) => void
  addToHistory: (
    query: string,
    type: 'basic' | 'agentic',
    results_count: number,
    processing_time: number
  ) => void
  clearHistory: () => void
  setIsQuerying: (loading: boolean) => void
  setError: (error: string | null) => void

  // 搜索偏好管理
  updateSearchPreferences: (preferences: Partial<SearchPreferences>) => void
  resetSearchPreferences: () => void

  // 索引状态管理
  updateIndexStatus: (stats: StatsResponse) => void
  setIndexLoading: (loading: boolean) => void

  // ==================== Agentic RAG操作 ====================

  // 智能查询管理
  setIsAgenticQuerying: (loading: boolean) => void
  setAgenticResults: (results: AgenticQueryResponse | null) => void

  // 会话管理
  createSession: (name?: string) => string
  switchSession: (sessionId: string) => void
  updateSession: (sessionId: string, updates: Partial<AgenticSession>) => void
  deleteSession: (sessionId: string) => void
  addToSessionHistory: (sessionId: string, query: string) => void

  // 查询分析管理
  setQueryAnalysis: (analysis: QueryAnalysisInfo | null) => void
  setExpandedQueries: (queries: string[]) => void

  // 检索过程管理
  setRetrievalProgress: (progress: RagState['retrievalProgress']) => void
  clearRetrievalProgress: () => void

  // 解释和反馈管理
  setShowExplanation: (show: boolean) => void
  setExplanationData: (data: any | null) => void
  setFeedbackData: (data: RagState['feedbackData']) => void
  clearFeedback: () => void

  // ==================== 通用操作 ====================

  reset: () => void
  clearErrors: () => void
}

// ==================== 默认值定义 ====================

const defaultSearchPreferences: SearchPreferences = {
  default_search_type: 'hybrid',
  default_limit: 10,
  default_score_threshold: 0.7,
  save_history: true,
  auto_expand_queries: true,
}

const defaultIndexStatus: IndexStatus = {
  status: 'unknown',
  total_documents: 0,
  total_vectors: 0,
  index_size: 0,
  last_updated: '',
  is_loading: false,
}

// ==================== Zustand Store ====================

export const useRagStore = create<RagState & RagActions>()(
  persist(
    (set, get) => ({
      // ==================== 基础RAG初始状态 ====================

      currentQuery: '',
      isQuerying: false,
      queryResults: [],
      queryHistory: [],
      lastQuery: null,
      lastResponse: null,

      searchPreferences: defaultSearchPreferences,
      indexStatus: defaultIndexStatus,
      error: null,

      // ==================== Agentic RAG初始状态 ====================

      isAgenticQuerying: false,
      agenticResults: null,
      currentSession: null,
      sessions: [],

      queryAnalysis: null,
      expandedQueries: [],

      retrievalProgress: null,

      showExplanation: false,
      explanationData: null,
      feedbackData: null,

      // ==================== 基础RAG操作实现 ====================

      setCurrentQuery: query => set({ currentQuery: query }),

      setQueryResults: results => set({ queryResults: results }),

      addToHistory: (query, type, results_count, processing_time) => {
        const { queryHistory, searchPreferences } = get()
        if (!searchPreferences.save_history) return

        const newHistory: QueryHistory = {
          id: Date.now().toString(),
          query,
          timestamp: new Date(),
          type,
          results_count,
          processing_time,
        }

        const updatedHistory = [newHistory, ...queryHistory].slice(0, 50) // 保留最近50条
        set({ queryHistory: updatedHistory })
      },

      clearHistory: () => set({ queryHistory: [] }),

      setIsQuerying: loading => set({ isQuerying: loading }),

      setError: error => set({ error }),

      updateSearchPreferences: preferences => {
        const current = get().searchPreferences
        set({ searchPreferences: { ...current, ...preferences } })
      },

      resetSearchPreferences: () =>
        set({ searchPreferences: defaultSearchPreferences }),

      updateIndexStatus: stats => {
        if (stats.success) {
          set({
            indexStatus: {
              status: 'healthy',
              total_documents: stats.stats.total_documents,
              total_vectors: stats.stats.total_vectors,
              index_size: stats.stats.index_size,
              last_updated: stats.stats.last_updated,
              is_loading: false,
            },
          })
        } else {
          set({
            indexStatus: {
              ...get().indexStatus,
              status: 'unhealthy',
              is_loading: false,
            },
          })
        }
      },

      setIndexLoading: loading => {
        const current = get().indexStatus
        set({ indexStatus: { ...current, is_loading: loading } })
      },

      // ==================== Agentic RAG操作实现 ====================

      setIsAgenticQuerying: loading => set({ isAgenticQuerying: loading }),

      setAgenticResults: results => set({ agenticResults: results }),

      createSession: name => {
        const sessionId = Date.now().toString()
        const newSession: AgenticSession = {
          id: sessionId,
          name: name || `会话 ${sessionId}`,
          created_at: new Date(),
          last_active: new Date(),
          query_count: 0,
          context_history: [],
        }

        const { sessions } = get()
        set({
          sessions: [newSession, ...sessions],
          currentSession: newSession,
        })

        return sessionId
      },

      switchSession: sessionId => {
        const { sessions } = get()
        const session = sessions.find(s => s.id === sessionId)
        if (session) {
          set({ currentSession: session })
        }
      },

      updateSession: (sessionId, updates) => {
        const { sessions, currentSession } = get()
        const updatedSessions = sessions.map(session =>
          session.id === sessionId
            ? { ...session, ...updates, last_active: new Date() }
            : session
        )

        set({
          sessions: updatedSessions,
          currentSession:
            currentSession?.id === sessionId
              ? { ...currentSession, ...updates, last_active: new Date() }
              : currentSession,
        })
      },

      deleteSession: sessionId => {
        const { sessions, currentSession } = get()
        const updatedSessions = sessions.filter(s => s.id !== sessionId)

        set({
          sessions: updatedSessions,
          currentSession:
            currentSession?.id === sessionId
              ? updatedSessions[0] || null
              : currentSession,
        })
      },

      addToSessionHistory: (sessionId, query) => {
        const { sessions } = get()
        const session = sessions.find(s => s.id === sessionId)
        if (session) {
          const updatedHistory = [...session.context_history, query].slice(-20) // 保留最近20条
          get().updateSession(sessionId, {
            context_history: updatedHistory,
            query_count: session.query_count + 1,
          })
        }
      },

      setQueryAnalysis: analysis => set({ queryAnalysis: analysis }),

      setExpandedQueries: queries => set({ expandedQueries: queries }),

      setRetrievalProgress: progress => set({ retrievalProgress: progress }),

      clearRetrievalProgress: () => set({ retrievalProgress: null }),

      setShowExplanation: show => set({ showExplanation: show }),

      setExplanationData: data => set({ explanationData: data }),

      setFeedbackData: data => set({ feedbackData: data }),

      clearFeedback: () => set({ feedbackData: null }),

      // ==================== 通用操作实现 ====================

      reset: () =>
        set({
          currentQuery: '',
          isQuerying: false,
          queryResults: [],
          lastQuery: null,
          lastResponse: null,
          error: null,

          isAgenticQuerying: false,
          agenticResults: null,
          queryAnalysis: null,
          expandedQueries: [],
          retrievalProgress: null,
          showExplanation: false,
          explanationData: null,
          feedbackData: null,
        }),

      clearErrors: () => set({ error: null }),
    }),
    {
      name: 'rag-store',
      // 只持久化用户偏好和历史记录
      partialize: state => ({
        queryHistory: state.queryHistory,
        searchPreferences: state.searchPreferences,
        sessions: state.sessions,
      }),
    }
  )
)

export default useRagStore
