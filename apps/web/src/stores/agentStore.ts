import { create } from 'zustand'
import { AgentStatus } from '../types'

interface AgentState {
  // 智能体状态
  status: AgentStatus | null

  // 连接状态
  connected: boolean

  // 错误状态
  error: string | null

  // 统计信息
  stats: {
    totalMessages: number
    totalTools: number
    averageResponseTime: number
  }

  // Actions
  setStatus: (status: AgentStatus | null) => void
  setConnected: (connected: boolean) => void
  setError: (error: string | null) => void
  updateStats: (updates: Partial<AgentState['stats']>) => void
  incrementMessageCount: (durationMs?: number) => void
  incrementToolCount: () => void
  resetStats: () => void
}

export const useAgentStore = create<AgentState>(set => ({
  // 初始状态
  status: null,
  connected: false,
  error: null,
  stats: {
    totalMessages: 0,
    totalTools: 0,
    averageResponseTime: 0,
  },

  // Actions
  setStatus: status => set({ status, error: null }),

  setConnected: connected => set({ connected }),

  setError: error => set({ error }),

  updateStats: updates =>
    set(state => ({
      stats: { ...state.stats, ...updates },
    })),

  incrementMessageCount: (durationMs?: number) =>
    set(state => {
      const nextTotal = state.stats.totalMessages + 1
      const nextAverage =
        typeof durationMs === 'number' && Number.isFinite(durationMs)
          ? (state.stats.averageResponseTime * state.stats.totalMessages +
              durationMs) /
            nextTotal
          : state.stats.averageResponseTime
      return {
        stats: {
          ...state.stats,
          totalMessages: nextTotal,
          averageResponseTime: nextAverage,
        },
      }
    }),

  incrementToolCount: () =>
    set(state => ({
      stats: {
        ...state.stats,
        totalTools: state.stats.totalTools + 1,
      },
    })),

  resetStats: () =>
    set({
      stats: {
        totalMessages: 0,
        totalTools: 0,
        averageResponseTime: 0,
      },
    }),
}))
