/**
 * Supervisor状态管理
 * 使用Zustand进行状态管理
 */

import { create } from 'zustand'
import { devtools } from 'zustand/middleware'
import {
  SupervisorStatusResponse,
  SupervisorTask,
  SupervisorDecision,
  SupervisorStats,
  SupervisorConfig,
  AgentLoadMetrics,
  TaskSubmissionRequest,
  TaskAssignmentResponse
} from '../types/supervisor'
import { supervisorApiService } from '../services/supervisorApi'

interface SupervisorStore {
  // 状态数据
  currentSupervisorId: string | null
  status: SupervisorStatusResponse | null
  tasks: SupervisorTask[]
  selectedTask: SupervisorTask | null
  decisions: SupervisorDecision[]
  stats: SupervisorStats | null
  config: SupervisorConfig | null
  agentMetrics: AgentLoadMetrics[]
  
  // UI状态
  loading: {
    status: boolean
    tasks: boolean
    taskDetails: boolean
    decisions: boolean
    stats: boolean
    config: boolean
    metrics: boolean
    submitting: boolean
  }
  error: string | null
  
  // 分页信息
  pagination: {
    tasks: {
      page: number
      pageSize: number
      total: number
      totalPages: number
    }
    decisions: {
      page: number
      pageSize: number
      total: number
      totalPages: number
    }
  }
  
  // 自动刷新
  autoRefresh: boolean
  refreshInterval: number
  
  // Actions
  setSupervisorId: (id: string) => void
  loadStatus: () => Promise<void>
  loadTasks: (page?: number, pageSize?: number) => Promise<void>
  loadTaskDetails: (taskId: string) => Promise<void>
  loadDecisions: (page?: number, pageSize?: number) => Promise<void>
  loadStats: () => Promise<void>
  loadConfig: () => Promise<void>
  loadMetrics: () => Promise<void>
  submitTask: (taskRequest: TaskSubmissionRequest) => Promise<TaskAssignmentResponse>
  updateConfig: (config: Partial<SupervisorConfig>) => Promise<void>
  refreshAll: () => Promise<void>
  setAutoRefresh: (enabled: boolean) => void
  setRefreshInterval: (interval: number) => void
  clearError: () => void
  reset: () => void
}

export const useSupervisorStore = create<SupervisorStore>()(
  devtools(
    (set, get) => ({
      // 初始状态
      currentSupervisorId: null,
      status: null,
      tasks: [],
      selectedTask: null,
      decisions: [],
      stats: null,
      config: null,
      agentMetrics: [],
      
      loading: {
        status: false,
        tasks: false,
        taskDetails: false,
        decisions: false,
        stats: false,
        config: false,
        metrics: false,
        submitting: false,
      },
      error: null,
      
      pagination: {
        tasks: {
          page: 1,
          pageSize: 20,
          total: 0,
          totalPages: 0,
        },
        decisions: {
          page: 1,
          pageSize: 20,
          total: 0,
          totalPages: 0,
        },
      },
      
      autoRefresh: false,
      refreshInterval: 10000, // 10秒
      
      // Actions
      setSupervisorId: (id: string) => {
        set({ currentSupervisorId: id, error: null })
      },
      
      loadStatus: async () => {
        const { currentSupervisorId } = get()
        if (!currentSupervisorId) {
          set({ error: 'Supervisor ID未设置' })
          return
        }
        
        set({ loading: { ...get().loading, status: true }, error: null })
        
        try {
          const status = await supervisorApiService.getStatus(currentSupervisorId)
          set({ status, loading: { ...get().loading, status: false } })
        } catch (error) {
          set({
            error: error instanceof Error ? error.message : '加载状态失败',
            loading: { ...get().loading, status: false }
          })
        }
      },
      
      loadTasks: async (page = 1, pageSize = 20) => {
        const { currentSupervisorId } = get()
        if (!currentSupervisorId) {
          set({ error: 'Supervisor ID未设置' })
          return
        }
        
        set({ loading: { ...get().loading, tasks: true }, error: null })
        
        try {
          const result = await supervisorApiService.getTasks(currentSupervisorId, page, pageSize)
          set({
            tasks: result.tasks,
            pagination: {
              ...get().pagination,
              tasks: {
                page: result.page,
                pageSize: result.pageSize,
                total: result.total,
                totalPages: result.totalPages,
              }
            },
            loading: { ...get().loading, tasks: false }
          })
        } catch (error) {
          set({
            error: error instanceof Error ? error.message : '加载任务失败',
            loading: { ...get().loading, tasks: false }
          })
        }
      },

      loadTaskDetails: async (taskId: string) => {
        set({ loading: { ...get().loading, taskDetails: true }, error: null })
        
        try {
          const taskDetails = await supervisorApiService.getTask(taskId)
          set({
            selectedTask: taskDetails,
            loading: { ...get().loading, taskDetails: false }
          })
        } catch (error) {
          set({
            error: error instanceof Error ? error.message : '加载任务详情失败',
            loading: { ...get().loading, taskDetails: false }
          })
        }
      },
      
      loadDecisions: async (page = 1, pageSize = 20) => {
        const { currentSupervisorId } = get()
        if (!currentSupervisorId) {
          set({ error: 'Supervisor ID未设置' })
          return
        }
        
        set({ loading: { ...get().loading, decisions: true }, error: null })
        
        try {
          const result = await supervisorApiService.getDecisionHistory(currentSupervisorId, page, pageSize)
          set({
            decisions: result.decisions,
            pagination: {
              ...get().pagination,
              decisions: {
                page: result.page,
                pageSize: result.pageSize,
                total: result.total,
                totalPages: result.totalPages,
              }
            },
            loading: { ...get().loading, decisions: false }
          })
        } catch (error) {
          set({
            error: error instanceof Error ? error.message : '加载决策历史失败',
            loading: { ...get().loading, decisions: false }
          })
        }
      },
      
      loadStats: async () => {
        const { currentSupervisorId } = get()
        if (!currentSupervisorId) {
          set({ error: 'Supervisor ID未设置' })
          return
        }
        
        set({ loading: { ...get().loading, stats: true }, error: null })
        
        try {
          const stats = await supervisorApiService.getStats(currentSupervisorId)
          set({ stats, loading: { ...get().loading, stats: false } })
        } catch (error) {
          set({
            error: error instanceof Error ? error.message : '加载统计数据失败',
            loading: { ...get().loading, stats: false }
          })
        }
      },
      
      loadConfig: async () => {
        const { currentSupervisorId } = get()
        if (!currentSupervisorId) {
          set({ error: 'Supervisor ID未设置' })
          return
        }
        
        set({ loading: { ...get().loading, config: true }, error: null })
        
        try {
          const config = await supervisorApiService.getConfig(currentSupervisorId)
          set({ config, loading: { ...get().loading, config: false } })
        } catch (error) {
          set({
            error: error instanceof Error ? error.message : '加载配置失败',
            loading: { ...get().loading, config: false }
          })
        }
      },
      
      loadMetrics: async () => {
        const { currentSupervisorId } = get()
        if (!currentSupervisorId) {
          set({ error: 'Supervisor ID未设置' })
          return
        }
        
        set({ loading: { ...get().loading, metrics: true }, error: null })
        
        try {
          const agentMetrics = await supervisorApiService.getAgentMetrics(currentSupervisorId)
          set({ agentMetrics, loading: { ...get().loading, metrics: false } })
        } catch (error) {
          set({
            error: error instanceof Error ? error.message : '加载智能体指标失败',
            loading: { ...get().loading, metrics: false }
          })
        }
      },
      
      submitTask: async (taskRequest: TaskSubmissionRequest): Promise<TaskAssignmentResponse> => {
        const { currentSupervisorId } = get()
        if (!currentSupervisorId) {
          throw new Error('Supervisor ID未设置')
        }
        
        set({ loading: { ...get().loading, submitting: true }, error: null })
        
        try {
          const result = await supervisorApiService.submitTask(currentSupervisorId, taskRequest)
          
          // 提交成功后刷新任务列表和统计数据
          get().loadTasks()
          get().loadStats()
          
          set({ loading: { ...get().loading, submitting: false } })
          return result
        } catch (error) {
          set({
            error: error instanceof Error ? error.message : '任务提交失败',
            loading: { ...get().loading, submitting: false }
          })
          throw error
        }
      },
      
      updateConfig: async (configUpdate: Partial<SupervisorConfig>) => {
        const { currentSupervisorId } = get()
        if (!currentSupervisorId) {
          set({ error: 'Supervisor ID未设置' })
          return
        }
        
        set({ loading: { ...get().loading, config: true }, error: null })
        
        try {
          const config = await supervisorApiService.updateConfig(currentSupervisorId, configUpdate)
          set({ config, loading: { ...get().loading, config: false } })
        } catch (error) {
          set({
            error: error instanceof Error ? error.message : '更新配置失败',
            loading: { ...get().loading, config: false }
          })
        }
      },
      
      refreshAll: async () => {
        const { currentSupervisorId } = get()
        if (!currentSupervisorId) return
        
        // 并行加载所有数据
        await Promise.allSettled([
          get().loadStatus(),
          get().loadTasks(),
          get().loadDecisions(),
          get().loadStats(),
          get().loadConfig(),
          get().loadMetrics(),
        ])
      },
      
      setAutoRefresh: (enabled: boolean) => {
        set({ autoRefresh: enabled })
      },
      
      setRefreshInterval: (interval: number) => {
        set({ refreshInterval: interval })
      },
      
      clearError: () => {
        set({ error: null })
      },
      
      reset: () => {
        set({
          currentSupervisorId: null,
          status: null,
          tasks: [],
          selectedTask: null,
          decisions: [],
          stats: null,
          config: null,
          agentMetrics: [],
          loading: {
            status: false,
            tasks: false,
            taskDetails: false,
            decisions: false,
            stats: false,
            config: false,
            metrics: false,
            submitting: false,
          },
          error: null,
          pagination: {
            tasks: {
              page: 1,
              pageSize: 20,
              total: 0,
              totalPages: 0,
            },
            decisions: {
              page: 1,
              pageSize: 20,
              total: 0,
              totalPages: 0,
            },
          },
          autoRefresh: false,
        })
      },
    }),
    {
      name: 'supervisor-store',
    }
  )
)