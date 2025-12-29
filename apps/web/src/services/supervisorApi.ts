/**
 * Supervisor API服务（仅调用后端真实接口，不做任何静态/模拟回填）
 */

import {
  SupervisorStatusResponse,
  SupervisorTask,
  SupervisorDecision,
  TaskSubmissionRequest,
  TaskAssignmentResponse,
  SupervisorConfig,
  SupervisorStats,
  AgentLoadMetrics,
} from '../types/supervisor'
import { apiFetchJson } from '../utils/apiBase'

type ApiResponse<T> = {
  success: boolean
  message: string
  data: T
  timestamp: string
  pagination?: any
}

const API_BASE = '/supervisor'

class SupervisorApiService {
  private async request<T>(url: string, init?: RequestInit): Promise<ApiResponse<T>> {
    const body = await apiFetchJson<ApiResponse<T>>(url, init)
    if (body?.success === false) {
      throw new Error(body.message || '请求失败')
    }
    return body
  }

  /**
   * 获取Supervisor状态
   */
  async getStatus(supervisorId: string): Promise<SupervisorStatusResponse> {
    const resp = await this.request<any>(
      `${API_BASE}/status?supervisor_id=${encodeURIComponent(supervisorId)}`,
    )
    const { current_config, ...rest } = resp.data || {}
    return { ...rest, configuration: current_config } as SupervisorStatusResponse
  }

  /**
   * 提交任务给Supervisor分配
   */
  async submitTask(
    supervisorId: string, 
    taskRequest: TaskSubmissionRequest
  ): Promise<TaskAssignmentResponse> {
    const resp = await this.request<any>(`${API_BASE}/tasks?supervisor_id=${encodeURIComponent(supervisorId)}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(taskRequest),
    })

    const data = resp.data || {}
    const alternatives = Array.isArray(data?.decision_metadata?.alternatives)
      ? data.decision_metadata.alternatives.map((alt: any) => ({
          agent: String(alt?.agent_name ?? alt?.agent ?? ''),
          score: Number(alt?.match_score ?? alt?.score ?? 0),
          reason: String(alt?.reason ?? ''),
        }))
      : undefined

    return {
      task_id: String(data.task_id),
      assigned_agent: String(data.assigned_agent),
      assignment_reason: String(data.assignment_reason),
      confidence_level: Number(data.confidence_level),
      estimated_completion_time: data.estimated_completion_time ? String(data.estimated_completion_time) : undefined,
      alternatives_considered: alternatives,
    }
  }

  /**
   * 获取任务列表
   */
  async getTasks(supervisorId: string, page = 1, pageSize = 20): Promise<{
    tasks: SupervisorTask[]
    total: number
    page: number
    pageSize: number
    totalPages: number
  }> {
    const offset = Math.max(0, (page - 1) * pageSize)
    const resp = await this.request<{ tasks: any[]; pagination: { total: number } }>(
      `${API_BASE}/tasks?supervisor_id=${encodeURIComponent(supervisorId)}&limit=${pageSize}&offset=${offset}`,
    )

    const backendTasks = resp.data?.tasks || []
    const tasks: SupervisorTask[] = backendTasks.map((t: any) => ({
      id: String(t.id),
      name: String(t.name),
      description: String(t.description),
      task_type: t.task_type,
      priority: t.priority,
      status: t.status,
      assigned_agent_id: t.assigned_agent_id ?? undefined,
      assigned_agent_name: t.assigned_agent_name ?? undefined,
      supervisor_id: String(t.supervisor_id ?? supervisorId),
      input_data: t.input_data ?? undefined,
      output_data: t.output_data ?? undefined,
      execution_metadata: t.execution_metadata ?? undefined,
      complexity_score: t.complexity_score ?? undefined,
      estimated_time_seconds: t.estimated_time_seconds ?? undefined,
      actual_time_seconds: t.actual_time_seconds ?? undefined,
      created_at: String(t.created_at),
      updated_at: String(t.updated_at),
      started_at: t.started_at ?? undefined,
      completed_at: t.completed_at ?? undefined,
    }))

    const total = Number(resp.data?.pagination?.total ?? 0)
    const totalPages = Math.ceil(total / pageSize) || 0
    
    return {
      tasks,
      total,
      page,
      pageSize,
      totalPages
    }
  }

  /**
   * 获取特定任务详情
   */
  async getTask(taskId: string): Promise<SupervisorTask> {
    const resp = await this.request<any>(`${API_BASE}/tasks/${encodeURIComponent(taskId)}/details`)
    return resp.data as SupervisorTask
  }

  /**
   * 获取决策历史
   */
  async getDecisionHistory(
    supervisorId: string, 
    page = 1, 
    pageSize = 20
  ): Promise<{
    decisions: SupervisorDecision[]
    total: number
    page: number
    pageSize: number
    totalPages: number
  }> {
    const offset = Math.max(0, (page - 1) * pageSize)
    const resp = await this.request<SupervisorDecision[]>(
      `${API_BASE}/decisions?supervisor_id=${encodeURIComponent(supervisorId)}&limit=${pageSize}&offset=${offset}`,
    )

    const decisions = (resp.data || []).map((d: any) => ({ ...d, supervisor_id: supervisorId })) as SupervisorDecision[]
    const total = Number(resp.pagination?.total ?? decisions.length)
    const totalPages = Math.ceil(total / pageSize) || 0
    
    return {
      decisions,
      total,
      page,
      pageSize,
      totalPages
    }
  }

  /**
   * 获取智能体负载指标
   */
  async getAgentMetrics(supervisorId: string): Promise<AgentLoadMetrics[]> {
    const resp = await this.request<AgentLoadMetrics[]>(
      `${API_BASE}/metrics?supervisor_id=${encodeURIComponent(supervisorId)}`,
    )
    return resp.data || []
  }

  /**
   * 获取Supervisor统计数据
   */
  async getStats(supervisorId: string): Promise<SupervisorStats> {
    const resp = await this.request<any>(
      `${API_BASE}/stats?supervisor_id=${encodeURIComponent(supervisorId)}`,
    )

    const statusDist = resp.data?.task_statistics?.status_distribution || {}
    const totalTasks = Object.values(statusDist).reduce((sum: number, v: any) => sum + Number(v || 0), 0)
    const completedTasks = Number(statusDist.completed || 0)
    const failedTasks = Number(statusDist.failed || 0)
    const runningTasks = Number(statusDist.running || 0)
    const pendingTasks = Number(statusDist.pending || 0)
    const successDenom = completedTasks + failedTasks

    return {
      total_tasks: totalTasks,
      completed_tasks: completedTasks,
      failed_tasks: failedTasks,
      pending_tasks: pendingTasks,
      running_tasks: runningTasks,
      average_completion_time: Number(resp.data?.task_statistics?.average_completion_time_seconds || 0),
      success_rate: successDenom > 0 ? completedTasks / successDenom : 0,
      agent_utilization: resp.data?.agent_loads || {},
      decision_accuracy: Number(resp.data?.decision_statistics?.success_rate || 0),
      recent_decisions: [],
    }
  }
  /**
   * 获取Supervisor配置
   */
  async getConfig(supervisorId: string): Promise<SupervisorConfig> {
    const resp = await this.request<SupervisorConfig>(
      `${API_BASE}/config?supervisor_id=${encodeURIComponent(supervisorId)}`,
    )
    return resp.data
  }

  /**
   * 更新Supervisor配置
   */
  async updateConfig(supervisorId: string, config: Partial<SupervisorConfig>): Promise<SupervisorConfig> {
    const resp = await this.request<SupervisorConfig>(
      `${API_BASE}/config?supervisor_id=${encodeURIComponent(supervisorId)}`,
      {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(config),
      },
    )
    return resp.data
  }

  /**
   * 健康检查
   */
  async healthCheck(): Promise<{ status: string; timestamp: string; version: string }> {
    const resp = await this.request<{ status: string; timestamp: string; version: string }>(
      `${API_BASE}/health`,
    )
    return resp.data
  }
}

// 导出单例实例
export const supervisorApiService = new SupervisorApiService()
export default supervisorApiService
