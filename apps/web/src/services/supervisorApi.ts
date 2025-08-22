/**
 * Supervisor API服务
 * 处理与后端Supervisor系统的通信
 */

import {
  SupervisorStatusResponse,
  SupervisorTask,
  SupervisorDecision,
  TaskSubmissionRequest,
  TaskAssignmentResponse,
  SupervisorConfig,
  SupervisorStats,
  SupervisorApiResponse,
  AgentLoadMetrics
} from '../types/supervisor'

const API_BASE = '/api/v1/supervisor'

class SupervisorApiService {
  /**
   * 获取Supervisor状态
   */
  async getStatus(supervisorId: string): Promise<SupervisorStatusResponse> {
    const response = await fetch(`${API_BASE}/status?supervisor_id=${supervisorId}`)
    
    if (!response.ok) {
      throw new Error(`获取Supervisor状态失败: ${response.statusText}`)
    }
    
    const data: SupervisorApiResponse<SupervisorStatusResponse> = await response.json()
    
    if (!data.success) {
      throw new Error(data.message || '获取状态失败')
    }
    
    return data.data
  }

  /**
   * 提交任务给Supervisor分配
   */
  async submitTask(
    supervisorId: string, 
    taskRequest: TaskSubmissionRequest
  ): Promise<TaskAssignmentResponse> {
    const response = await fetch(`${API_BASE}/tasks?supervisor_id=${supervisorId}`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(taskRequest),
    })
    
    if (!response.ok) {
      throw new Error(`任务提交失败: ${response.statusText}`)
    }
    
    const data: SupervisorApiResponse<TaskAssignmentResponse> = await response.json()
    
    if (!data.success) {
      throw new Error(data.message || '任务提交失败')
    }
    
    return data.data
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
    // 将页码转换为offset
    const offset = (page - 1) * pageSize
    const response = await fetch(
      `${API_BASE}/tasks?supervisor_id=${supervisorId}&limit=${pageSize}&offset=${offset}`
    )
    
    if (!response.ok) {
      throw new Error(`获取任务列表失败: ${response.statusText}`)
    }
    
    const data = await response.json()
    
    if (!data.success) {
      throw new Error(data.message || '获取任务列表失败')
    }
    
    // 转换后端返回的数据格式
    const tasks = data.data.tasks || []
    const total = data.data.pagination?.total || tasks.length
    const totalPages = Math.ceil(total / pageSize)
    
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
    const response = await fetch(`${API_BASE}/tasks/${taskId}/details`)
    
    if (!response.ok) {
      throw new Error(`获取任务详情失败: ${response.statusText}`)
    }
    
    const data: SupervisorApiResponse<SupervisorTask> = await response.json()
    
    if (!data.success) {
      throw new Error(data.message || '获取任务详情失败')
    }
    
    return data.data
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
    // 将页码转换为offset
    const offset = (page - 1) * pageSize
    const response = await fetch(
      `${API_BASE}/decisions?supervisor_id=${supervisorId}&limit=${pageSize}&offset=${offset}`
    )
    
    if (!response.ok) {
      throw new Error(`获取决策历史失败: ${response.statusText}`)
    }
    
    const data = await response.json()
    
    if (!data.success) {
      throw new Error(data.message || '获取决策历史失败')
    }
    
    // 转换后端返回的数据格式
    const decisions = data.data || []
    const total = data.pagination?.total || decisions.length
    const totalPages = Math.ceil(total / pageSize)
    
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
    const response = await fetch(`${API_BASE}/metrics?supervisor_id=${supervisorId}`)
    
    if (!response.ok) {
      throw new Error(`获取智能体指标失败: ${response.statusText}`)
    }
    
    const data = await response.json()
    
    if (!data.success) {
      throw new Error(data.message || '获取智能体指标失败')
    }
    
    // 后端返回的是LoadStatistics对象，需要转换为AgentLoadMetrics数组
    const loadStats = data.data
    const agentMetrics: AgentLoadMetrics[] = []
    
    if (loadStats.agent_loads) {
      Object.entries(loadStats.agent_loads).forEach(([agentName, load], index) => {
        agentMetrics.push({
          id: `agent_${index}`,
          agent_name: agentName,
          supervisor_id: supervisorId,
          current_load: typeof load === 'number' ? load : 0,
          task_count: loadStats.task_counts?.running || 0,
          average_task_time: 120, // 模拟数据
          success_rate: 0.85, // 模拟数据
          response_time_avg: 2.5, // 模拟数据
          error_rate: 0.05, // 模拟数据
          availability_score: typeof load === 'number' ? Math.max(0, 1 - load) : 0.5,
          window_start: new Date(Date.now() - 24 * 60 * 60 * 1000).toISOString(),
          window_end: new Date().toISOString(),
          created_at: new Date().toISOString(),
          updated_at: loadStats.last_update || new Date().toISOString()
        })
      })
    }
    
    return agentMetrics
  }

  /**
   * 获取Supervisor统计数据
   */
  async getStats(supervisorId: string): Promise<SupervisorStats> {
    const response = await fetch(`${API_BASE}/stats?supervisor_id=${supervisorId}`)
    
    if (!response.ok) {
      throw new Error(`获取统计数据失败: ${response.statusText}`)
    }
    
    const data: SupervisorApiResponse<SupervisorStats> = await response.json()
    
    if (!data.success) {
      throw new Error(data.message || '获取统计数据失败')
    }
    
    return data.data
  }

  /**
   * 获取Supervisor配置
   */
  async getConfig(supervisorId: string): Promise<SupervisorConfig> {
    const response = await fetch(`${API_BASE}/config?supervisor_id=${supervisorId}`)
    
    if (!response.ok) {
      throw new Error(`获取配置失败: ${response.statusText}`)
    }
    
    const data: SupervisorApiResponse<SupervisorConfig> = await response.json()
    
    if (!data.success) {
      throw new Error(data.message || '获取配置失败')
    }
    
    return data.data
  }

  /**
   * 更新Supervisor配置
   */
  async updateConfig(supervisorId: string, config: Partial<SupervisorConfig>): Promise<SupervisorConfig> {
    const response = await fetch(`${API_BASE}/config?supervisor_id=${supervisorId}`, {
      method: 'PUT',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(config),
    })
    
    if (!response.ok) {
      throw new Error(`更新配置失败: ${response.statusText}`)
    }
    
    const data: SupervisorApiResponse<SupervisorConfig> = await response.json()
    
    if (!data.success) {
      throw new Error(data.message || '更新配置失败')
    }
    
    return data.data
  }

  /**
   * 健康检查
   */
  async healthCheck(): Promise<{ status: string; timestamp: string; version: string }> {
    const response = await fetch(`${API_BASE}/health`)
    
    if (!response.ok) {
      throw new Error(`健康检查失败: ${response.statusText}`)
    }
    
    const data = await response.json()
    
    if (!data.success) {
      throw new Error(data.message || '健康检查失败')
    }
    
    return data.data
  }
}

// 导出单例实例
export const supervisorApiService = new SupervisorApiService()
export default supervisorApiService