import apiClient from './apiClient'

import { logger } from '../utils/logger'
// 类型定义
export interface DecompositionRequest {
  problem_statement: string
  context?: string
  strategy?: string
  max_depth?: number
  target_complexity?: number
  enable_branching?: boolean
  time_limit_minutes?: number
}

export interface ExecutionRequest {
  workflow_definition_id: string
  execution_mode?: string
  max_parallel_steps?: number
  scheduling_strategy?: string
  input_data?: Record<string, any>
}

export interface ExecutionControlRequest {
  execution_id: string
  action: 'pause' | 'resume' | 'cancel'
}

export interface TaskDAG {
  id: string
  name: string
  description: string
  nodes: TaskNode[]
  edges: TaskEdge[]
  parallel_groups: string[][]
  critical_path: string[]
  is_acyclic: boolean
  total_nodes: number
  max_depth: number
}

export interface TaskNode {
  id: string
  name: string
  description: string
  task_type: string
  dependencies: string[]
  complexity_score: number
  estimated_duration_minutes: number
  priority: number
}

export interface TaskEdge {
  from: string
  to: string
}

export interface WorkflowDefinition {
  id: string
  name: string
  description: string
  steps: WorkflowStep[]
  execution_mode: string
  max_parallel_steps: number
  metadata: Record<string, any>
}

export interface WorkflowStep {
  id: string
  name: string
  step_type: string
  description: string
  dependencies: string[]
  config: Record<string, any>
  timeout_seconds: number
}

export interface ExecutionResponse {
  execution_id: string
  status: string
  workflow_definition_id: string
  progress: number
  current_step?: string
  start_time: string
  estimated_completion?: string
}

export interface SystemMetrics {
  active_workers: number
  queue_depth: number
  average_wait_time: number
  success_rate: number
  throughput: number
  resource_utilization: Record<string, number>
}

export interface DecompositionResponse {
  task_dag: TaskDAG
  workflow_definition: WorkflowDefinition
  decomposition_metadata: {
    strategy_used: string
    complexity_achieved: number
    total_estimated_time: number
    parallelization_factor: number
    critical_path_length: number
  }
}

export interface ExecutionResults {
  execution_id: string
  results: {
    summary: string
    validation_score: number
    aggregated_result: Record<string, unknown>
    step_results: Record<string, unknown>
  }
  performance_metrics: {
    total_duration: number
    average_step_duration: number
    parallelization_efficiency: number
    resource_utilization: number
  }
}

// 多步推理工作流API服务
export class MultiStepReasoningApi {
  private baseURL = '/multi-step-reasoning'

  // 问题分解
  async decomposeProblem(
    request: DecompositionRequest
  ): Promise<DecompositionResponse> {
    try {
      const response = await apiClient.post(
        `${this.baseURL}/decompose`,
        request
      )
      return response.data
    } catch (error) {
      logger.error('问题分解失败:', error)
      throw error
    }
  }

  // 启动执行
  async startExecution(request: ExecutionRequest): Promise<ExecutionResponse> {
    try {
      const response = await apiClient.post(`${this.baseURL}/execute`, request)
      return response.data
    } catch (error) {
      logger.error('启动执行失败:', error)
      throw error
    }
  }

  // 获取执行状态
  async getExecutionStatus(executionId: string): Promise<ExecutionResponse> {
    try {
      const response = await apiClient.get(
        `${this.baseURL}/executions/${executionId}`
      )
      return response.data
    } catch (error) {
      logger.error('获取执行状态失败:', error)
      throw error
    }
  }

  // 控制执行
  async controlExecution(
    request: ExecutionControlRequest
  ): Promise<{ message: string; status: string }> {
    try {
      const response = await apiClient.post(
        `${this.baseURL}/executions/control`,
        request
      )
      return response.data
    } catch (error) {
      logger.error('执行控制失败:', error)
      throw error
    }
  }

  // 获取系统指标
  async getSystemMetrics(): Promise<SystemMetrics> {
    try {
      const response = await apiClient.get(`${this.baseURL}/system/metrics`)
      return response.data
    } catch (error) {
      logger.error('获取系统指标失败:', error)
      throw error
    }
  }

  // 获取工作流列表
  async listWorkflows(): Promise<{
    workflows: Array<{
      id: string
      name: string
      description: string
      step_count: number
      execution_mode: string
      created_at: string
    }>
  }> {
    try {
      const response = await apiClient.get(`${this.baseURL}/workflows`)
      return response.data
    } catch (error) {
      logger.error('获取工作流列表失败:', error)
      throw error
    }
  }

  // 获取执行列表
  async listExecutions(): Promise<{
    executions: Array<{
      id: string
      workflow_definition_id: string
      status: string
      progress: number
      started_at: string
      total_steps: number
      completed_steps: number
    }>
  }> {
    try {
      const response = await apiClient.get(`${this.baseURL}/executions`)
      return response.data
    } catch (error) {
      logger.error('获取执行列表失败:', error)
      throw error
    }
  }

  // 删除执行
  async deleteExecution(executionId: string): Promise<{ message: string }> {
    try {
      const response = await apiClient.delete(
        `${this.baseURL}/executions/${executionId}`
      )
      return response.data
    } catch (error) {
      logger.error('删除执行失败:', error)
      throw error
    }
  }

  // 获取执行结果
  async getExecutionResults(executionId: string): Promise<ExecutionResults> {
    try {
      const response = await apiClient.get(
        `${this.baseURL}/executions/${executionId}/results`
      )
      return response.data
    } catch (error) {
      logger.error('获取执行结果失败:', error)
      throw error
    }
  }

  // 轮询执行状态
  pollExecutionStatus(
    executionId: string,
    onUpdate: (status: ExecutionResponse) => void,
    interval: number = 2000
  ): () => void {
    let isPolling = true
    let timeoutId: ReturnType<typeof setTimeout> | null = null

    const poll = async () => {
      if (!isPolling) return
      try {
        const status = await this.getExecutionStatus(executionId)
        if (!isPolling) return
        onUpdate(status)

        const normalizedStatus = status.status.toLowerCase()
        const isTerminal =
          normalizedStatus === 'completed' ||
          normalizedStatus === 'failed' ||
          normalizedStatus === 'cancelled' ||
          normalizedStatus === 'canceled'
        if (isPolling && !isTerminal) {
          timeoutId = setTimeout(() => {
            void poll()
          }, interval)
        }
      } catch (error) {
        if (!isPolling) return
        logger.error('轮询失败:', error)
        if (isPolling) {
          timeoutId = setTimeout(() => {
            void poll()
          }, interval * 2)
        }
      }
    }

    void poll()

    return () => {
      isPolling = false
      if (timeoutId) {
        clearTimeout(timeoutId)
      }
    }
  }

  // 监控系统指标
  monitorSystemMetrics(
    onUpdate: (metrics: SystemMetrics) => void,
    interval: number = 5000
  ): () => void {
    let isMonitoring = true

    const monitor = async () => {
      if (!isMonitoring) return

      try {
        const metrics = await this.getSystemMetrics()
        if (!isMonitoring) return
        onUpdate(metrics)
      } catch (error) {
        logger.error('系统指标监控失败:', error)
      }

      if (isMonitoring) {
        setTimeout(monitor, interval)
      }
    }

    monitor()

    // 返回停止监控的函数
    return () => {
      isMonitoring = false
    }
  }
}

// 导出单例实例
export const multiStepReasoningApi = new MultiStepReasoningApi()
