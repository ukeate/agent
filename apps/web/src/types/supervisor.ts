/**
 * Supervisor系统相关类型定义
 */

// 任务类型枚举
export type TaskType =
  | 'code_generation'
  | 'code_review'
  | 'documentation'
  | 'analysis'
  | 'planning'

// 任务优先级
export type TaskPriority = 'low' | 'medium' | 'high' | 'urgent'

// 任务状态
export type TaskStatus =
  | 'pending'
  | 'running'
  | 'completed'
  | 'failed'
  | 'cancelled'

// 智能体状态
export type AgentRole =
  | 'code_expert'
  | 'architect'
  | 'doc_expert'
  | 'supervisor'
  | 'rag_specialist'

export type SupervisorAgentStatus = 'active' | 'idle' | 'busy' | 'offline'

// 路由策略
export type RoutingStrategy =
  | 'round_robin'
  | 'capability_based'
  | 'load_balanced'
  | 'hybrid'

// Supervisor任务接口
export interface SupervisorTask {
  id: string
  name: string
  description: string
  task_type: TaskType
  priority: TaskPriority
  status: TaskStatus
  assigned_agent_id?: string
  assigned_agent_name?: string
  supervisor_id: string
  input_data?: Record<string, any>
  output_data?: Record<string, any>
  execution_metadata?: Record<string, any>
  complexity_score?: number
  estimated_time_seconds?: number
  actual_time_seconds?: number
  created_at: string
  updated_at: string
  started_at?: string
  completed_at?: string
}

// Supervisor智能体接口
export interface SupervisorAgent {
  id: string
  name: string
  role: string
  status: SupervisorAgentStatus
  capabilities: string[]
  configuration: {
    model: string
    temperature: number
    max_tokens: number
    tools: string[]
    system_prompt: string
  }
  performance_metrics?: Record<string, any>
  created_at: string
  updated_at: string
}

// 决策记录接口
export interface SupervisorDecision {
  id: string
  decision_id: string
  supervisor_id: string
  task_id: string
  task_description: string
  assigned_agent: string
  assignment_reason: string
  confidence_level: number
  match_score: number
  routing_strategy?: string
  alternative_agents?: string[]
  alternatives_considered?: Array<{
    agent: string
    score: number
    reason: string
  }>
  decision_metadata?: Record<string, any>
  routing_metadata?: Record<string, any>
  task_success?: boolean
  task_completion_time?: number
  quality_score?: number
  timestamp: string
  estimated_completion_time?: string
  actual_completion_time?: string
}

// 智能体负载指标
export interface AgentLoadMetrics {
  id: string
  agent_name: string
  supervisor_id: string
  current_load: number
  task_count: number
  average_task_time?: number
  success_rate?: number
  response_time_avg?: number
  error_rate: number
  availability_score: number
  window_start: string
  window_end: string
  created_at: string
  updated_at: string
}

// Supervisor状态响应
export interface SupervisorStatusResponse {
  supervisor_name: string
  status: SupervisorAgentStatus
  available_agents: string[]
  agent_loads: Record<string, number>
  decision_history_count: number
  task_queue_length: number
  performance_metrics?: Record<string, any>
  configuration?: Record<string, any>
}

// 任务提交请求
export interface TaskSubmissionRequest {
  name: string
  description: string
  task_type: TaskType
  priority: TaskPriority
  input_data?: Record<string, any>
  constraints?: Record<string, any>
  timeout_minutes?: number
}

// 任务分配响应
export interface TaskAssignmentResponse {
  task_id: string
  assigned_agent: string
  assignment_reason: string
  confidence_level: number
  estimated_completion_time?: string
  alternatives_considered?: Array<{
    agent: string
    score: number
    reason: string
  }>
}

// Supervisor配置
export interface SupervisorConfig {
  id: string
  supervisor_id: string
  config_name: string
  config_version: string
  routing_strategy: RoutingStrategy
  load_threshold: number
  capability_weight: number
  load_weight: number
  availability_weight: number
  enable_quality_assessment: boolean
  min_confidence_threshold: number
  enable_learning: boolean
  learning_rate: number
  optimization_interval_hours: number
  max_concurrent_tasks: number
  task_timeout_minutes: number
  enable_fallback: boolean
  config_metadata?: Record<string, any>
  is_active: boolean
  created_at: string
  updated_at: string
}

// API响应包装类型
export interface SupervisorApiResponse<T> {
  success: boolean
  message: string
  data: T
  timestamp: string
}

// Dashboard统计数据
export interface SupervisorStats {
  total_tasks: number
  completed_tasks: number
  failed_tasks: number
  pending_tasks: number
  running_tasks: number
  average_completion_time: number
  success_rate: number
  agent_utilization: Record<string, number>
  decision_accuracy: number
  recent_decisions: SupervisorDecision[]
}
