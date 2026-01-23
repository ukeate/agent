/**
 * 流式处理服务
 *
 * 提供流式处理、背压控制和队列监控的API接口
 */

import { buildApiUrl, buildWsUrl } from '../utils/apiBase'
import apiClient from './apiClient'

export interface StreamingMetrics {
  total_sessions: number
  active_sessions: number
  total_sessions_created: number
  total_tokens_processed: number
  total_events_processed: number
  total_buffer_usage: number
  active_streamers: number
  active_buffers: number
  uptime: number
}

export interface BackpressureStatus {
  throttle_level: string
  buffer_usage: number
  buffer_usage_ratio: number
  is_monitoring: boolean
  pressure_metrics: {
    [key: string]: {
      current_value: number
      threshold: number
      severity: number
      over_threshold: boolean
    }
  }
  active_throttles: Array<{
    level: string
    action_type: string
    target: string
    parameters: Record<string, any>
    applied_at: string
  }>
}

export interface FlowControlMetrics {
  backpressure_enabled: boolean
  max_concurrent_sessions: number
  current_sessions: number
  backpressure_status?: BackpressureStatus
  rate_limiter_stats?: {
    rate: number
    per: number
    burst: number
    current_allowance: number
    total_requests: number
    total_allowed: number
    total_rejected: number
    rejection_rate: number
  }
  circuit_breaker_state?: {
    state: string
    failure_count: number
    failure_threshold: number
    last_failure_time: string | null
    recovery_timeout: number
  }
  queue_metrics?: {
    name: string
    current_size: number
    max_size: number
    enqueue_rate: number
    dequeue_rate: number
    average_wait_time: number
    oldest_item_age: number
    timestamp: string
  }
}

export interface QueueMetrics {
  name: string
  current_size: number
  max_size: number
  utilization: number
  enqueue_rate: number
  dequeue_rate: number
  average_wait_time: number
  oldest_item_age: number
  is_overloaded: boolean
  throughput_ratio: number
}

export interface QueueStatus {
  queue_metrics: Record<string, QueueMetrics>
  system_summary: {
    total_queues: number
    overloaded_queues: number
    overloaded_queue_names: string[]
    total_items: number
    total_capacity: number
    system_utilization: number
    average_utilization: number
    is_running: boolean
  }
  overloaded_queues: string[]
}

export interface SessionMetrics {
  session_id: string
  agent_id: string
  status: string
  duration: number | null
  token_count: number
  event_count: number
  error_count: number
  tokens_per_second: number
  last_activity: number
  stream_metrics: Record<string, any>
  buffer_metrics: Record<string, any> | null
}

export interface StreamingSession {
  session_id: string
  status: string
  message: string
}

class StreamingService {
  private baseUrl = '/streaming'

  // 系统指标
  async getSystemMetrics(): Promise<{
    system_metrics: StreamingMetrics
    timestamp: string
  }> {
    const response = await apiClient.get(`${this.baseUrl}/metrics`)
    return response.data
  }

  // 背压状态
  async getBackpressureStatus(): Promise<{
    backpressure_enabled: false
    backpressure_status?: BackpressureStatus
    message?: string
    timestamp: string
  }> {
    const response = await apiClient.get(`${this.baseUrl}/backpressure/status`)
    return response.data
  }

  // 流量控制指标
  async getFlowControlMetrics(): Promise<{
    flow_control_metrics: FlowControlMetrics
    timestamp: string
  }> {
    const response = await apiClient.get(`${this.baseUrl}/flow-control/metrics`)
    return response.data
  }

  // 队列状态
  async getQueueStatus(): Promise<QueueStatus & { timestamp: string }> {
    const response = await apiClient.get(`${this.baseUrl}/queue/status`)
    return response.data
  }

  // 会话管理
  async getSessions(): Promise<{
    sessions: Record<string, SessionMetrics>
    total_sessions: number
    timestamp: string
  }> {
    const response = await apiClient.get(`${this.baseUrl}/sessions`)
    return response.data
  }

  async getSessionMetrics(sessionId: string): Promise<{
    session_metrics: SessionMetrics
    timestamp: string
  }> {
    const response = await apiClient.get(
      `${this.baseUrl}/sessions/${sessionId}/metrics`
    )
    return response.data
  }

  async createSession(data: {
    agent_id: string
    message: string
    session_id?: string
    buffer_size?: number
  }): Promise<StreamingSession> {
    const response = await apiClient.post(`${this.baseUrl}/start`, data)
    return response.data
  }

  async stopSession(sessionId: string): Promise<{
    session_id: string
    status: string
    message: string
  }> {
    const response = await apiClient.delete(
      `${this.baseUrl}/sessions/${sessionId}`
    )
    return response.data
  }

  async cleanupSession(sessionId: string): Promise<{
    session_id: string
    status: string
    message: string
  }> {
    const response = await apiClient.post(
      `${this.baseUrl}/sessions/${sessionId}/cleanup`
    )
    return response.data
  }

  // 健康检查
  async getHealthStatus(): Promise<{
    status: string
    service: string
    active_sessions: number
    total_sessions: number
    uptime: number
    timestamp: string
    error?: string
  }> {
    const response = await apiClient.get(`${this.baseUrl}/health`)
    return response.data
  }

  // SSE流式连接
  createSSEConnection(sessionId: string, message: string): EventSource {
    const url = new URL(
      buildApiUrl(`${this.baseUrl}/sse/${sessionId}`),
      window.location.origin
    )
    url.searchParams.set('message', message)
    return new EventSource(url.toString())
  }

  // WebSocket流式连接
  createWebSocketConnection(sessionId: string): WebSocket {
    return new WebSocket(buildWsUrl(`${this.baseUrl}/ws/${sessionId}`))
  }
}

export const streamingService = new StreamingService()
