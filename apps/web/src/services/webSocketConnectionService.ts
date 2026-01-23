/**
 * WebSocket 连接管理服务
 *
 * 提供 WebSocket 连接状态监控和管理功能
 */

import apiClient from './apiClient'
import { buildApiUrl } from '../utils/apiBase'

import { logger } from '../utils/logger'
// WebSocket 连接详情接口
export interface WebSocketConnection {
  index: number
  state: 'connected' | 'disconnected' | 'connecting' | 'error'
  connected_at: string
  client_info?: {
    user_agent?: string
    ip_address?: string
    session_id?: string
  }
  last_activity?: string
  messages_sent?: number
  messages_received?: number
}

// WebSocket 连接状态响应接口
export interface WebSocketConnectionsResponse {
  active_connections: number
  connection_details: WebSocketConnection[]
  total_connections_today?: number
  peak_connections_today?: number
  server_stats?: {
    uptime_seconds: number
    memory_usage_mb: number
    cpu_usage_percent: number
  }
}

// WebSocket 连接统计接口
export interface WebSocketConnectionStats {
  current_connections: number
  total_connections: number
  peak_connections: number
  connection_rate: number
  disconnection_rate: number
  average_duration: number
  success_rate: number
}

class WebSocketConnectionService {
  private baseUrl = '/ws'

  /**
   * 获取当前 WebSocket 连接状态
   */
  async getConnections(): Promise<WebSocketConnectionsResponse> {
    const response = await apiClient.get(`${this.baseUrl}/connections`)
    return response.data
  }

  /**
   * 获取 WebSocket 连接统计
   */
  async getConnectionStats(): Promise<WebSocketConnectionStats> {
    const response = await apiClient.get(`${this.baseUrl}/connections/stats`)
    return response.data
  }

  /**
   * 强制断开指定连接
   */
  async disconnectConnection(connectionIndex: number): Promise<{
    success: boolean
    message: string
  }> {
    try {
      const response = await apiClient.delete(
        `${this.baseUrl}/connections/${connectionIndex}`
      )
      return response.data
    } catch (error) {
      logger.error('断开连接失败:', error)
      throw error
    }
  }

  /**
   * 断开所有连接
   */
  async disconnectAllConnections(): Promise<{
    success: boolean
    disconnected_count: number
    message: string
  }> {
    try {
      const response = await apiClient.delete(`${this.baseUrl}/connections/all`)
      return response.data
    } catch (error) {
      logger.error('断开所有连接失败:', error)
      throw error
    }
  }

  /**
   * 向指定连接发送消息
   */
  async sendMessageToConnection(
    connectionIndex: number,
    message: any
  ): Promise<{
    success: boolean
    message: string
  }> {
    try {
      const response = await apiClient.post(
        `${this.baseUrl}/connections/${connectionIndex}/send`,
        {
          message,
        }
      )
      return response.data
    } catch (error) {
      logger.error('发送消息失败:', error)
      throw error
    }
  }

  /**
   * 广播消息到所有连接
   */
  async broadcastMessage(message: any): Promise<{
    success: boolean
    sent_count: number
    message: string
  }> {
    try {
      const response = await apiClient.post(
        `${this.baseUrl}/connections/broadcast`,
        {
          message,
        }
      )
      return response.data
    } catch (error) {
      logger.error('广播消息失败:', error)
      throw error
    }
  }

  /**
   * 获取连接健康状态
   */
  async getConnectionHealth(): Promise<{
    status: 'healthy' | 'degraded' | 'unhealthy'
    active_connections: number
    failed_connections: number
    average_latency_ms: number
    last_check: string
  }> {
    const response = await apiClient.get(`${this.baseUrl}/connections/health`)
    return response.data
  }

  /**
   * 获取连接日志
   */
  async getConnectionLogs(limit: number = 100): Promise<{
    logs: Array<{
      timestamp: string
      level: 'info' | 'warn' | 'error'
      message: string
      connection_index?: number
      details?: any
    }>
    total_count: number
  }> {
    const response = await apiClient.get(`${this.baseUrl}/connections/logs`, {
      params: { limit },
    })
    return response.data
  }

  /**
   * 启动连接监控
   */
  startConnectionMonitoring(
    onUpdate: (connections: WebSocketConnectionsResponse) => void,
    intervalMs: number = 5000
  ): () => void {
    const interval = setInterval(async () => {
      try {
        const connections = await this.getConnections()
        onUpdate(connections)
      } catch (error) {
        logger.error('连接监控更新失败:', error)
      }
    }, intervalMs)

    // 立即执行一次
    this.getConnections()
      .then(onUpdate)
      .catch(error => logger.error('获取连接状态失败:', error))

    // 返回停止监控的函数
    return () => clearInterval(interval)
  }

  /**
   * 实时连接事件流
   */
  subscribeToConnectionEvents(
    onEvent: (event: {
      type: 'connect' | 'disconnect' | 'message' | 'error'
      connection_index: number
      connection_id?: string
      timestamp: string
      data?: any
    }) => void
  ): () => void {
    // 使用 EventSource 或 WebSocket 连接实时事件流
    const eventSource = new EventSource(buildApiUrl('/ws/connections/events'))

    eventSource.onmessage = event => {
      try {
        const data = JSON.parse(event.data)
        onEvent(data)
      } catch (error) {
        logger.error('解析连接事件失败:', error)
      }
    }

    eventSource.onerror = error => {
      logger.error('连接事件流错误:', error)
      eventSource.close()
    }

    // 返回断开连接的函数
    return () => {
      eventSource.close()
    }
  }
}

// 导出服务实例
export const webSocketConnectionService = new WebSocketConnectionService()
export default webSocketConnectionService
