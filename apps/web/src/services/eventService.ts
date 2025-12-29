/**
 * 事件服务
 * 处理与事件API的交互
 */

import { buildWsUrl } from '../utils/apiBase'
import apiClient from './apiClient'

import { logger } from '../utils/logger'
export interface Event {
  id: string
  timestamp: string
  type: 'info' | 'warning' | 'error' | 'success'
  source: string
  target?: string
  title: string
  message: string
  agent?: string
  severity: 'low' | 'medium' | 'high' | 'critical'
  data?: Record<string, any>
}

export interface EventStats {
  total: number
  info: number
  warning: number
  error: number
  success: number
  critical: number
  by_source: Record<string, number>
  by_type: Record<string, number>
}

export interface EventQuery {
  start_time?: string
  end_time?: string
  event_types?: string[]
  source?: string
  target?: string
  severity?: string
  limit?: number
  offset?: number
}

export interface ClusterStatus {
  node_id: string
  role: string
  status: string
  load: number
  active_nodes: number
  nodes: Record<string, any>
  stats: Record<string, any>
}

class EventService {
  private wsConnection: WebSocket | null = null
  private eventHandlers: Set<(event: Event) => void> = new Set()
  private reconnectTimer: ReturnType<typeof setTimeout> | null = null
  private heartbeatInterval: ReturnType<typeof setInterval> | null = null
  private reconnectAttempts = 0
  private readonly maxReconnectAttempts = 10

  /**
   * 获取事件列表
   */
  async getEvents(query: EventQuery = {}): Promise<Event[]> {
    try {
      const params = new URLSearchParams()
      
      if (query.start_time) params.append('start_time', query.start_time)
      if (query.end_time) params.append('end_time', query.end_time)
      if (query.source) params.append('source', query.source)
      if (query.target) params.append('target', query.target)
      if (query.severity) params.append('severity', query.severity)
      if (query.limit) params.append('limit', query.limit.toString())
      if (query.offset) params.append('offset', query.offset.toString())
      
      if (query.event_types?.length) {
        query.event_types.forEach(type => params.append('event_types', type))
      }

      const response = await apiClient.get(`/events/list?${params.toString()}`)
      return response.data
    } catch (error) {
      logger.error('获取事件列表失败:', error)
      throw error
    }
  }

  /**
   * 获取事件统计
   */
  async getEventStats(hours: number = 24): Promise<EventStats> {
    try {
      const response = await apiClient.get(`/events/stats?hours=${hours}`)
      return response.data
    } catch (error) {
      logger.error('获取事件统计失败:', error)
      throw error
    }
  }

  /**
   * 获取集群状态
   */
  async getClusterStatus(): Promise<ClusterStatus> {
    try {
      const response = await apiClient.get('/events/cluster/status')
      return response.data
    } catch (error) {
      logger.error('获取集群状态失败:', error)
      throw error
    }
  }

  /**
   * 获取监控指标
   */
  async getMonitoringMetrics(): Promise<any> {
    try {
      const response = await apiClient.get('/events/monitoring/metrics')
      return response.data
    } catch (error) {
      logger.error('获取监控指标失败:', error)
      throw error
    }
  }

  /**
   * 重播事件
   */
  async replayEvents(params: {
    agent_id?: string
    conversation_id?: string
    start_time: string
    end_time?: string
  }): Promise<any> {
    try {
      const response = await apiClient.post('/events/replay', params)
      return response.data
    } catch (error) {
      logger.error('重播事件失败:', error)
      throw error
    }
  }

  /**
   * 手动提交事件（测试用）
   */
  async submitEvent(params: {
    event_type: string
    source: string
    message?: string
    priority?: string
  }): Promise<{ status: string; event_id: string }> {
    try {
      const response = await apiClient.post('/events/submit', null, { params })
      return response.data
    } catch (error) {
      logger.error('提交事件失败:', error)
      throw error
    }
  }

  /**
   * 连接WebSocket事件流
   */
  connectEventStream(onEvent: (event: Event) => void): void {
    // 添加事件处理器
    this.eventHandlers.add(onEvent)
    this.connectIfNeeded()
  }

  /**
   * 断开WebSocket连接
   */
  disconnectEventStream(): void {
    this.closeConnection()
    this.eventHandlers.clear()
  }

  /**
   * 移除事件处理器
   */
  removeEventHandler(handler: (event: Event) => void): void {
    this.eventHandlers.delete(handler)
    
    // 如果没有处理器了，断开连接
    if (this.eventHandlers.size === 0) {
      this.disconnectEventStream()
    }
  }

  private connectIfNeeded(): void {
    if (this.wsConnection && this.wsConnection.readyState !== WebSocket.CLOSED) {
      return
    }
    this.clearReconnectTimer()
    this.openConnection()
  }

  private openConnection(): void {
    const wsUrl = buildWsUrl('/events/stream')

    try {
      this.wsConnection = new WebSocket(wsUrl)

      this.wsConnection.onopen = () => {
        logger.log('事件流WebSocket连接已建立')
        this.reconnectAttempts = 0
        this.startHeartbeat()
      }

      this.wsConnection.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data)

          if (data.type === 'event' && data.data) {
            this.eventHandlers.forEach(handler => {
              handler(data.data as Event)
            })
          }
        } catch (error) {
          logger.error('解析事件消息失败:', error)
        }
      }

      this.wsConnection.onerror = (error) => {
        logger.error('WebSocket错误:', error)
        if (
          this.wsConnection &&
          this.wsConnection.readyState !== WebSocket.CLOSING &&
          this.wsConnection.readyState !== WebSocket.CLOSED
        ) {
          this.wsConnection.close()
        }
      }

      this.wsConnection.onclose = () => {
        logger.log('事件流WebSocket连接已关闭')
        this.wsConnection = null
        this.stopHeartbeat()
        this.scheduleReconnect()
      }
    } catch (error) {
      logger.error('建立WebSocket连接失败:', error)
      this.scheduleReconnect()
    }
  }

  private scheduleReconnect(): void {
    if (this.eventHandlers.size === 0) return
    this.clearReconnectTimer()
    if (this.reconnectAttempts >= this.maxReconnectAttempts) {
      logger.error('事件流重连次数已达上限')
      return
    }
    this.reconnectAttempts += 1
    const delay = Math.min(5000 * Math.pow(2, this.reconnectAttempts - 1), 30000)
    this.reconnectTimer = setTimeout(() => {
      if (this.eventHandlers.size > 0) {
        logger.log('尝试重新连接事件流...')
        this.openConnection()
      }
    }, delay)
  }

  private clearReconnectTimer(): void {
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer)
      this.reconnectTimer = null
    }
  }

  private startHeartbeat(): void {
    this.stopHeartbeat()
    this.heartbeatInterval = setInterval(() => {
      if (this.wsConnection?.readyState === WebSocket.OPEN) {
        this.wsConnection.send('ping')
      } else {
        this.stopHeartbeat()
      }
    }, 30000)
  }

  private stopHeartbeat(): void {
    if (this.heartbeatInterval) {
      clearInterval(this.heartbeatInterval)
      this.heartbeatInterval = null
    }
  }

  private closeConnection(): void {
    this.clearReconnectTimer()
    this.stopHeartbeat()
    this.reconnectAttempts = 0
    if (this.wsConnection) {
      this.wsConnection.close()
      this.wsConnection = null
    }
  }
}

export default new EventService()
