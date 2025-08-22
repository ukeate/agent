/**
 * 事件服务
 * 处理与事件API的交互
 */

import apiClient from './apiClient'

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
      console.error('获取事件列表失败:', error)
      // 返回模拟数据作为fallback
      return this.getMockEvents()
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
      console.error('获取事件统计失败:', error)
      // 返回模拟数据
      return {
        total: 10,
        info: 4,
        warning: 3,
        error: 2,
        success: 1,
        critical: 0,
        by_source: { 'System': 5, 'Agent Manager': 3, 'Task Scheduler': 2 },
        by_type: { 'MESSAGE_SENT': 4, 'TASK_COMPLETED': 3, 'ERROR_OCCURRED': 3 }
      }
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
      console.error('获取集群状态失败:', error)
      return {
        node_id: 'local',
        role: 'standalone',
        status: 'active',
        load: 0.1,
        active_nodes: 1,
        nodes: { 'local': { status: 'active', role: 'leader', load: 0.1 } },
        stats: { events_processed: 0 }
      }
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
      console.error('获取监控指标失败:', error)
      return {
        event_counts: {},
        processing_times: {},
        error_rates: {},
        queue_sizes: {}
      }
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
      console.error('重播事件失败:', error)
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
      console.error('提交事件失败:', error)
      throw error
    }
  }

  /**
   * 连接WebSocket事件流
   */
  connectEventStream(onEvent: (event: Event) => void): void {
    // 断开现有连接
    this.disconnectEventStream()
    
    // 添加事件处理器
    this.eventHandlers.add(onEvent)
    
    // 建立WebSocket连接
    const wsUrl = `ws://localhost:8000/api/v1/events/stream`
    
    try {
      this.wsConnection = new WebSocket(wsUrl)
      
      this.wsConnection.onopen = () => {
        console.log('事件流WebSocket连接已建立')
      }
      
      this.wsConnection.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data)
          
          if (data.type === 'event' && data.data) {
            // 广播事件到所有处理器
            this.eventHandlers.forEach(handler => {
              handler(data.data as Event)
            })
          }
        } catch (error) {
          console.error('解析事件消息失败:', error)
        }
      }
      
      this.wsConnection.onerror = (error) => {
        console.error('WebSocket错误:', error)
      }
      
      this.wsConnection.onclose = () => {
        console.log('事件流WebSocket连接已关闭')
        this.wsConnection = null
        
        // 5秒后尝试重连
        setTimeout(() => {
          if (this.eventHandlers.size > 0) {
            console.log('尝试重新连接事件流...')
            const handlers = Array.from(this.eventHandlers)
            this.connectEventStream(handlers[0])
          }
        }, 5000)
      }
      
      // 定期发送心跳
      const heartbeatInterval = setInterval(() => {
        if (this.wsConnection?.readyState === WebSocket.OPEN) {
          this.wsConnection.send('ping')
        } else {
          clearInterval(heartbeatInterval)
        }
      }, 30000)
      
    } catch (error) {
      console.error('建立WebSocket连接失败:', error)
    }
  }

  /**
   * 断开WebSocket连接
   */
  disconnectEventStream(): void {
    if (this.wsConnection) {
      this.wsConnection.close()
      this.wsConnection = null
    }
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

  /**
   * 获取模拟事件数据
   */
  private getMockEvents(): Event[] {
    const now = new Date()
    return [
      {
        id: '1',
        timestamp: new Date(now.getTime() - 1000 * 60 * 5).toISOString(),
        type: 'error',
        source: 'Agent Manager',
        title: '智能体连接失败',
        message: 'RAG处理器智能体连接超时，正在尝试重连',
        agent: 'RAG处理器',
        severity: 'high'
      },
      {
        id: '2',
        timestamp: new Date(now.getTime() - 1000 * 60 * 10).toISOString(),
        type: 'success',
        source: 'Task Scheduler',
        title: '任务完成',
        message: '文档分析任务已成功完成，处理了125个文档',
        agent: '文档分析器',
        severity: 'low'
      },
      {
        id: '3',
        timestamp: new Date(now.getTime() - 1000 * 60 * 15).toISOString(),
        type: 'warning',
        source: 'Resource Monitor',
        title: 'CPU使用率过高',
        message: '代码生成器CPU使用率达到85%，建议优化性能',
        agent: '代码生成器',
        severity: 'medium'
      },
      {
        id: '4',
        timestamp: new Date(now.getTime() - 1000 * 60 * 20).toISOString(),
        type: 'info',
        source: 'System',
        title: '系统启动',
        message: '多智能体系统已成功启动，所有组件运行正常',
        severity: 'low'
      }
    ]
  }
}

export default new EventService()