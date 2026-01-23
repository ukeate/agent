import {
  apiFetch,
  apiFetchJson,
  buildApiUrl,
  buildWsUrl,
} from '../utils/apiBase'
import { logger } from '../utils/logger'
interface EventSubmissionRequest {
  events: BehaviorEvent[]
  batch_id?: string
}

interface BehaviorEvent {
  event_id: string
  user_id: string
  session_id?: string
  event_type: string
  timestamp: string
  properties?: Record<string, any>
  context?: Record<string, any>
  duration?: number
}

interface AnalysisRequest {
  user_id?: string
  session_id?: string
  start_time?: string
  end_time?: string
  event_types?: string[]
  analysis_types: string[]
}

interface PatternQuery {
  user_id?: string
  pattern_type?: string
  min_support?: number
  limit?: number
}

interface AnomalyQuery {
  user_id?: string
  severity?: string
  start_time?: string
  end_time?: string
  limit?: number
  use_real_detection?: boolean
}

interface ReportRequest {
  report_type: string
  format: string
  filters?: Record<string, any>
  include_visualizations?: boolean
}

class BehaviorAnalyticsService {
  private baseUrl = buildApiUrl('/analytics')
  private wsUrl = buildWsUrl('/analytics/ws')

  // 提交用户行为事件
  async submitEvents(request: EventSubmissionRequest): Promise<any> {
    return apiFetchJson(`${this.baseUrl}/events`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(request),
    })
  }

  // 查询行为事件
  async getEvents(
    params: {
      user_id?: string
      session_id?: string
      event_type?: string
      start_time?: string
      end_time?: string
      limit?: number
      offset?: number
    } = {}
  ): Promise<any> {
    const queryString = new URLSearchParams()
    Object.entries(params).forEach(([key, value]) => {
      if (value !== undefined) {
        queryString.append(key, String(value))
      }
    })

    return apiFetchJson(`${this.baseUrl}/events?${queryString}`)
  }

  // 查询用户会话
  async getSessions(
    params: {
      user_id?: string
      start_time?: string
      end_time?: string
      status?: string
      min_duration?: number
      max_duration?: number
      min_events?: number
      max_events?: number
      limit?: number
      offset?: number
    } = {}
  ): Promise<any> {
    const queryString = new URLSearchParams()
    Object.entries(params).forEach(([key, value]) => {
      if (value !== undefined) {
        queryString.append(key, String(value))
      }
    })

    return apiFetchJson(`${this.baseUrl}/sessions?${queryString}`)
  }

  // 获取会话统计
  async getSessionStats(
    params: { user_id?: string; start_time?: string; end_time?: string } = {}
  ): Promise<any> {
    const queryString = new URLSearchParams()
    Object.entries(params).forEach(([key, value]) => {
      if (value !== undefined) {
        queryString.append(key, String(value))
      }
    })
    const suffix = queryString.toString() ? `?${queryString}` : ''
    return apiFetchJson(`${this.baseUrl}/sessions/stats${suffix}`)
  }

  // 执行行为分析
  async analyzeBehavior(request: AnalysisRequest): Promise<any> {
    return apiFetchJson(`${this.baseUrl}/analyze`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(request),
    })
  }

  // 获取行为模式
  async getPatterns(params: PatternQuery = {}): Promise<any> {
    const queryString = new URLSearchParams()
    Object.entries(params).forEach(([key, value]) => {
      if (value !== undefined) {
        queryString.append(key, String(value))
      }
    })

    return apiFetchJson(`${this.baseUrl}/patterns?${queryString}`)
  }

  // 获取异常检测结果
  async getAnomalies(params: AnomalyQuery = {}): Promise<any> {
    const queryString = new URLSearchParams()
    Object.entries(params).forEach(([key, value]) => {
      if (value !== undefined) {
        queryString.append(key, String(value))
      }
    })

    return apiFetchJson(`${this.baseUrl}/anomalies?${queryString}`)
  }

  // 生成分析报告
  async generateReport(request: ReportRequest): Promise<any> {
    return apiFetchJson(`${this.baseUrl}/reports/generate`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(request),
    })
  }

  // 获取报告
  async getReport(reportId: string): Promise<any> {
    return apiFetchJson(`${this.baseUrl}/reports/${reportId}`)
  }

  // 获取报告列表
  async getReports(
    params: { limit?: number; offset?: number } = {}
  ): Promise<any> {
    const queryString = new URLSearchParams()
    Object.entries(params).forEach(([key, value]) => {
      if (value !== undefined) {
        queryString.append(key, String(value))
      }
    })
    const suffix = queryString.toString() ? `?${queryString}` : ''
    return apiFetchJson(`${this.baseUrl}/reports${suffix}`)
  }

  // 删除报告
  async deleteReport(reportId: string): Promise<any> {
    return apiFetchJson(`${this.baseUrl}/reports/${reportId}`, {
      method: 'DELETE',
    })
  }

  // 下载报告
  async downloadReport(
    reportId: string,
    format: string = 'json'
  ): Promise<Blob> {
    const response = await apiFetch(
      `${this.baseUrl}/reports/${reportId}/download?format=${format}`
    )
    return response.blob()
  }

  // 获取仪表板统计数据
  async getDashboardStats(
    timeRange: string = '24h',
    userId?: string
  ): Promise<any> {
    const queryString = new URLSearchParams({ time_range: timeRange })
    if (userId) {
      queryString.append('user_id', userId)
    }

    return apiFetchJson(`${this.baseUrl}/dashboard/stats?${queryString}`)
  }

  // 导出事件数据
  async exportEvents(
    format: string = 'json',
    params: {
      user_id?: string
      start_time?: string
      end_time?: string
      limit?: number
    } = {}
  ): Promise<Blob> {
    const queryString = new URLSearchParams({ format })
    Object.entries(params).forEach(([key, value]) => {
      if (value !== undefined) {
        queryString.append(key, String(value))
      }
    })

    const response = await apiFetch(
      `${this.baseUrl}/export/events?${queryString}`
    )
    return response.blob()
  }

  // 获取WebSocket连接统计
  async getWebSocketStats(): Promise<any> {
    return apiFetchJson(`${this.baseUrl}/ws/stats`)
  }

  // 获取实时事件流（Server-Sent Events）
  subscribeToRealtimeEvents(
    onEvent: (event: any) => void,
    onConnectionChange?: (connected: boolean) => void
  ): () => void {
    const eventSource = new EventSource(
      buildApiUrl(`${this.baseUrl}/realtime/events`)
    )

    eventSource.onopen = () => {
      onConnectionChange?.(true)
    }

    eventSource.onmessage = event => {
      try {
        const data = JSON.parse(event.data)
        onEvent(data)
      } catch (error) {
        logger.error('解析实时事件失败:', error)
      }
    }

    eventSource.onerror = error => {
      logger.error('实时事件流错误:', error)
      eventSource.close()
      onConnectionChange?.(false)
    }

    // 返回取消订阅函数
    return () => {
      eventSource.close()
      onConnectionChange?.(false)
    }
  }

  // 广播实时消息
  async broadcastMessage(
    messageType: string,
    data: any,
    userId?: string,
    sessionId?: string
  ): Promise<any> {
    const queryString = new URLSearchParams({ message_type: messageType })
    if (userId) queryString.append('user_id', userId)
    if (sessionId) queryString.append('session_id', sessionId)

    return apiFetchJson(`${this.baseUrl}/realtime/broadcast?${queryString}`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(data),
    })
  }

  // 健康检查
  async healthCheck(): Promise<any> {
    return apiFetchJson(`${this.baseUrl}/health`)
  }

  // 创建WebSocket连接
  connectWebSocket(userId?: string, sessionId?: string): WebSocket {
    let wsUrl = this.wsUrl
    const params = new URLSearchParams()

    if (userId) params.append('user_id', userId)
    if (sessionId) params.append('session_id', sessionId)

    if (params.toString()) {
      wsUrl += '?' + params.toString()
    }

    const ws = new WebSocket(wsUrl)

    // 设置心跳
    let heartbeatInterval: ReturnType<typeof setTimeout>

    ws.onopen = () => {
      logger.log('WebSocket连接已建立')

      // 开始心跳
      heartbeatInterval = setInterval(() => {
        if (ws.readyState === WebSocket.OPEN) {
          ws.send(JSON.stringify({ action: 'ping' }))
        }
      }, 30000)
    }

    ws.onclose = () => {
      logger.log('WebSocket连接已关闭')
      if (heartbeatInterval) {
        clearInterval(heartbeatInterval)
      }
    }

    ws.onerror = error => {
      logger.error('WebSocket错误:', error)
      if (
        ws.readyState !== WebSocket.CLOSING &&
        ws.readyState !== WebSocket.CLOSED
      ) {
        ws.close()
      }
    }

    return ws
  }

  // 订阅实时数据
  subscribeToRealtime(ws: WebSocket, subscriptionType: string): void {
    if (ws.readyState === WebSocket.OPEN) {
      ws.send(
        JSON.stringify({
          action: 'subscribe',
          type: subscriptionType,
        })
      )
    }
  }

  // 取消订阅
  unsubscribeFromRealtime(ws: WebSocket, subscriptionType: string): void {
    if (ws.readyState === WebSocket.OPEN) {
      ws.send(
        JSON.stringify({
          action: 'unsubscribe',
          type: subscriptionType,
        })
      )
    }
  }

  // 批量提交事件的便捷方法
  async submitEvent(
    eventType: string,
    userId: string,
    properties: Record<string, any> = {},
    context: Record<string, any> = {}
  ): Promise<any> {
    const event: BehaviorEvent = {
      event_id: crypto.randomUUID(),
      user_id: userId,
      event_type: eventType,
      timestamp: new Date().toISOString(),
      properties,
      context,
    }

    return this.submitEvents({ events: [event] })
  }

  // 实时事件提交（用于页面埋点）
  trackEvent(eventType: string, properties: Record<string, any> = {}): void {
    // 获取用户ID和会话ID（实际应用中从认证状态获取）
    const userId = this.getCurrentUserId()
    const sessionId = this.getCurrentSessionId()

    if (userId) {
      // 异步提交，不阻塞用户操作
      this.submitEvent(eventType, userId, properties, {
        page: window.location.pathname,
        user_agent: navigator.userAgent,
        timestamp: new Date().toISOString(),
      }).catch(error => {
        logger.error('事件跟踪失败:', error)
      })
    }
  }

  // 获取当前用户ID
  private getCurrentUserId(): string | null {
    const key = 'user_id'
    let userId = localStorage.getItem(key)
    if (!userId) {
      userId = crypto.randomUUID()
      localStorage.setItem(key, userId)
    }
    return userId
  }

  // 获取当前会话ID
  private getCurrentSessionId(): string | null {
    const key = 'session_id'
    let sessionId = sessionStorage.getItem(key)
    if (!sessionId) {
      sessionId = crypto.randomUUID()
      sessionStorage.setItem(key, sessionId)
    }
    return sessionId
  }
}

export const behaviorAnalyticsService = new BehaviorAnalyticsService()
