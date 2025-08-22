interface EventSubmissionRequest {
  events: BehaviorEvent[];
  batch_id?: string;
}

interface BehaviorEvent {
  event_id: string;
  user_id: string;
  session_id?: string;
  event_type: string;
  timestamp: string;
  properties?: Record<string, any>;
  context?: Record<string, any>;
  duration?: number;
}

interface AnalysisRequest {
  user_id?: string;
  session_id?: string;
  start_time?: string;
  end_time?: string;
  event_types?: string[];
  analysis_types: string[];
}

interface PatternQuery {
  user_id?: string;
  pattern_type?: string;
  min_support?: number;
  limit?: number;
}

interface AnomalyQuery {
  user_id?: string;
  severity?: string;
  start_time?: string;
  end_time?: string;
  limit?: number;
  use_real_detection?: boolean;
}

interface ReportRequest {
  report_type: string;
  format: string;
  filters?: Record<string, any>;
  include_visualizations?: boolean;
}

class BehaviorAnalyticsService {
  private baseUrl = '/api/v1/analytics';
  private wsUrl: string;

  constructor() {
    // 构建WebSocket URL
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const host = window.location.host;
    this.wsUrl = `${protocol}//${host}${this.baseUrl}/ws`;
  }

  // 提交用户行为事件
  async submitEvents(request: EventSubmissionRequest): Promise<any> {
    const response = await fetch(`${this.baseUrl}/events`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(request),
    });

    if (!response.ok) {
      throw new Error(`提交事件失败: ${response.statusText}`);
    }

    return response.json();
  }

  // 查询行为事件
  async getEvents(params: {
    user_id?: string;
    session_id?: string;
    event_type?: string;
    start_time?: string;
    end_time?: string;
    limit?: number;
    offset?: number;
  } = {}): Promise<any> {
    const queryString = new URLSearchParams();
    Object.entries(params).forEach(([key, value]) => {
      if (value !== undefined) {
        queryString.append(key, String(value));
      }
    });

    const response = await fetch(`${this.baseUrl}/events?${queryString}`);
    if (!response.ok) {
      throw new Error(`查询事件失败: ${response.statusText}`);
    }

    return response.json();
  }

  // 查询用户会话
  async getSessions(params: {
    user_id?: string;
    start_time?: string;
    end_time?: string;
    min_duration?: number;
    limit?: number;
    offset?: number;
  } = {}): Promise<any> {
    const queryString = new URLSearchParams();
    Object.entries(params).forEach(([key, value]) => {
      if (value !== undefined) {
        queryString.append(key, String(value));
      }
    });

    const response = await fetch(`${this.baseUrl}/sessions?${queryString}`);
    if (!response.ok) {
      throw new Error(`查询会话失败: ${response.statusText}`);
    }

    return response.json();
  }

  // 执行行为分析
  async analyzeBehavior(request: AnalysisRequest): Promise<any> {
    const response = await fetch(`${this.baseUrl}/analyze`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(request),
    });

    if (!response.ok) {
      throw new Error(`行为分析失败: ${response.statusText}`);
    }

    return response.json();
  }

  // 获取行为模式
  async getPatterns(params: PatternQuery = {}): Promise<any> {
    const queryString = new URLSearchParams();
    Object.entries(params).forEach(([key, value]) => {
      if (value !== undefined) {
        queryString.append(key, String(value));
      }
    });

    const response = await fetch(`${this.baseUrl}/patterns?${queryString}`);
    if (!response.ok) {
      throw new Error(`获取模式失败: ${response.statusText}`);
    }

    return response.json();
  }

  // 获取异常检测结果
  async getAnomalies(params: AnomalyQuery = {}): Promise<any> {
    const queryString = new URLSearchParams();
    Object.entries(params).forEach(([key, value]) => {
      if (value !== undefined) {
        queryString.append(key, String(value));
      }
    });

    const response = await fetch(`${this.baseUrl}/anomalies?${queryString}`);
    if (!response.ok) {
      throw new Error(`获取异常失败: ${response.statusText}`);
    }

    return response.json();
  }

  // 生成分析报告
  async generateReport(request: ReportRequest): Promise<any> {
    const response = await fetch(`${this.baseUrl}/reports/generate`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(request),
    });

    if (!response.ok) {
      throw new Error(`生成报告失败: ${response.statusText}`);
    }

    return response.json();
  }

  // 获取报告
  async getReport(reportId: string): Promise<any> {
    const response = await fetch(`${this.baseUrl}/reports/${reportId}`);
    if (!response.ok) {
      throw new Error(`获取报告失败: ${response.statusText}`);
    }

    return response.json();
  }

  // 下载报告
  async downloadReport(reportId: string, format: string = 'json'): Promise<Blob> {
    const response = await fetch(`${this.baseUrl}/reports/${reportId}/download?format=${format}`);
    if (!response.ok) {
      throw new Error(`下载报告失败: ${response.statusText}`);
    }

    return response.blob();
  }

  // 获取仪表板统计数据
  async getDashboardStats(timeRange: string = '24h', userId?: string): Promise<any> {
    const queryString = new URLSearchParams({ time_range: timeRange });
    if (userId) {
      queryString.append('user_id', userId);
    }

    const response = await fetch(`${this.baseUrl}/dashboard/stats?${queryString}`);
    if (!response.ok) {
      throw new Error(`获取统计失败: ${response.statusText}`);
    }

    return response.json();
  }

  // 导出事件数据
  async exportEvents(format: string = 'json', params: {
    user_id?: string;
    start_time?: string;
    end_time?: string;
    limit?: number;
  } = {}): Promise<Blob> {
    const queryString = new URLSearchParams({ format });
    Object.entries(params).forEach(([key, value]) => {
      if (value !== undefined) {
        queryString.append(key, String(value));
      }
    });

    const response = await fetch(`${this.baseUrl}/export/events?${queryString}`);
    if (!response.ok) {
      throw new Error(`导出数据失败: ${response.statusText}`);
    }

    return response.blob();
  }

  // 获取WebSocket连接统计
  async getWebSocketStats(): Promise<any> {
    const response = await fetch(`${this.baseUrl}/ws/stats`);
    if (!response.ok) {
      throw new Error(`获取WebSocket统计失败: ${response.statusText}`);
    }

    return response.json();
  }

  // 广播实时消息
  async broadcastMessage(messageType: string, data: any, userId?: string, sessionId?: string): Promise<any> {
    const queryString = new URLSearchParams({ message_type: messageType });
    if (userId) queryString.append('user_id', userId);
    if (sessionId) queryString.append('session_id', sessionId);

    const response = await fetch(`${this.baseUrl}/realtime/broadcast?${queryString}`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(data),
    });

    if (!response.ok) {
      throw new Error(`广播消息失败: ${response.statusText}`);
    }

    return response.json();
  }

  // 健康检查
  async healthCheck(): Promise<any> {
    const response = await fetch(`${this.baseUrl}/health`);
    if (!response.ok) {
      throw new Error(`健康检查失败: ${response.statusText}`);
    }

    return response.json();
  }

  // 创建WebSocket连接
  connectWebSocket(userId?: string, sessionId?: string): WebSocket {
    let wsUrl = this.wsUrl;
    const params = new URLSearchParams();
    
    if (userId) params.append('user_id', userId);
    if (sessionId) params.append('session_id', sessionId);
    
    if (params.toString()) {
      wsUrl += '?' + params.toString();
    }

    const ws = new WebSocket(wsUrl);
    
    // 设置心跳
    let heartbeatInterval: NodeJS.Timeout;
    
    ws.onopen = () => {
      console.log('WebSocket连接已建立');
      
      // 开始心跳
      heartbeatInterval = setInterval(() => {
        if (ws.readyState === WebSocket.OPEN) {
          ws.send(JSON.stringify({ action: 'ping' }));
        }
      }, 30000);
    };

    ws.onclose = () => {
      console.log('WebSocket连接已关闭');
      if (heartbeatInterval) {
        clearInterval(heartbeatInterval);
      }
    };

    ws.onerror = (error) => {
      console.error('WebSocket错误:', error);
    };

    return ws;
  }

  // 订阅实时数据
  subscribeToRealtime(ws: WebSocket, subscriptionType: string): void {
    if (ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({
        action: 'subscribe',
        type: subscriptionType
      }));
    }
  }

  // 取消订阅
  unsubscribeFromRealtime(ws: WebSocket, subscriptionType: string): void {
    if (ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({
        action: 'unsubscribe',
        type: subscriptionType
      }));
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
      event_id: `${eventType}_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      user_id: userId,
      event_type: eventType,
      timestamp: new Date().toISOString(),
      properties,
      context
    };

    return this.submitEvents({ events: [event] });
  }

  // 实时事件提交（用于页面埋点）
  trackEvent(eventType: string, properties: Record<string, any> = {}): void {
    // 获取用户ID和会话ID（实际应用中从认证状态获取）
    const userId = this.getCurrentUserId();
    const sessionId = this.getCurrentSessionId();

    if (userId) {
      // 异步提交，不阻塞用户操作
      this.submitEvent(eventType, userId, properties, {
        page: window.location.pathname,
        user_agent: navigator.userAgent,
        timestamp: new Date().toISOString()
      }).catch(error => {
        console.error('事件跟踪失败:', error);
      });
    }
  }

  // 获取当前用户ID（示例实现）
  private getCurrentUserId(): string | null {
    // 实际应用中应该从认证状态获取
    return localStorage.getItem('user_id') || 'anonymous_user';
  }

  // 获取当前会话ID（示例实现）
  private getCurrentSessionId(): string | null {
    // 实际应用中应该从会话管理获取
    return sessionStorage.getItem('session_id') || null;
  }
}

export const behaviorAnalyticsService = new BehaviorAnalyticsService();