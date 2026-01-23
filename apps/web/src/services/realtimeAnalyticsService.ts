/**
 * 实时分析仪表板服务
 * 提供实时监控和分析数据
 */

import { buildApiUrl } from '../utils/apiBase'
import apiClient from './apiClient'

import { logger } from '../utils/logger'
// 实时分析数据接口
export interface RealtimeAnalyticsData {
  current_load: {
    active_users: number
    concurrent_sessions: number
    requests_per_minute: number
    data_throughput_mb: number
  }
  system_health: {
    api_response_time: number
    database_connection_pool: number
    cache_hit_rate: number
    error_rate: number
  }
  business_metrics: {
    conversion_rate: number
    revenue_per_hour: number
    customer_satisfaction: number
    active_experiments: number
  }
  ai_model_performance: {
    prediction_accuracy: number
    inference_time_ms: number
    model_confidence: number
    feature_importance_updated: string
  }
}

// 实时指标接口
export interface RealtimeMetric {
  timestamp: string
  metric_name: string
  value: number
  unit: string
  trend: 'up' | 'down' | 'stable'
  alert_level?: 'warning' | 'critical'
}

// 告警配置接口
export interface AlertConfig {
  metric_name: string
  threshold: number
  condition: 'greater_than' | 'less_than' | 'equals'
  enabled: boolean
  notification_channels: string[]
}

// 仪表板配置接口
export interface DashboardConfig {
  dashboard_id: string
  name: string
  widgets: Array<{
    widget_id: string
    type: 'line_chart' | 'bar_chart' | 'gauge' | 'metric_card' | 'table'
    title: string
    metrics: string[]
    position: { x: number; y: number; width: number; height: number }
    refresh_interval_seconds: number
  }>
}

class RealtimeAnalyticsService {
  private baseUrl = '/realtime'

  /**
   * 获取实时分析仪表板数据
   */
  async getAnalyticsDashboard(): Promise<RealtimeAnalyticsData> {
    const response = await apiClient.get(`${this.baseUrl}/analytics-dashboard`)
    return response.data
  }

  /**
   * 获取实时指标数据流
   */
  async getMetricsStream(
    metrics: string[],
    timeRange: '1m' | '5m' | '15m' | '1h' = '5m'
  ): Promise<RealtimeMetric[]> {
    const response = await apiClient.get(`${this.baseUrl}/metrics/stream`, {
      params: {
        metrics: metrics.join(','),
        time_range: timeRange,
      },
    })
    return response.data
  }

  /**
   * 订阅实时数据推送
   */
  subscribeToUpdates(
    metrics: string[],
    callback: (data: RealtimeMetric[]) => void
  ): () => void {
    const eventSource = new EventSource(
      buildApiUrl(`${this.baseUrl}/subscribe?metrics=${metrics.join(',')}`)
    )

    eventSource.onmessage = event => {
      try {
        const data = JSON.parse(event.data)
        callback(data)
      } catch (error) {
        logger.error('解析实时数据失败:', error)
      }
    }

    eventSource.onerror = error => {
      logger.error('实时数据推送连接错误:', error)
      eventSource.close()
    }

    // 返回取消订阅函数
    return () => {
      eventSource.close()
    }
  }

  /**
   * 获取可用指标列表
   */
  async getAvailableMetrics(): Promise<
    Array<{
      name: string
      display_name: string
      description: string
      unit: string
      category: string
    }>
  > {
    const response = await apiClient.get(`${this.baseUrl}/metrics/available`)
    return response.data
  }

  /**
   * 获取系统性能指标
   */
  async getPerformanceMetrics(): Promise<{
    cpu_usage: number
    memory_usage: number
    disk_io: { read: number; write: number }
    network_io: { rx: number; tx: number }
    active_connections: number
    queue_lengths: Record<string, number>
  }> {
    const response = await apiClient.get(`${this.baseUrl}/performance`)
    return response.data
  }

  /**
   * 获取用户活动指标
   */
  async getUserActivityMetrics(): Promise<{
    active_users_now: number
    new_users_today: number
    session_duration_avg: number
    bounce_rate: number
    page_views_per_session: number
    geographic_distribution: Record<string, number>
  }> {
    const response = await apiClient.get(`${this.baseUrl}/user-activity`)
    return response.data
  }

  /**
   * 获取业务指标
   */
  async getBusinessMetrics(): Promise<{
    revenue_today: number
    orders_count: number
    conversion_rate: number
    average_order_value: number
    customer_acquisition_cost: number
    lifetime_value: number
  }> {
    const response = await apiClient.get(`${this.baseUrl}/business`)
    return response.data
  }

  /**
   * 配置告警规则
   */
  async setAlert(
    alertConfig: AlertConfig
  ): Promise<{ message: string; alert_id: string }> {
    const response = await apiClient.post(`${this.baseUrl}/alerts`, alertConfig)
    return response.data
  }

  /**
   * 获取告警列表
   */
  async getAlerts(active_only: boolean = false): Promise<
    Array<
      AlertConfig & {
        alert_id: string
        created_at: string
        last_triggered?: string
        trigger_count: number
      }
    >
  > {
    const response = await apiClient.get(`${this.baseUrl}/alerts`, {
      params: { active_only },
    })
    return response.data
  }

  /**
   * 删除告警规则
   */
  async deleteAlert(alertId: string): Promise<{ message: string }> {
    const response = await apiClient.delete(`${this.baseUrl}/alerts/${alertId}`)
    return response.data
  }

  /**
   * 保存仪表板配置
   */
  async saveDashboard(
    dashboardConfig: DashboardConfig
  ): Promise<{ message: string }> {
    const response = await apiClient.post(
      `${this.baseUrl}/dashboards`,
      dashboardConfig
    )
    return response.data
  }

  /**
   * 获取仪表板配置列表
   */
  async getDashboards(): Promise<DashboardConfig[]> {
    const response = await apiClient.get(`${this.baseUrl}/dashboards`)
    return response.data
  }

  /**
   * 获取特定仪表板配置
   */
  async getDashboard(dashboardId: string): Promise<DashboardConfig> {
    const response = await apiClient.get(
      `${this.baseUrl}/dashboards/${dashboardId}`
    )
    return response.data
  }

  /**
   * 删除仪表板
   */
  async deleteDashboard(dashboardId: string): Promise<{ message: string }> {
    const response = await apiClient.delete(
      `${this.baseUrl}/dashboards/${dashboardId}`
    )
    return response.data
  }

  /**
   * 获取历史数据
   */
  async getHistoricalData(
    metric: string,
    startTime: string,
    endTime: string,
    granularity: '1m' | '5m' | '15m' | '1h' | '1d' = '5m'
  ): Promise<
    Array<{
      timestamp: string
      value: number
    }>
  > {
    const response = await apiClient.get(`${this.baseUrl}/history`, {
      params: {
        metric,
        start_time: startTime,
        end_time: endTime,
        granularity,
      },
    })
    return response.data
  }

  /**
   * 导出分析报告
   */
  async exportReport(
    format: 'pdf' | 'excel' | 'csv',
    metrics: string[],
    timeRange: {
      start: string
      end: string
    }
  ): Promise<Blob> {
    const response = await apiClient.get(`${this.baseUrl}/export`, {
      params: {
        format,
        metrics: metrics.join(','),
        start_time: timeRange.start,
        end_time: timeRange.end,
      },
      responseType: 'blob',
    })
    return response.data
  }

  /**
   * 获取系统健康状态
   */
  async getSystemHealth(): Promise<{
    overall_status: 'healthy' | 'warning' | 'critical'
    services: Array<{
      name: string
      status: 'up' | 'down' | 'degraded'
      response_time: number
      last_check: string
    }>
    resource_usage: {
      cpu: number
      memory: number
      disk: number
    }
  }> {
    const response = await apiClient.get(`${this.baseUrl}/health`)
    return response.data
  }
}

export const realtimeAnalyticsService = new RealtimeAnalyticsService()
