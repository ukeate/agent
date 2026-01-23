import { buildWsUrl } from '../utils/apiBase'
import apiClient from './apiClient'

import { logger } from '../utils/logger'
// ==================== 类型定义 ====================

export interface UnifiedMetrics {
  timestamp: string
  system: {
    cpu_usage: number
    memory_usage: number
    disk_usage: number
    network_io: number
    uptime: number
  }
  application: {
    active_users: number
    requests_per_second: number
    average_response_time: number
    error_rate: number
    success_rate: number
  }
  business: {
    transactions: number
    revenue: number
    conversion_rate: number
    user_engagement: number
  }
  ml_models: {
    active_models: number
    predictions_per_second: number
    average_accuracy: number
    model_latency: number
  }
}

export interface ModuleStatusItem {
  name: string
  import_path: string
  status: 'active' | 'inactive'
  health: 'healthy' | 'unhealthy'
  version: string
  last_check: string
  error?: string | null
}

export interface ModuleStatusSummary {
  total_attempted: number
  loaded: number
  failed: number
  success_rate: string
}

export interface ModulesStatus {
  timestamp: string
  summary: ModuleStatusSummary
  modules: Record<string, ModuleStatusItem>
}

export interface UnifiedDashboard {
  id: string
  name: string
  description: string
  widgets: Widget[]
  layout: any
  refresh_interval: number
  created_at: string
  updated_at: string
}

export interface Widget {
  id: string
  type: 'chart' | 'stat' | 'table' | 'map' | 'alert' | 'custom'
  title: string
  data_source: string
  config: WidgetConfig
  position: {
    x: number
    y: number
    width: number
    height: number
  }
}

export interface WidgetConfig {
  metric?: string
  chart_type?: 'line' | 'bar' | 'pie' | 'area' | 'scatter'
  time_range?: string
  aggregation?: string
  filters?: Record<string, any>
  thresholds?: {
    warning?: number
    critical?: number
  }
}

export interface UnifiedAlert {
  id: string
  source: string
  type: string
  severity: 'info' | 'warning' | 'error' | 'critical'
  title: string
  message: string
  timestamp: string
  acknowledged: boolean
  resolved: boolean
  metadata?: Record<string, any>
}

export interface DataSource {
  id: string
  name: string
  type: 'database' | 'api' | 'file' | 'stream' | 'custom'
  connection_string?: string
  status: 'connected' | 'disconnected' | 'error'
  last_sync?: string
  config?: Record<string, any>
}

export interface UnifiedQuery {
  query: string
  sources?: string[]
  time_range?: {
    start: string
    end: string
  }
  aggregations?: string[]
  filters?: Record<string, any>
}

export interface UnifiedReport {
  id: string
  title: string
  description: string
  sections: ReportSection[]
  generated_at: string
  format: 'pdf' | 'html' | 'json'
}

export interface ReportSection {
  title: string
  type: 'summary' | 'detail' | 'chart' | 'table'
  content: any
  insights?: string[]
}

// ==================== Service Class ====================

class UnifiedService {
  private baseUrl = '/unified'

  // ==================== 统一指标 ====================

  async getUnifiedMetrics(): Promise<UnifiedMetrics> {
    const response = await apiClient.get(`${this.baseUrl}/metrics`)
    return response.data as UnifiedMetrics
  }

  async getModulesStatus(): Promise<ModulesStatus> {
    const response = await apiClient.get('/modules/status')
    return response.data as ModulesStatus
  }

  async getSystemMetrics(): Promise<{
    timestamp: string
    system: any
    application: any
    services: any
  }> {
    const response = await apiClient.get('/metrics')
    return response.data
  }

  async getMonitoringSummary(): Promise<{
    timestamp: string
    overall_health: string
    health_score: number
    active_alerts: number
    services_status: any
    performance_metrics: any
    resource_usage: any
    trends: any
  }> {
    const response = await apiClient.get('/platform/monitoring/metrics')
    const metrics = response.data?.metrics
    if (!Array.isArray(metrics) || metrics.length === 0) {
      return {
        timestamp: new Date().toISOString(),
        overall_health: 'unknown',
        health_score: 0,
        active_alerts: 0,
        services_status: null,
        performance_metrics: null,
        resource_usage: null,
        trends: null,
      }
    }
    const latest = metrics[metrics.length - 1] || {}
    const errorRate = Number.isFinite(Number(latest.error_rate))
      ? Number(latest.error_rate)
      : 0
    const successRate = Math.max(0, 100 - errorRate)
    return {
      timestamp: latest.timestamp || new Date().toISOString(),
      overall_health: errorRate > 10 ? 'degraded' : 'healthy',
      health_score: Math.max(0, Math.round(100 - errorRate)),
      active_alerts: 0,
      services_status: null,
      performance_metrics: {
        avg_response_time: Number.isFinite(Number(latest.response_time))
          ? Number(latest.response_time)
          : 0,
        success_rate_percent: successRate,
      },
      resource_usage: {
        cpu_percent: Number.isFinite(Number(latest.cpu_usage))
          ? Number(latest.cpu_usage)
          : 0,
        memory_percent: Number.isFinite(Number(latest.memory_usage))
          ? Number(latest.memory_usage)
          : 0,
        disk_percent: Number.isFinite(Number(latest.disk_usage))
          ? Number(latest.disk_usage)
          : 0,
      },
      trends: null,
    }
  }

  async getMonitoringAlerts(): Promise<
    Array<{
      id: string
      type: string
      severity: string
      title: string
      message: string
      source: string
      timestamp: string
      status: string
      acknowledged: boolean
      metadata?: any
    }>
  > {
    const response = await apiClient.get('/platform/monitoring/alerts')
    return response.data?.alerts || []
  }

  async getMetricsByCategory(
    category: 'system' | 'application' | 'business' | 'ml_models'
  ): Promise<any> {
    const response = await apiClient.get(`${this.baseUrl}/metrics/${category}`)
    return response.data
  }

  async aggregateMetrics(params: {
    sources: string[]
    metrics: string[]
    aggregation: string
    time_range?: string
  }): Promise<any> {
    const response = await apiClient.post(
      `${this.baseUrl}/metrics/aggregate`,
      params
    )
    return response.data
  }

  // ==================== 仪表板管理 ====================

  async getDashboards(): Promise<UnifiedDashboard[]> {
    const response = await apiClient.get(`${this.baseUrl}/dashboards`)
    return response.data
  }

  async getDashboard(dashboardId: string): Promise<UnifiedDashboard> {
    const response = await apiClient.get(
      `${this.baseUrl}/dashboards/${dashboardId}`
    )
    return response.data
  }

  async createDashboard(
    dashboard: Omit<UnifiedDashboard, 'id' | 'created_at' | 'updated_at'>
  ): Promise<UnifiedDashboard> {
    const response = await apiClient.post(
      `${this.baseUrl}/dashboards`,
      dashboard
    )
    return response.data
  }

  async updateDashboard(
    dashboardId: string,
    updates: Partial<UnifiedDashboard>
  ): Promise<UnifiedDashboard> {
    const response = await apiClient.put(
      `${this.baseUrl}/dashboards/${dashboardId}`,
      updates
    )
    return response.data
  }

  async deleteDashboard(dashboardId: string): Promise<{ success: boolean }> {
    const response = await apiClient.delete(
      `${this.baseUrl}/dashboards/${dashboardId}`
    )
    return response.data
  }

  // ==================== Widget管理 ====================

  async addWidget(
    dashboardId: string,
    widget: Omit<Widget, 'id'>
  ): Promise<Widget> {
    const response = await apiClient.post(
      `${this.baseUrl}/dashboards/${dashboardId}/widgets`,
      widget
    )
    return response.data
  }

  async updateWidget(
    dashboardId: string,
    widgetId: string,
    updates: Partial<Widget>
  ): Promise<Widget> {
    const response = await apiClient.put(
      `${this.baseUrl}/dashboards/${dashboardId}/widgets/${widgetId}`,
      updates
    )
    return response.data
  }

  async removeWidget(
    dashboardId: string,
    widgetId: string
  ): Promise<{ success: boolean }> {
    const response = await apiClient.delete(
      `${this.baseUrl}/dashboards/${dashboardId}/widgets/${widgetId}`
    )
    return response.data
  }

  // ==================== 数据源管理 ====================

  async getDataSources(): Promise<DataSource[]> {
    const response = await apiClient.get(`${this.baseUrl}/datasources`)
    return response.data
  }

  async addDataSource(
    dataSource: Omit<DataSource, 'id' | 'status'>
  ): Promise<DataSource> {
    const response = await apiClient.post(
      `${this.baseUrl}/datasources`,
      dataSource
    )
    return response.data
  }

  async testDataSource(dataSourceId: string): Promise<{
    success: boolean
    message: string
    latency_ms?: number
  }> {
    const response = await apiClient.post(
      `${this.baseUrl}/datasources/${dataSourceId}/test`
    )
    return response.data
  }

  async syncDataSource(dataSourceId: string): Promise<{
    success: boolean
    records_synced: number
    duration_ms: number
  }> {
    const response = await apiClient.post(
      `${this.baseUrl}/datasources/${dataSourceId}/sync`
    )
    return response.data
  }

  async removeDataSource(dataSourceId: string): Promise<{ success: boolean }> {
    const response = await apiClient.delete(
      `${this.baseUrl}/datasources/${dataSourceId}`
    )
    return response.data
  }

  // ==================== 统一查询 ====================

  async executeQuery(query: UnifiedQuery): Promise<{
    results: any[]
    metadata: {
      execution_time_ms: number
      rows_returned: number
      sources_queried: string[]
    }
  }> {
    const response = await apiClient.post(`${this.baseUrl}/query`, query)
    return response.data
  }

  async saveQuery(
    name: string,
    query: UnifiedQuery
  ): Promise<{
    id: string
    name: string
    query: UnifiedQuery
  }> {
    const response = await apiClient.post(`${this.baseUrl}/queries/save`, {
      name,
      ...query,
    })
    return response.data
  }

  async getSavedQueries(): Promise<
    Array<{
      id: string
      name: string
      query: UnifiedQuery
      created_at: string
    }>
  > {
    const response = await apiClient.get(`${this.baseUrl}/queries`)
    return response.data
  }

  // ==================== 告警管理 ====================

  async getUnifiedAlerts(params?: {
    severity?: string
    source?: string
    resolved?: boolean
    limit?: number
  }): Promise<UnifiedAlert[]> {
    const response = await apiClient.get(`${this.baseUrl}/alerts`, { params })
    return response.data
  }

  async acknowledgeAlert(alertId: string): Promise<{ success: boolean }> {
    const response = await apiClient.post(
      `${this.baseUrl}/alerts/${alertId}/acknowledge`
    )
    return response.data
  }

  async resolveAlert(
    alertId: string,
    resolution?: string
  ): Promise<{ success: boolean }> {
    const response = await apiClient.post(
      `${this.baseUrl}/alerts/${alertId}/resolve`,
      {
        resolution,
      }
    )
    return response.data
  }

  async getAlertStatistics(): Promise<{
    total: number
    by_severity: Record<string, number>
    by_source: Record<string, number>
    trend: Array<{ timestamp: string; count: number }>
  }> {
    const response = await apiClient.get(`${this.baseUrl}/alerts/statistics`)
    return response.data
  }

  // ==================== 报告生成 ====================

  async generateUnifiedReport(params: {
    title: string
    description: string
    sections: string[]
    format?: 'pdf' | 'html' | 'json'
    time_range?: {
      start: string
      end: string
    }
  }): Promise<UnifiedReport> {
    const response = await apiClient.post(
      `${this.baseUrl}/reports/generate`,
      params
    )
    return response.data
  }

  async getReport(reportId: string): Promise<UnifiedReport> {
    const response = await apiClient.get(`${this.baseUrl}/reports/${reportId}`)
    return response.data
  }

  async downloadReport(reportId: string): Promise<Blob> {
    const response = await apiClient.get(
      `${this.baseUrl}/reports/${reportId}/download`,
      {
        responseType: 'blob',
      }
    )
    return response.data
  }

  async listReports(params?: { limit?: number; offset?: number }): Promise<{
    reports: UnifiedReport[]
    total: number
  }> {
    const response = await apiClient.get(`${this.baseUrl}/reports`, { params })
    return response.data
  }

  // ==================== 实时同步 ====================

  connectUnifiedStream(
    onUpdate: (data: any) => void,
    sources?: string[]
  ): () => void {
    const params = sources ? `?sources=${sources.join(',')}` : ''
    const wsUrl = buildWsUrl(`${this.baseUrl}/stream${params}`)
    const ws = new WebSocket(wsUrl)

    ws.onopen = () => {
      logger.log('统一流连接已建立')
    }

    ws.onmessage = event => {
      try {
        const data = JSON.parse(event.data)
        onUpdate(data)
      } catch (error) {
        logger.error('解析统一流更新失败:', error)
      }
    }

    ws.onerror = error => {
      logger.error('统一流连接错误:', error)
      if (
        ws.readyState !== WebSocket.CLOSING &&
        ws.readyState !== WebSocket.CLOSED
      ) {
        ws.close()
      }
    }

    // 返回清理函数
    return () => {
      ws.close()
    }
  }

  // ==================== 配置管理 ====================

  async getConfiguration(): Promise<{
    refresh_interval: number
    max_widgets_per_dashboard: number
    max_data_sources: number
    cache_ttl_seconds: number
    features: {
      realtime_sync: boolean
      auto_refresh: boolean
      export_enabled: boolean
    }
  }> {
    const response = await apiClient.get(`${this.baseUrl}/config`)
    return response.data
  }

  async updateConfiguration(config: any): Promise<{ success: boolean }> {
    const response = await apiClient.put(`${this.baseUrl}/config`, config)
    return response.data
  }
}

// ==================== 导出 ====================

export const unifiedService = new UnifiedService()
export default unifiedService
