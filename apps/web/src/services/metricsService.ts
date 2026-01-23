/**
 * 指标服务
 */
import { buildWsUrl } from '../utils/apiBase'
import apiClient from './apiClient'

import { logger } from '../utils/logger'
// 指标数据
export interface MetricData {
  name: string
  value: number
  unit?: string
  change: number
  significant: boolean
  pValue?: number
  confidence?: number
  timestamp: Date
}

// 告警数据
export interface AlertData {
  id: string
  title: string
  description: string
  severity: 'info' | 'warning' | 'critical'
  metric?: string
  threshold?: number
  actualValue?: number
  timestamp: Date
  acknowledged: boolean
}

// 健康状态
export interface HealthStatus {
  status: 'healthy' | 'warning' | 'error'
  issues: string[]
  lastCheck: Date
  metrics: {
    srm: boolean
    dataQuality: boolean
    sampleSize: boolean
    performance: boolean
  }
}

// 时间序列数据点
export interface TimeSeriesPoint {
  timestamp: Date
  value: number
  variant?: string
}

// 指标参数
export interface MetricsParams {
  timeRange?: string
  granularity?: 'minute' | 'hour' | 'day'
  variants?: string[]
  metrics?: string[]
}

class MetricsService {
  private baseUrl = ''

  /**
   * 获取实验指标
   */
  async getExperimentMetrics(
    experimentId: string,
    params: MetricsParams = {}
  ): Promise<MetricData[]> {
    const response = await apiClient.get(
      `${this.baseUrl}/realtime-metrics/${experimentId}`,
      { params }
    )
    return response.data.metrics
  }

  /**
   * 获取时间序列数据
   */
  async getTimeSeries(
    experimentId: string,
    metric: string,
    params: MetricsParams = {}
  ): Promise<TimeSeriesPoint[]> {
    const response = await apiClient.get(
      `${this.baseUrl}/realtime-metrics/${experimentId}/timeseries`,
      {
        params: {
          metric,
          ...params,
        },
      }
    )
    return response.data
  }

  /**
   * 获取实时统计
   */
  async getRealTimeStats(experimentId: string): Promise<any> {
    const response = await apiClient.get(
      `${this.baseUrl}/realtime-metrics/${experimentId}/stats`
    )
    return response.data
  }

  /**
   * 获取告警列表
   */
  async getAlerts(experimentId: string): Promise<AlertData[]> {
    const response = await apiClient.get(`${this.baseUrl}/alert-rules/alerts`, {
      params: { experiment_id: experimentId },
    })
    return response.data.alerts
  }

  /**
   * 确认告警
   */
  async acknowledgeAlert(alertId: string): Promise<void> {
    await apiClient.post(
      `${this.baseUrl}/alert-rules/alerts/${alertId}/acknowledge`
    )
  }

  /**
   * 获取健康状态
   */
  async getHealthStatus(experimentId: string): Promise<HealthStatus> {
    const response = await apiClient.get(
      `${this.baseUrl}/experiments/${experimentId}/health`
    )
    return response.data
  }

  /**
   * 获取变体对比
   */
  async getVariantComparison(experimentId: string): Promise<any> {
    const response = await apiClient.get(
      `${this.baseUrl}/statistical-analysis/compare-variants`,
      { params: { experiment_id: experimentId } }
    )
    return response.data
  }

  /**
   * 获取转化漏斗
   */
  async getConversionFunnel(
    experimentId: string,
    steps: string[]
  ): Promise<any> {
    const response = await apiClient.post(
      `${this.baseUrl}/realtime-metrics/${experimentId}/funnel`,
      { steps }
    )
    return response.data
  }

  /**
   * 获取异常检测结果
   */
  async getAnomalies(experimentId: string): Promise<any> {
    const response = await apiClient.get(
      `${this.baseUrl}/anomaly-detection/detect`,
      { params: { experiment_id: experimentId } }
    )
    return response.data.anomalies
  }

  /**
   * 订阅实时更新 (WebSocket)
   */
  subscribeToUpdates(
    experimentId: string,
    onUpdate: (data: any) => void
  ): () => void {
    const ws = new WebSocket(
      buildWsUrl(`/ws/experiments/${experimentId}/metrics`)
    )

    ws.onmessage = event => {
      try {
        const data = JSON.parse(event.data)
        onUpdate(data)
      } catch (error) {
        logger.error('解析指标更新失败:', error)
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

    return () => {
      ws.close()
    }
  }

  /**
   * 导出指标数据
   */
  async exportMetrics(
    experimentId: string,
    format: 'csv' | 'json' | 'excel' = 'csv'
  ): Promise<Blob> {
    const response = await apiClient.get(
      `${this.baseUrl}/realtime-metrics/${experimentId}/export`,
      {
        params: { format },
        responseType: 'blob',
      }
    )
    return response.data
  }

  /**
   * 获取指标历史
   */
  async getMetricHistory(
    experimentId: string,
    metric: string,
    days: number = 7
  ): Promise<any> {
    const response = await apiClient.get(
      `${this.baseUrl}/realtime-metrics/${experimentId}/history`,
      {
        params: {
          metric,
          days,
        },
      }
    )
    return response.data
  }

  /**
   * 计算统计显著性
   */
  async calculateSignificance(
    experimentId: string,
    metric: string
  ): Promise<{
    pValue: number
    significant: boolean
    confidence: number
    effect: number
  }> {
    const response = await apiClient.post(
      `${this.baseUrl}/statistical-analysis/significance`,
      {
        experiment_id: experimentId,
        metric,
      }
    )
    return response.data
  }

  /**
   * 获取性能指标
   */
  async getPerformanceMetrics(experimentId: string): Promise<any> {
    const response = await apiClient.get(
      `${this.baseUrl}/realtime-metrics/${experimentId}/performance`
    )
    return response.data
  }

  /**
   * 获取用户分布
   */
  async getUserDistribution(experimentId: string): Promise<any> {
    const response = await apiClient.get(
      `${this.baseUrl}/experiments/${experimentId}/distribution`
    )
    return response.data
  }
}

export const metricsService = new MetricsService()
