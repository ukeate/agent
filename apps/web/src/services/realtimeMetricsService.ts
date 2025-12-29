import { buildWsUrl } from '../utils/apiBase'
import apiClient from './apiClient';

import { logger } from '../utils/logger'
// ==================== 类型定义 ====================

export type MetricType = 'conversion' | 'continuous' | 'count' | 'ratio';
export type MetricCategory = 'primary' | 'secondary' | 'guardrail' | 'diagnostic';
export type AggregationType = 'sum' | 'avg' | 'min' | 'max' | 'count' | 'percentile';
export type TimeWindow = 'REAL_TIME' | 'HOURLY' | 'DAILY' | 'WEEKLY' | 'CUMULATIVE';

export interface MetricDefinition {
  id?: string;
  name: string;
  display_name: string;
  metric_type: MetricType;
  category: MetricCategory;
  aggregation: AggregationType;
  unit?: string;
  description?: string;
  formula?: string;
  numerator_event?: string;
  denominator_event?: string;
  threshold_lower?: number;
  threshold_upper?: number;
}

export interface MetricSnapshot {
  metric_name: string;
  value: number;
  timestamp: string;
  change_percentage?: number;
  trend?: 'up' | 'down' | 'stable';
  status?: 'normal' | 'warning' | 'critical';
}

export interface MetricTrend {
  metric_name: string;
  timestamps: string[];
  values: number[];
  aggregation: AggregationType;
  granularity: TimeWindow;
}

export interface MetricComparison {
  metric_name: string;
  control_value: number;
  treatment_value: number;
  difference: number;
  percentage_change: number;
  statistical_significance?: boolean;
  p_value?: number;
  confidence_interval?: [number, number];
}

export interface RealtimeMetricsData {
  timestamp: string;
  metrics: MetricSnapshot[];
  summary: {
    total_metrics: number;
    critical_metrics: number;
    warning_metrics: number;
    healthy_metrics: number;
  };
}

export interface MetricAlert {
  id: string;
  metric_name: string;
  alert_type: 'threshold' | 'anomaly' | 'trend';
  severity: 'info' | 'warning' | 'critical';
  message: string;
  value: number;
  threshold?: number;
  triggered_at: string;
  resolved?: boolean;
  resolved_at?: string;
}

export interface MetricsDashboard {
  id: string;
  name: string;
  metrics: string[];
  layout: any;
  refresh_interval: number;
  time_range: TimeWindow;
}

// ==================== Service Class ====================

class RealtimeMetricsService {
  private baseUrl = '/realtime-metrics';

  // ==================== 指标定义管理 ====================

  async createMetricDefinition(definition: MetricDefinition): Promise<MetricDefinition> {
    const response = await apiClient.post(`${this.baseUrl}/definitions`, definition);
    return response.data;
  }

  async getMetricDefinition(metricName: string): Promise<MetricDefinition> {
    const response = await apiClient.get(`${this.baseUrl}/definitions/${metricName}`);
    return response.data;
  }

  async listMetricDefinitions(category?: MetricCategory): Promise<MetricDefinition[]> {
    const params = category ? { category } : {};
    const response = await apiClient.get(`${this.baseUrl}/definitions`, { params });
    return response.data;
  }

  async updateMetricDefinition(metricName: string, updates: Partial<MetricDefinition>): Promise<MetricDefinition> {
    const response = await apiClient.put(`${this.baseUrl}/definitions/${metricName}`, updates);
    return response.data;
  }

  async deleteMetricDefinition(metricName: string): Promise<{ success: boolean }> {
    const response = await apiClient.delete(`${this.baseUrl}/definitions/${metricName}`);
    return response.data;
  }

  // ==================== 实时指标获取 ====================

  async getRealtimeMetrics(metrics?: string[]): Promise<RealtimeMetricsData> {
    const params = metrics ? { metrics: metrics.join(',') } : {};
    const response = await apiClient.get(`${this.baseUrl}/realtime`, { params });
    return response.data;
  }

  async getMetricSnapshot(metricName: string): Promise<MetricSnapshot> {
    const response = await apiClient.get(`${this.baseUrl}/snapshot/${metricName}`);
    return response.data;
  }

  async getMetricValue(metricName: string): Promise<{
    value: number;
    timestamp: string;
    unit?: string;
  }> {
    const response = await apiClient.get(`${this.baseUrl}/value/${metricName}`);
    return response.data;
  }

  // ==================== 指标计算与分析 ====================

  async calculateMetrics(params: {
    experiment_id: string;
    time_window?: TimeWindow;
    metrics?: string[];
  }): Promise<MetricSnapshot[]> {
    const response = await apiClient.post(`${this.baseUrl}/calculate`, params);
    return response.data;
  }

  async compareGroups(params: {
    experiment_id: string;
    control_group: string;
    treatment_group: string;
    metrics?: string[];
  }): Promise<MetricComparison[]> {
    const response = await apiClient.post(`${this.baseUrl}/compare`, params);
    return response.data;
  }

  async getMetricTrends(params: {
    experiment_id?: string;
    metric_name: string;
    granularity?: TimeWindow;
    start_time?: string;
    end_time?: string;
  }): Promise<MetricTrend> {
    const response = await apiClient.post(`${this.baseUrl}/trends`, params);
    return response.data;
  }

  async detectAnomalies(metricName: string, params?: {
    sensitivity?: 'low' | 'medium' | 'high';
    time_window?: TimeWindow;
  }): Promise<{
    anomalies: Array<{
      timestamp: string;
      value: number;
      expected_value: number;
      deviation: number;
      severity: string;
    }>;
  }> {
    const response = await apiClient.post(`${this.baseUrl}/anomalies/${metricName}`, params || {});
    return response.data;
  }

  // ==================== 告警管理 ====================

  async getMetricAlerts(params?: {
    metric_name?: string;
    severity?: string;
    resolved?: boolean;
    limit?: number;
  }): Promise<MetricAlert[]> {
    const response = await apiClient.get(`${this.baseUrl}/alerts`, { params });
    return response.data;
  }

  async acknowledgeAlert(alertId: string): Promise<{ success: boolean }> {
    const response = await apiClient.post(`${this.baseUrl}/alerts/${alertId}/acknowledge`);
    return response.data;
  }

  async resolveAlert(alertId: string): Promise<{ success: boolean }> {
    const response = await apiClient.post(`${this.baseUrl}/alerts/${alertId}/resolve`);
    return response.data;
  }

  async configureAlertThreshold(metricName: string, params: {
    lower_threshold?: number;
    upper_threshold?: number;
    alert_on?: 'breach' | 'trend' | 'both';
  }): Promise<{ success: boolean }> {
    const response = await apiClient.post(`${this.baseUrl}/alerts/configure/${metricName}`, params);
    return response.data;
  }

  // ==================== 仪表板管理 ====================

  async createDashboard(dashboard: Omit<MetricsDashboard, 'id'>): Promise<MetricsDashboard> {
    const response = await apiClient.post(`${this.baseUrl}/dashboards`, dashboard);
    return response.data;
  }

  async getDashboard(dashboardId: string): Promise<MetricsDashboard> {
    const response = await apiClient.get(`${this.baseUrl}/dashboards/${dashboardId}`);
    return response.data;
  }

  async updateDashboard(dashboardId: string, updates: Partial<MetricsDashboard>): Promise<MetricsDashboard> {
    const response = await apiClient.put(`${this.baseUrl}/dashboards/${dashboardId}`, updates);
    return response.data;
  }

  async deleteDashboard(dashboardId: string): Promise<{ success: boolean }> {
    const response = await apiClient.delete(`${this.baseUrl}/dashboards/${dashboardId}`);
    return response.data;
  }

  // ==================== 导出功能 ====================

  async exportMetrics(params: {
    metrics?: string[];
    start_time?: string;
    end_time?: string;
    format?: 'json' | 'csv' | 'excel';
  }): Promise<Blob> {
    const response = await apiClient.get(`${this.baseUrl}/export`, {
      params,
      responseType: 'blob'
    });
    return response.data;
  }

  async generateReport(params: {
    experiment_id?: string;
    metrics?: string[];
    time_window?: TimeWindow;
    format?: 'pdf' | 'html';
  }): Promise<Blob> {
    const response = await apiClient.post(`${this.baseUrl}/report`, params, {
      responseType: 'blob'
    });
    return response.data;
  }

  // ==================== WebSocket 实时流 ====================

  connectMetricsStream(
    metrics: string[],
    onUpdate: (data: MetricSnapshot) => void,
    onError?: (error: any) => void
  ): () => void {
    const wsUrl = buildWsUrl(`${this.baseUrl}/stream?metrics=${metrics.join(',')}`);
    const ws = new WebSocket(wsUrl);

    ws.onopen = () => {
      logger.log('指标流连接已建立');
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        onUpdate(data);
      } catch (error) {
        logger.error('解析指标更新失败:', error);
        onError?.(error);
      }
    };

    ws.onerror = (error) => {
      logger.error('指标流连接错误:', error);
      onError?.(error);
      if (ws.readyState !== WebSocket.CLOSING && ws.readyState !== WebSocket.CLOSED) {
        ws.close();
      }
    };

    ws.onclose = () => {
      logger.log('指标流连接已断开');
    };

    // 返回清理函数
    return () => {
      ws.close();
    };
  }

  // ==================== 配置管理 ====================

  async getConfiguration(): Promise<{
    update_interval_seconds: number;
    retention_days: number;
    aggregation_windows: TimeWindow[];
    enabled_categories: MetricCategory[];
    alert_enabled: boolean;
  }> {
    const response = await apiClient.get(`${this.baseUrl}/config`);
    return response.data;
  }

  async updateConfiguration(config: Partial<{
    update_interval_seconds?: number;
    retention_days?: number;
    alert_enabled?: boolean;
  }>): Promise<{ success: boolean }> {
    const response = await apiClient.put(`${this.baseUrl}/config`, config);
    return response.data;
  }

  // ==================== 匹配 realtime_metrics.py 的端点 ====================

  // 注册指标定义
  async registerMetric(metricDef: {
    name: string;
    display_name: string;
    metric_type: 'conversion' | 'continuous' | 'count' | 'ratio';
    category: 'primary' | 'secondary' | 'guardrail' | 'diagnostic';
    aggregation: string;
    unit?: string;
    description?: string;
    formula?: string;
    numerator_event?: string;
    denominator_event?: string;
    threshold_lower?: number;
    threshold_upper?: number;
  }): Promise<{
    status: string;
    metric: any;
    message: string;
  }> {
    const response = await apiClient.post(`${this.baseUrl}/register-metric`, metricDef);
    return response.data;
  }

  // 计算实验指标
  async calculateExperimentMetrics(request: {
    experiment_id: string;
    time_window: 'cumulative' | 'hourly' | 'daily' | 'weekly';
    metrics?: string[];
  }): Promise<{
    experiment_id: string;
    groups: Record<string, any>;
    timestamp: string;
    time_window: string;
    message: string;
  }> {
    const response = await apiClient.post(`${this.baseUrl}/calculate`, request);
    return response.data;
  }

  // 比较实验组
  async compareExperimentGroups(request: {
    experiment_id: string;
    control_group: string;
    treatment_group: string;
    metrics?: string[];
  }): Promise<{
    experiment_id: string;
    control_group: string;
    treatment_group: string;
    comparisons: Record<string, any>;
    summary: {
      total_metrics: number;
      significant_metrics: number;
      significant_metric_names: string[];
    };
    message: string;
  }> {
    const response = await apiClient.post(`${this.baseUrl}/compare-groups`, request);
    return response.data;
  }

  // 获取指标趋势
  async getExperimentMetricTrends(request: {
    experiment_id: string;
    metric_name: string;
    granularity: 'cumulative' | 'hourly' | 'daily' | 'weekly';
    start_time?: string;
    end_time?: string;
  }): Promise<{
    metric_name: string;
    trends: any[];
    granularity: string;
    message: string;
  }> {
    const response = await apiClient.post(`${this.baseUrl}/trends`, request);
    return response.data;
  }

  // 启动实时监控
  async startRealtimeMonitoring(experimentId: string): Promise<{
    status: string;
    experiment_id: string;
    message: string;
  }> {
    const response = await apiClient.post(`${this.baseUrl}/start-monitoring/${experimentId}`);
    return response.data;
  }

  // 停止实时监控
  async stopRealtimeMonitoring(): Promise<{
    status: string;
    message: string;
  }> {
    const response = await apiClient.post(`${this.baseUrl}/stop-monitoring`);
    return response.data;
  }

  // 获取指标目录
  async getRealtimeMetricsCatalog(): Promise<{
    catalog: {
      primary: any[];
      secondary: any[];
      guardrail: any[];
      diagnostic: any[];
    };
    total_metrics: number;
    message: string;
  }> {
    const response = await apiClient.get(`${this.baseUrl}/metrics-catalog`);
    return response.data;
  }

  // 获取指标定义
  async getRealtimeMetricDefinition(metricName: string): Promise<{
    metric: any;
    message: string;
  }> {
    const response = await apiClient.get(`${this.baseUrl}/metric-definition/${metricName}`);
    return response.data;
  }

  // 获取实验指标摘要
  async getRealtimeExperimentSummary(
    experimentId: string,
    timeWindow: 'cumulative' | 'hourly' | 'daily' | 'weekly' = 'cumulative'
  ): Promise<{
    summary: {
      experiment_id: string;
      time_window: string;
      groups: Record<string, any>;
      primary_metrics: Record<string, any>;
      guardrail_metrics: Record<string, any>;
    };
    message: string;
  }> {
    const response = await apiClient.get(
      `${this.baseUrl}/experiment/${experimentId}/summary`,
      { params: { time_window: timeWindow } }
    );
    return response.data;
  }

  // 健康检查
  async realtimeHealthCheck(): Promise<{
    status: 'healthy' | 'unhealthy';
    service: string;
    redis_status?: string;
    registered_metrics?: number;
    error?: string;
    message: string;
  }> {
    const response = await apiClient.get(`${this.baseUrl}/health`);
    return response.data;
  }
}

// ==================== 导出 ====================

export const realtimeMetricsService = new RealtimeMetricsService();
export default realtimeMetricsService;
