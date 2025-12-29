import apiClient from './apiClient';

export interface MetricPoint {
  timestamp: string;
  value: number;
}

export interface Metric {
  name: string;
  type: 'counter' | 'gauge' | 'histogram' | 'summary';
  description: string;
  unit: string;
  current_value: number | null;
  points: MetricPoint[];
  statistics: {
    count: number;
    min: number;
    max: number;
    avg: number;
    median: number;
    latest: number;
  };
}

export interface Alert {
  id: string;
  metric_name: string;
  level: 'info' | 'warning' | 'critical' | 'emergency';
  message: string;
  timestamp: string;
  threshold: number;
  actual_value: number;
  resolved: boolean;
}

export interface DashboardData {
  timestamp: string;
  metrics: Record<string, Metric>;
  alerts: {
    active: Alert[];
    recent: Alert[];
  };
  summary: {
    system_status: 'healthy' | 'warning' | 'critical';
    total_metrics: number;
    active_alerts: number;
    critical_alerts: number;
    key_metrics: Record<string, { value: number | null; unit: string }>;
  };
}

class MonitoringService {
  private baseUrl = '/monitoring';

  async getDashboardData(): Promise<DashboardData> {
    const response = await apiClient.get(`${this.baseUrl}/dashboard`);
    return response.data;
  }

  async getMetrics(): Promise<Record<string, Metric>> {
    const response = await apiClient.get(`${this.baseUrl}/metrics`);
    if (response.data && response.data.system_metrics) {
      return this.transformBackendMetrics(response.data);
    }
    return response.data;
  }

  private transformBackendMetrics(backendData: any): Record<string, Metric> {
    const metrics: Record<string, Metric> = {};
    
    // 转换系统指标
    if (backendData.system_metrics) {
      const sys = backendData.system_metrics;
      if (sys.cpu_usage !== undefined) {
        metrics.cpu_usage = {
          name: 'cpu_usage',
          type: 'gauge',
          description: 'CPU使用率',
          unit: '%',
          current_value: sys.cpu_usage,
          points: [],
          statistics: { count: 100, min: 0, max: 100, avg: sys.cpu_usage, median: sys.cpu_usage, latest: sys.cpu_usage }
        };
      }
      if (sys.memory_usage !== undefined) {
        metrics.memory_usage = {
          name: 'memory_usage',
          type: 'gauge',
          description: '内存使用率', 
          unit: '%',
          current_value: sys.memory_usage,
          points: [],
          statistics: { count: 100, min: 0, max: 100, avg: sys.memory_usage, median: sys.memory_usage, latest: sys.memory_usage }
        };
      }
    }

    // 转换API指标
    if (backendData.api_metrics) {
      const api = backendData.api_metrics;
      if (api.response_time !== undefined) {
        metrics.api_response_time = {
          name: 'api_response_time',
          type: 'histogram',
          description: 'API响应时间',
          unit: 'ms',
          current_value: api.response_time,
          points: [],
          statistics: { count: api.request_count || 100, min: 50, max: 500, avg: api.response_time, median: api.response_time, latest: api.response_time }
        };
      }
    }

    return metrics;
  }

  async getMetric(metricName: string): Promise<Metric> {
    const response = await apiClient.get(`${this.baseUrl}/metrics/${metricName}`);
    return response.data;
  }

  async getMetricHistory(metricName: string, startTime?: string, endTime?: string): Promise<MetricPoint[]> {
    const params = { start: startTime, end: endTime };
    const response = await apiClient.get(`${this.baseUrl}/metrics/${metricName}/history`, { params });
    return response.data;
  }

  async getAlerts(status?: 'active' | 'resolved'): Promise<Alert[]> {
    const params = status ? { status } : {};
    const response = await apiClient.get(`${this.baseUrl}/alerts`, { params });
    return response.data;
  }

  async acknowledgeAlert(alertId: string): Promise<void> {
    await apiClient.post(`${this.baseUrl}/alerts/${alertId}/acknowledge`);
  }

  async resolveAlert(alertId: string): Promise<void> {
    await apiClient.post(`${this.baseUrl}/alerts/${alertId}/resolve`);
  }

  async getSystemStatus(): Promise<'healthy' | 'warning' | 'critical'> {
    const response = await apiClient.get(`${this.baseUrl}/status`);
    return response.data.status;
  }

  async exportMetrics(format: 'json' | 'csv' | 'prometheus'): Promise<Blob> {
    const response = await apiClient.get(`${this.baseUrl}/export`, {
      params: { format },
      responseType: 'blob'
    });
    return response.data;
  }

  async getModulesStatus(): Promise<any> {
    const response = await apiClient.get('/modules/status');
    return response.data;
  }

  // 系统优化 - 调用未使用的API
  async optimizeSystem(params?: {
    target_metrics?: string[];
    optimization_level?: 'basic' | 'aggressive' | 'balanced';
    auto_apply?: boolean;
  }): Promise<{
    recommendations: Array<{
      metric: string;
      current_value: number;
      recommended_value: number;
      impact: string;
      priority: 'low' | 'medium' | 'high';
    }>;
    estimated_improvement: number;
    applied: boolean;
  }> {
    const response = await apiClient.post('/monitoring/optimize', params || {});
    return response.data;
  }

  // 获取追踪信息 - 调用未使用的API
  async getTrace(traceId: string): Promise<{
    trace_id: string;
    service_name: string;
    operation: string;
    start_time: string;
    duration_ms: number;
    status: 'ok' | 'error';
    spans: Array<{
      span_id: string;
      parent_span_id?: string;
      operation: string;
      start_time: string;
      duration_ms: number;
      tags: Record<string, any>;
      logs: Array<{
        timestamp: string;
        message: string;
        level: string;
      }>;
    }>;
    metadata: Record<string, any>;
  }> {
    const response = await apiClient.get(`/monitoring/traces/${traceId}`);
    return response.data;
  }
}

export const monitoringService = new MonitoringService();
