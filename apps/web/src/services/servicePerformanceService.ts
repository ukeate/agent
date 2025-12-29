import apiClient from './apiClient';

export interface MetricData {
  timestamp: string;
  requests: number;
  response_time: number;
  error_rate: number;
  throughput: number;
  cpu_usage: number;
  memory_usage: number;
  network_io: number;
  disk_io: number;
  connections: number;
  registrations: number;
  discoveries: number;
  health_checks: number;
}

export interface ServiceMetric {
  service_name: string;
  service_type: string;
  requests: number;
  response_time: number;
  error_rate: number;
  uptime: number;
  connections: number;
  region: string;
  status: 'healthy' | 'warning' | 'error';
}

export interface LoadBalancerMetric {
  strategy: string;
  requests: number;
  success_rate: number;
  avg_response_time: number;
  distribution: Record<string, number>;
}

export interface OverallMetrics {
  total_requests: number;
  avg_response_time: number;
  error_rate: number;
  uptime: number;
  active_services: number;
  healthy_services: number;
  total_throughput: number;
  peak_throughput: number;
}

export interface PerformanceReport {
  overall_metrics: OverallMetrics;
  metrics_data: MetricData[];
  service_metrics: ServiceMetric[];
  load_balancer_metrics: LoadBalancerMetric[];
  timestamp: string;
}

class ServicePerformanceService {
  private baseUrl = '/service-performance';

  async getPerformanceMetrics(timeRange: string = '1h'): Promise<PerformanceReport> {
    const response = await apiClient.get(`${this.baseUrl}/metrics`, {
      params: { time_range: timeRange }
    });
    return response.data;
  }

  async getServiceMetrics(serviceName?: string): Promise<ServiceMetric[]> {
    const params = serviceName ? { service_name: serviceName } : {};
    const response = await apiClient.get(`${this.baseUrl}/services`, { params });
    return response.data;
  }

  async getLoadBalancerMetrics(): Promise<LoadBalancerMetric[]> {
    const response = await apiClient.get(`${this.baseUrl}/load-balancer`);
    return response.data;
  }

  async getHistoricalData(startTime: string, endTime: string, granularity: string = '5m'): Promise<MetricData[]> {
    const response = await apiClient.get(`${this.baseUrl}/historical`, {
      params: {
        start_time: startTime,
        end_time: endTime,
        granularity
      }
    });
    return response.data;
  }

  async getServiceHealth(serviceName: string): Promise<{
    status: 'healthy' | 'warning' | 'error';
    uptime: number;
    last_check: string;
    issues: string[];
  }> {
    const response = await apiClient.get(`${this.baseUrl}/services/${serviceName}/health`);
    return response.data;
  }

  async exportMetrics(format: 'json' | 'csv' | 'excel' = 'json', timeRange: string = '1h'): Promise<Blob> {
    const response = await apiClient.get(`${this.baseUrl}/export`, {
      params: { format, time_range: timeRange },
      responseType: 'blob'
    });
    return response.data;
  }

  async getAlerts(severity?: 'info' | 'warning' | 'critical'): Promise<Array<{
    id: string;
    service_name: string;
    metric: string;
    threshold: number;
    current_value: number;
    severity: string;
    message: string;
    timestamp: string;
  }>> {
    const params = severity ? { severity } : {};
    const response = await apiClient.get(`${this.baseUrl}/alerts`, { params });
    return response.data;
  }

  async getCapacityUtilization(): Promise<{
    cpu: { current: number; limit: number; trend: 'up' | 'down' | 'stable' };
    memory: { current: number; limit: number; trend: 'up' | 'down' | 'stable' };
    disk: { current: number; limit: number; trend: 'up' | 'down' | 'stable' };
    network: { current: number; limit: number; trend: 'up' | 'down' | 'stable' };
  }> {
    const response = await apiClient.get(`${this.baseUrl}/capacity`);
    return response.data;
  }
}

export const servicePerformanceService = new ServicePerformanceService();