import { buildWsUrl } from '../utils/apiBase'
import apiClient from './apiClient';

import { logger } from '../utils/logger'
// ==================== 类型定义 ====================

export type HealthStatus = 'HEALTHY' | 'DEGRADED' | 'UNHEALTHY';
export type ServiceStatus = 'UP' | 'DOWN' | 'UNKNOWN';

export interface HealthCheckResult {
  status: HealthStatus;
  timestamp: string;
  version?: string;
  uptime?: number;
  services?: ServiceHealthStatus;
  metrics?: SystemMetrics;
  errors?: string[];
  warnings?: string[];
}

export interface ServiceHealthStatus {
  database?: ComponentHealth;
  redis?: ComponentHealth;
  qdrant?: ComponentHealth;
  api?: ComponentHealth;
  worker?: ComponentHealth;
  [key: string]: ComponentHealth | undefined;
}

export interface ComponentHealth {
  status: ServiceStatus;
  latency?: number;
  message?: string;
  error?: string;
  last_check?: string;
}

export interface SystemMetrics {
  cpu_usage?: number;
  memory_usage?: number;
  disk_usage?: number;
  request_rate?: number;
  error_rate?: number;
  response_time_ms?: number;
}

export interface LivenessCheck {
  status: 'alive' | 'dead';
  error?: string;
}

export interface ReadinessCheck {
  status: 'ready' | 'not_ready';
  services_ready?: boolean;
  database_ready?: boolean;
  cache_ready?: boolean;
  message?: string;
}

export interface DetailedHealthInfo {
  status: HealthStatus;
  timestamp: string;
  components: {
    [key: string]: {
      status: ServiceStatus;
      type: string;
      metrics?: {
        uptime: number;
        requests_total?: number;
        errors_total?: number;
        latency_p50?: number;
        latency_p95?: number;
        latency_p99?: number;
      };
      dependencies?: string[];
      last_error?: {
        message: string;
        timestamp: string;
        count: number;
      };
    };
  };
  system: {
    version: string;
    environment: string;
    start_time: string;
    uptime_seconds: number;
    hostname?: string;
  };
  checks: {
    name: string;
    status: 'pass' | 'fail' | 'warn';
    duration_ms: number;
    message?: string;
  }[];
}

export interface HealthAlert {
  id: string;
  severity: 'info' | 'warning' | 'error' | 'critical';
  component: string;
  message: string;
  timestamp: string;
  resolved: boolean;
  resolution_time?: string;
}

export interface HealthTrend {
  timestamp: string;
  status: HealthStatus;
  availability: number;
  error_rate: number;
  response_time: number;
}

// ==================== Service Class ====================

class HealthService {
  private baseUrl = '/health';
  private pollingInterval: number | null = null;
  private healthCallbacks: ((health: HealthCheckResult) => void)[] = [];

  // ==================== 基础健康检查 ====================

  async getHealth(detailed: boolean = false): Promise<HealthCheckResult> {
    const response = await apiClient.get(this.baseUrl, {
      params: { detailed }
    });
    return response.data;
  }

  async checkLiveness(): Promise<LivenessCheck> {
    const response = await apiClient.get(`${this.baseUrl}/live`);
    return response.data;
  }

  async checkReadiness(): Promise<ReadinessCheck> {
    const response = await apiClient.get(`${this.baseUrl}/ready`);
    return response.data;
  }

  // ==================== 详细健康信息 ====================

  async getDetailedHealth(): Promise<DetailedHealthInfo> {
    const response = await apiClient.get(this.baseUrl, {
      params: { detailed: true }
    });
    return response.data;
  }

  async getSystemMetrics(): Promise<SystemMetrics> {
    const response = await apiClient.get(`${this.baseUrl}/metrics`);
    return response.data;
  }

  // ==================== 健康监控 ====================

  startHealthMonitoring(intervalMs: number = 30000, callback?: (health: HealthCheckResult) => void): void {
    if (callback) {
      this.healthCallbacks.push(callback);
    }

    if (this.pollingInterval) {
      return; // 已经在监控中
    }

    this.pollingInterval = window.setInterval(async () => {
      try {
        const health = await this.getHealth();
        this.healthCallbacks.forEach(cb => cb(health));
      } catch (error) {
        logger.error('健康监控错误:', error);
        this.healthCallbacks.forEach(cb => cb({
          status: 'UNHEALTHY',
          timestamp: new Date().toISOString(),
          errors: ['获取健康状态失败']
        }));
      }
    }, intervalMs);

    // 立即执行一次
    this.getHealth().then(health => {
      this.healthCallbacks.forEach(cb => cb(health));
    });
  }

  stopHealthMonitoring(): void {
    if (this.pollingInterval) {
      clearInterval(this.pollingInterval);
      this.pollingInterval = null;
    }
    this.healthCallbacks = [];
  }

  // ==================== 健康告警 ====================

  async getHealthAlerts(params?: {
    severity?: string;
    component?: string;
    resolved?: boolean;
    limit?: number;
  }): Promise<HealthAlert[]> {
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

  // ==================== 健康趋势 ====================

  async getHealthTrends(params?: {
    start_time?: string;
    end_time?: string;
    granularity?: 'minute' | 'hour' | 'day';
  }): Promise<HealthTrend[]> {
    const response = await apiClient.get(`${this.baseUrl}/trends`, { params });
    return response.data;
  }

  async getAvailability(period: '24h' | '7d' | '30d' = '24h'): Promise<{
    availability_percentage: number;
    total_uptime_seconds: number;
    total_downtime_seconds: number;
    incidents: number;
  }> {
    const response = await apiClient.get(`${this.baseUrl}/availability`, {
      params: { period }
    });
    return response.data;
  }

  // ==================== 诊断功能 ====================

  async runDiagnostics(): Promise<{
    timestamp: string;
    results: {
      test: string;
      status: 'pass' | 'fail';
      duration_ms: number;
      details?: any;
    }[];
    recommendations: string[];
  }> {
    const response = await apiClient.get(`${this.baseUrl}/diagnostics`);
    return response.data;
  }

  async getHealthReport(format: 'json' | 'html' = 'json'): Promise<any> {
    const response = await apiClient.get(`${this.baseUrl}/report`, {
      params: { format },
      responseType: format === 'html' ? 'text' : 'json'
    });
    return response.data;
  }

  // ==================== WebSocket实时健康监控 ====================

  connectHealthStream(onUpdate: (data: any) => void): () => void {
    const wsUrl = buildWsUrl(`${this.baseUrl}/stream`);
    const ws = new WebSocket(wsUrl);

    ws.onopen = () => {
      logger.log('健康监控流连接已建立');
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        onUpdate(data);
      } catch (error) {
        logger.error('解析健康更新失败:', error);
      }
    };

    ws.onerror = (error) => {
      logger.error('健康监控WebSocket错误:', error);
      if (ws.readyState !== WebSocket.CLOSING && ws.readyState !== WebSocket.CLOSED) {
        ws.close();
      }
    };

    ws.onclose = () => {
      logger.log('健康监控流连接已断开');
    };

    // 返回断开连接函数
    return () => {
      ws.close();
    };
  }

  // ==================== 实用方法 ====================

  isHealthy(status: HealthStatus): boolean {
    return status === 'HEALTHY';
  }

  isDegraded(status: HealthStatus): boolean {
    return status === 'DEGRADED';
  }

  isUnhealthy(status: HealthStatus): boolean {
    return status === 'UNHEALTHY';
  }

  getStatusColor(status: HealthStatus): string {
    switch (status) {
      case 'HEALTHY':
        return '#52c41a';
      case 'DEGRADED':
        return '#faad14';
      case 'UNHEALTHY':
        return '#f5222d';
      default:
        return '#d9d9d9';
    }
  }

  getStatusIcon(status: HealthStatus): string {
    switch (status) {
      case 'HEALTHY':
        return '✅';
      case 'DEGRADED':
        return '⚠️';
      case 'UNHEALTHY':
        return '❌';
      default:
        return '❓';
    }
  }
}

// ==================== 导出 ====================

export const healthService = new HealthService();
export default healthService;
