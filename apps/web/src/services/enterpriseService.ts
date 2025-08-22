import apiClient from './apiClient';

export interface SystemHealthMetrics {
  overall_status: 'healthy' | 'degraded' | 'unhealthy';
  cpu_usage: number;
  memory_usage: number;
  disk_usage: number;
  active_agents: number;
  active_tasks: number;
  error_rate: number;
  response_time: number;
  timestamp: string;
}

export interface SecurityMetrics {
  threat_level: 'low' | 'medium' | 'high' | 'critical';
  detected_attacks: number;
  blocked_requests: number;
  security_events: number;
  compliance_score: number;
  last_security_scan: string;
  active_threats: Array<{
    id: string;
    type: string;
    severity: string;
    description: string;
    timestamp: string;
  }>;
}

export interface PerformanceMetrics {
  throughput: number;
  latency_p50: number;
  latency_p95: number;
  latency_p99: number;
  concurrent_users: number;
  cache_hit_rate: number;
  optimization_level: string;
  resource_utilization: {
    cpu: number;
    memory: number;
    io: number;
    network: number;
  };
}

export interface ComplianceData {
  overall_score: number;
  status: 'compliant' | 'partially_compliant' | 'non_compliant';
  standards: string[];
  last_assessment: string;
  issues_count: number;
  requirements_total: number;
  requirements_passed: number;
  detailed_results: Array<{
    requirement_id: string;
    title: string;
    status: string;
    score: number;
    last_tested: string;
  }>;
}

export interface EnterpriseConfiguration {
  security: {
    trism_enabled: boolean;
    attack_detection_enabled: boolean;
    auto_response_enabled: boolean;
    security_level: string;
  };
  performance: {
    optimization_level: string;
    max_concurrent_tasks: number;
    cache_size: number;
    load_balancing_strategy: string;
  };
  monitoring: {
    otel_enabled: boolean;
    audit_logging_enabled: boolean;
    metrics_retention_days: number;
    alert_thresholds: Record<string, number>;
  };
  compliance: {
    enabled_standards: string[];
    auto_assessment: boolean;
    notification_channels: string[];
  };
}

export interface EnterpriseOverview {
  system_health: SystemHealthMetrics;
  security: SecurityMetrics;
  performance: PerformanceMetrics;
  compliance: ComplianceData;
  configuration: EnterpriseConfiguration;
  last_updated: string;
}

class EnterpriseService {
  private baseUrl = '/api/v1/enterprise';

  /**
   * 获取企业架构总览
   */
  async getOverview(): Promise<EnterpriseOverview> {
    const response = await apiClient.get(`${this.baseUrl}/overview`);
    return response.data;
  }

  /**
   * 获取系统健康指标
   */
  async getSystemHealth(): Promise<SystemHealthMetrics> {
    const response = await apiClient.get(`${this.baseUrl}/health`);
    return response.data;
  }

  /**
   * 获取安全指标
   */
  async getSecurityMetrics(): Promise<SecurityMetrics> {
    const response = await apiClient.get(`${this.baseUrl}/security`);
    return response.data;
  }

  /**
   * 获取性能指标
   */
  async getPerformanceMetrics(): Promise<PerformanceMetrics> {
    const response = await apiClient.get(`${this.baseUrl}/performance`);
    return response.data;
  }

  /**
   * 获取合规数据
   */
  async getComplianceData(): Promise<ComplianceData> {
    const response = await apiClient.get(`${this.baseUrl}/compliance`);
    return response.data;
  }

  /**
   * 获取企业配置
   */
  async getConfiguration(): Promise<EnterpriseConfiguration> {
    const response = await apiClient.get(`${this.baseUrl}/configuration`);
    return response.data;
  }

  /**
   * 更新企业配置
   */
  async updateConfiguration(config: Partial<EnterpriseConfiguration>): Promise<EnterpriseConfiguration> {
    const response = await apiClient.put(`${this.baseUrl}/configuration`, config);
    return response.data;
  }

  /**
   * 运行合规评估
   */
  async runComplianceAssessment(standards?: string[]): Promise<{ assessment_id: string; status: string }> {
    const response = await apiClient.post(`${this.baseUrl}/compliance/assess`, {
      standards: standards || []
    });
    return response.data;
  }

  /**
   * 获取合规评估状态
   */
  async getAssessmentStatus(assessmentId: string): Promise<{ status: string; progress: number; result?: ComplianceData }> {
    const response = await apiClient.get(`${this.baseUrl}/compliance/assess/${assessmentId}`);
    return response.data;
  }

  /**
   * 触发安全扫描
   */
  async triggerSecurityScan(options?: { deep_scan?: boolean; target_components?: string[] }): Promise<{ scan_id: string }> {
    const response = await apiClient.post(`${this.baseUrl}/security/scan`, options || {});
    return response.data;
  }

  /**
   * 获取安全扫描结果
   */
  async getSecurityScanResult(scanId: string): Promise<{ status: string; findings: any[] }> {
    const response = await apiClient.get(`${this.baseUrl}/security/scan/${scanId}`);
    return response.data;
  }

  /**
   * 获取性能优化建议
   */
  async getPerformanceRecommendations(): Promise<{ recommendations: Array<{ type: string; description: string; impact: string; effort: string }> }> {
    const response = await apiClient.get(`${this.baseUrl}/performance/recommendations`);
    return response.data;
  }

  /**
   * 应用性能优化
   */
  async applyPerformanceOptimization(optimizations: string[]): Promise<{ applied: string[]; failed: string[] }> {
    const response = await apiClient.post(`${this.baseUrl}/performance/optimize`, {
      optimizations
    });
    return response.data;
  }

  /**
   * 获取监控告警
   */
  async getAlerts(params?: { level?: string; limit?: number; offset?: number }): Promise<{
    alerts: Array<{
      id: string;
      level: string;
      title: string;
      description: string;
      timestamp: string;
      resolved: boolean;
    }>;
    total: number;
  }> {
    const response = await apiClient.get(`${this.baseUrl}/alerts`, { params });
    return response.data;
  }

  /**
   * 处理告警
   */
  async resolveAlert(alertId: string, resolution?: string): Promise<{ success: boolean }> {
    const response = await apiClient.post(`${this.baseUrl}/alerts/${alertId}/resolve`, {
      resolution
    });
    return response.data;
  }

  /**
   * 获取审计日志
   */
  async getAuditLogs(params?: {
    start_time?: string;
    end_time?: string;
    event_type?: string;
    user_id?: string;
    limit?: number;
    offset?: number;
  }): Promise<{
    logs: Array<{
      event_id: string;
      event_type: string;
      timestamp: string;
      user_id?: string;
      action: string;
      result: string;
      details: Record<string, any>;
    }>;
    total: number;
  }> {
    const response = await apiClient.get(`${this.baseUrl}/audit/logs`, { params });
    return response.data;
  }

  /**
   * 导出合规报告
   */
  async exportComplianceReport(format: 'pdf' | 'excel' | 'json' = 'pdf'): Promise<Blob> {
    const response = await apiClient.get(`${this.baseUrl}/compliance/export`, {
      params: { format },
      responseType: 'blob'
    });
    return response.data;
  }

  /**
   * 获取架构拓扑
   */
  async getArchitectureTopology(): Promise<{
    nodes: Array<{
      id: string;
      type: string;
      name: string;
      status: string;
      metrics: Record<string, number>;
    }>;
    edges: Array<{
      source: string;
      target: string;
      type: string;
      metrics: Record<string, number>;
    }>;
  }> {
    const response = await apiClient.get(`${this.baseUrl}/topology`);
    return response.data;
  }

  /**
   * 获取容量规划建议
   */
  async getCapacityPlanning(timeRange: string = '30d'): Promise<{
    current_usage: Record<string, number>;
    projected_usage: Record<string, number>;
    recommendations: Array<{
      resource: string;
      action: string;
      timeline: string;
      impact: string;
    }>;
  }> {
    const response = await apiClient.get(`${this.baseUrl}/capacity/planning`, {
      params: { time_range: timeRange }
    });
    return response.data;
  }

  /**
   * 测试企业组件连接
   */
  async testConnections(): Promise<{
    components: Array<{
      name: string;
      status: 'healthy' | 'degraded' | 'unhealthy';
      response_time: number;
      error?: string;
    }>;
    overall_status: 'healthy' | 'degraded' | 'unhealthy';
  }> {
    const response = await apiClient.post(`${this.baseUrl}/test/connections`);
    return response.data;
  }

  /**
   * 获取历史趋势数据
   */
  async getTrends(metric: string, timeRange: string = '24h'): Promise<{
    metric: string;
    time_range: string;
    data_points: Array<{
      timestamp: string;
      value: number;
    }>;
    summary: {
      min: number;
      max: number;
      avg: number;
      trend: 'increasing' | 'decreasing' | 'stable';
    };
  }> {
    const response = await apiClient.get(`${this.baseUrl}/trends/${metric}`, {
      params: { time_range: timeRange }
    });
    return response.data;
  }
}

export default new EnterpriseService();