/**
 * 测试管理系统API服务
 * 提供完整的测试管理功能，包括集成测试、性能测试、负载测试等
 */

import { apiClient } from './apiClient';

const API_BASE = '/testing';

// 类型定义
export interface TestSuiteRequest {
  suite_name: string;
  test_types: string[];
  async_execution: boolean;
}

export interface BenchmarkRequest {
  benchmark_types: string[];
  compare_with_baseline: boolean;
}

export interface LoadTestRequest {
  target_qps: number;
  duration_minutes: number;
  ramp_up_seconds: number;
  endpoint_patterns: string[];
}

export interface StressTestRequest {
  max_concurrent_users: number;
  duration_minutes: number;
  failure_threshold: number;
}

export interface SecurityTestRequest {
  test_categories: string[];
  target_endpoints: string[];
  severity_levels: string[];
}

export interface TestResult {
  test_id: string;
  test_type: string;
  status: 'running' | 'completed' | 'failed' | 'cancelled';
  start_time: string;
  end_time?: string;
  duration?: number;
  results: any;
  metrics?: TestMetrics;
}

export interface TestMetrics {
  total_tests: number;
  passed_tests: number;
  failed_tests: number;
  skipped_tests: number;
  execution_time: number;
  coverage_percentage?: number;
  performance_metrics?: {
    avg_response_time: number;
    max_response_time: number;
    min_response_time: number;
    throughput: number;
    error_rate: number;
  };
}

export interface BenchmarkResult {
  benchmark_id: string;
  benchmark_type: string;
  execution_time: number;
  memory_usage: number;
  cpu_usage: number;
  baseline_comparison?: {
    performance_change: number;
    memory_change: number;
    cpu_change: number;
  };
  detailed_metrics: any;
}

export interface SecurityScanResult {
  scan_id: string;
  vulnerabilities: SecurityVulnerability[];
  risk_score: number;
  scan_duration: number;
  recommendations: string[];
}

export interface SecurityVulnerability {
  id: string;
  severity: 'low' | 'medium' | 'high' | 'critical';
  type: string;
  description: string;
  endpoint: string;
  remediation: string;
}

export interface SystemHealthStatus {
  overall_status: 'healthy' | 'warning' | 'critical';
  components: ComponentHealth[];
  last_updated: string;
  uptime: number;
}

export interface ComponentHealth {
  name: string;
  status: 'healthy' | 'degraded' | 'down';
  response_time: number;
  error_rate: number;
  last_check: string;
}

/**
 * 测试管理服务类
 */
export class TestingService {
  /**
   * 运行集成测试套件
   */
  async runIntegrationTests(request: TestSuiteRequest): Promise<TestResult> {
    try {
      const response = await apiClient.post(`${API_BASE}/integration/run`, request);
      return response.data;
    } catch (error) {
      throw error;
    }
  }

  /**
   * 获取集成测试结果
   */
  async getIntegrationTestResult(testId: string): Promise<TestResult> {
    const response = await apiClient.get(`${API_BASE}/integration/results/${testId}`);
    return response.data;
  }

  /**
   * 获取所有集成测试历史
   */
  async getIntegrationTestHistory(): Promise<TestResult[]> {
    try {
      const response = await apiClient.get<{ reports?: TestResult[] }>(`${API_BASE}/integration/reports`);
      return response.data?.reports || [];
    } catch (error) {
      throw error;
    }
  }

  /**
   * 运行性能基准测试
   */
  async runBenchmarkTest(request: BenchmarkRequest): Promise<BenchmarkResult> {
    try {
      const response = await apiClient.post(`${API_BASE}/benchmark/run`, request);
      return response.data;
    } catch (error) {
      throw error;
    }
  }

  /**
   * 获取基准测试结果
   */
  async getBenchmarkResult(benchmarkId: string): Promise<BenchmarkResult> {
    const response = await apiClient.get(`${API_BASE}/benchmark/results/${benchmarkId}`);
    return response.data;
  }

  /**
   * 获取基准测试历史
   */
  async getBenchmarkHistory(): Promise<BenchmarkResult[]> {
    const response = await apiClient.get(`${API_BASE}/benchmark/history`);
    return response.data;
  }

  /**
   * 运行负载测试
   */
  async runLoadTest(request: LoadTestRequest): Promise<TestResult> {
    const response = await apiClient.post(`${API_BASE}/load/run`, request);
    return response.data;
  }

  /**
   * 获取负载测试结果
   */
  async getLoadTestResult(testId: string): Promise<TestResult> {
    const response = await apiClient.get(`${API_BASE}/load/results/${testId}`);
    return response.data;
  }

  /**
   * 停止负载测试
   */
  async stopLoadTest(testId: string): Promise<void> {
    await apiClient.post(`${API_BASE}/load/stop/${testId}`);
  }

  /**
   * 运行压力测试
   */
  async runStressTest(request: StressTestRequest): Promise<TestResult> {
    const response = await apiClient.post(`${API_BASE}/stress/run`, request);
    return response.data;
  }

  /**
   * 获取压力测试结果
   */
  async getStressTestResult(testId: string): Promise<TestResult> {
    const response = await apiClient.get(`${API_BASE}/stress/results/${testId}`);
    return response.data;
  }

  /**
   * 停止压力测试
   */
  async stopStressTest(testId: string): Promise<void> {
    await apiClient.post(`${API_BASE}/stress/stop/${testId}`);
  }

  /**
   * 运行安全测试
   */
  async runSecurityTest(request: SecurityTestRequest): Promise<SecurityScanResult> {
    const response = await apiClient.post(`${API_BASE}/security/scan`, request);
    return response.data;
  }

  /**
   * 获取安全扫描结果
   */
  async getSecurityScanResult(scanId: string): Promise<SecurityScanResult> {
    const response = await apiClient.get(`${API_BASE}/security/results/${scanId}`);
    return response.data;
  }

  /**
   * 获取安全扫描历史
   */
  async getSecurityScanHistory(): Promise<SecurityScanResult[]> {
    const response = await apiClient.get(`${API_BASE}/security/history`);
    return response.data;
  }

  /**
   * 运行渗透测试
   */
  async runPenetrationTest(config: any): Promise<SecurityScanResult> {
    const response = await apiClient.post(`${API_BASE}/security/penetration`, config);
    return response.data;
  }

  /**
   * 获取系统健康状态
   */
  async getSystemHealth(): Promise<SystemHealthStatus> {
    const response = await apiClient.get(`${API_BASE}/health/status`);
    return response.data;
  }

  /**
   * 运行健康检查
   */
  async runHealthCheck(): Promise<SystemHealthStatus> {
    const response = await apiClient.post(`${API_BASE}/health/check`);
    return response.data;
  }

  /**
   * 获取详细的系统监控数据
   */
  async getDetailedMonitoring(): Promise<any> {
    const response = await apiClient.get(`${API_BASE}/health/detailed`);
    return response.data;
  }

  /**
   * 验证数据完整性
   */
  async validateDataIntegrity(): Promise<any> {
    const response = await apiClient.post(`${API_BASE}/validation/data-integrity`);
    return response.data;
  }

  /**
   * 验证API一致性
   */
  async validateAPIConsistency(): Promise<any> {
    const response = await apiClient.post(`${API_BASE}/validation/api-consistency`);
    return response.data;
  }

  /**
   * 验证业务逻辑
   */
  async validateBusinessLogic(): Promise<any> {
    const response = await apiClient.post(`${API_BASE}/validation/business-logic`);
    return response.data;
  }

  /**
   * 获取所有运行中的测试
   */
  async getRunningTests(): Promise<TestResult[]> {
    const response = await apiClient.get(`${API_BASE}/status/running`);
    return response.data;
  }

  /**
   * 取消测试
   */
  async cancelTest(testId: string): Promise<void> {
    await apiClient.post(`${API_BASE}/control/cancel/${testId}`);
  }

  /**
   * 暂停测试
   */
  async pauseTest(testId: string): Promise<void> {
    await apiClient.post(`${API_BASE}/control/pause/${testId}`);
  }

  /**
   * 恢复测试
   */
  async resumeTest(testId: string): Promise<void> {
    await apiClient.post(`${API_BASE}/control/resume/${testId}`);
  }

  /**
   * 获取测试统计信息
   */
  async getTestStatistics(): Promise<any> {
    const response = await apiClient.get(`${API_BASE}/statistics`);
    return response.data;
  }

  /**
   * 生成测试报告
   */
  async generateTestReport(config: {
    test_ids?: string[];
    time_range?: { start: string; end: string };
    report_format: 'json' | 'html' | 'pdf';
  }): Promise<any> {
    const response = await apiClient.post(`${API_BASE}/reports/generate`, config);
    return response.data;
  }

  /**
   * 下载测试报告
   */
  async downloadTestReport(reportId: string, format: string): Promise<Blob> {
    const response = await apiClient.get(`${API_BASE}/reports/download/${reportId}`, {
      params: { format },
      responseType: 'blob'
    });
    return response.data;
  }

  /**
   * 获取测试配置模板
   */
  async getTestTemplates(): Promise<any> {
    const response = await apiClient.get(`${API_BASE}/templates`);
    return response.data;
  }

  /**
   * 保存测试配置模板
   */
  async saveTestTemplate(template: any): Promise<any> {
    const response = await apiClient.post(`${API_BASE}/templates`, template);
    return response.data;
  }

  /**
   * 删除测试配置模板
   */
  async deleteTestTemplate(templateId: string): Promise<void> {
    await apiClient.delete(`${API_BASE}/templates/${templateId}`);
  }

  /**
   * 获取测试环境信息
   */
  async getTestEnvironments(): Promise<any> {
    const response = await apiClient.get(`${API_BASE}/environments`);
    return response.data;
  }

  /**
   * 设置测试环境
   */
  async setupTestEnvironment(config: any): Promise<any> {
    const response = await apiClient.post(`${API_BASE}/environments/setup`, config);
    return response.data;
  }

  /**
   * 清理测试环境
   */
  async cleanupTestEnvironment(environmentId: string): Promise<void> {
    await apiClient.post(`${API_BASE}/environments/cleanup/${environmentId}`);
  }
}

// 导出服务实例
export const testingService = new TestingService();
