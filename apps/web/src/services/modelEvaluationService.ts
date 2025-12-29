import { buildWsUrl } from '../utils/apiBase'
import apiClient from './apiClient';

import { logger } from '../utils/logger'
// ==================== 类型定义 ====================

export interface EvaluationRequest {
  model_name: string;
  model_path: string;
  task_type?: string;
  device?: string;
  batch_size?: number;
  max_length?: number;
  precision?: string;
  enable_optimizations?: boolean;
}

export interface BenchmarkRequest {
  name: string;
  tasks: string[];
  num_fewshot?: number;
  limit?: number;
  batch_size?: number;
  device?: string;
}

export interface BatchEvaluationRequest {
  models: EvaluationRequest[];
  benchmarks: BenchmarkRequest[];
  parallel_workers?: number;
  save_results?: boolean;
  output_format?: 'json' | 'html' | 'pdf';
}

export interface EvaluationResult {
  evaluation_id: string;
  model_name: string;
  task_type: string;
  metrics: {
    accuracy?: number;
    precision?: number;
    recall?: number;
    f1_score?: number;
    perplexity?: number;
    bleu_score?: number;
    rouge_scores?: {
      rouge1: number;
      rouge2: number;
      rougeL: number;
    };
    [key: string]: any;
  };
  performance: {
    inference_time: number;
    tokens_per_second: number;
    memory_usage: number;
    gpu_utilization?: number;
  };
  timestamp: string;
  duration: number;
  status: 'completed' | 'failed' | 'in_progress';
  error?: string;
}

export interface BenchmarkResult {
  benchmark_id: string;
  benchmark_name: string;
  model_name: string;
  tasks: Array<{
    task_name: string;
    metrics: Record<string, number>;
    samples_evaluated: number;
    duration: number;
  }>;
  overall_score: number;
  comparison?: {
    baseline_model: string;
    improvement: number;
  };
  timestamp: string;
}

export interface ModelComparison {
  models: string[];
  benchmarks: string[];
  comparison_matrix: Array<{
    model: string;
    benchmark: string;
    score: number;
    rank: number;
  }>;
  winner: string;
  summary: {
    best_overall: string;
    best_per_task: Record<string, string>;
    recommendations: string[];
  };
}

export interface PerformanceMetrics {
  model_name: string;
  timestamp: string;
  latency: {
    p50: number;
    p95: number;
    p99: number;
    mean: number;
  };
  throughput: {
    requests_per_second: number;
    tokens_per_second: number;
  };
  resource_usage: {
    cpu_percent: number;
    memory_mb: number;
    gpu_percent?: number;
    gpu_memory_mb?: number;
  };
  error_rate: number;
}

export interface EvaluationReport {
  report_id: string;
  title: string;
  generated_at: string;
  models: string[];
  benchmarks: string[];
  executive_summary: string;
  detailed_results: {
    evaluations: EvaluationResult[];
    benchmarks: BenchmarkResult[];
    comparisons: ModelComparison[];
  };
  visualizations?: {
    charts: Array<{
      type: string;
      title: string;
      data: any;
    }>;
  };
  recommendations: string[];
  download_url?: string;
}

export interface EvaluationHistory {
  evaluations: Array<{
    id: string;
    model_name: string;
    timestamp: string;
    status: string;
    score: number;
  }>;
  total: number;
  page: number;
  limit: number;
}

// ==================== Service Class ====================

class ModelEvaluationService {
  private baseUrl = '/model-evaluation';

  // ==================== 评估管理 ====================

  async startEvaluation(request: EvaluationRequest): Promise<{
    evaluation_id: string;
    status: string;
    message: string;
  }> {
    const response = await apiClient.post(`${this.baseUrl}/evaluate`, request);
    return response.data;
  }

  async listBenchmarks(): Promise<any[]> {
    const response = await apiClient.get(`${this.baseUrl}/benchmarks`);
    return response.data?.benchmarks || response.data || [];
  }

  async listModels(): Promise<any[]> {
    const response = await apiClient.get(`/model-service/models`);
    return response.data?.models || response.data || [];
  }

  async getPerformanceComparison(): Promise<any[]> {
    const response = await apiClient.get(`${this.baseUrl}/performance/comparison`);
    return response.data?.comparisons || response.data || [];
  }

  async startBatchEvaluation(request: BatchEvaluationRequest): Promise<{
    batch_id: string;
    total_tasks: number;
    status: string;
    message: string;
  }> {
    const response = await apiClient.post(`${this.baseUrl}/evaluate/batch`, request);
    return response.data;
  }

  async getEvaluationStatus(evaluationId: string): Promise<EvaluationResult> {
    const response = await apiClient.get(`${this.baseUrl}/evaluation/${evaluationId}`);
    return response.data;
  }

  async getEvaluationResults(evaluationId: string): Promise<EvaluationResult> {
    const response = await apiClient.get(`${this.baseUrl}/evaluation/${evaluationId}/results`);
    return response.data;
  }

  async cancelEvaluation(evaluationId: string): Promise<{ success: boolean; message: string }> {
    const response = await apiClient.post(`${this.baseUrl}/evaluation/${evaluationId}/cancel`);
    return response.data;
  }

  // ==================== 基准测试 ====================

  async runBenchmark(request: BenchmarkRequest): Promise<{
    benchmark_id: string;
    status: string;
    message: string;
  }> {
    const response = await apiClient.post(`${this.baseUrl}/benchmark`, request);
    return response.data;
  }

  async getBenchmarkResults(benchmarkId: string): Promise<BenchmarkResult> {
    const response = await apiClient.get(`${this.baseUrl}/benchmark/${benchmarkId}/results`);
    return response.data;
  }

  async listAvailableBenchmarks(): Promise<Array<{
    name: string;
    description: string;
    tasks: string[];
    difficulty: 'easy' | 'medium' | 'hard';
    estimated_duration: number;
  }>> {
    const response = await apiClient.get(`${this.baseUrl}/benchmarks`);
    return response.data?.benchmarks || [];
  }

  // ==================== 模型比较 ====================

  async compareModels(params: {
    models: string[];
    benchmarks?: string[];
    metrics?: string[];
  }): Promise<ModelComparison> {
    const response = await apiClient.post(`${this.baseUrl}/compare`, params);
    return response.data;
  }

  async getComparisonHistory(limit?: number, offset?: number): Promise<{
    comparisons: ModelComparison[];
    total: number;
  }> {
    const response = await apiClient.get(`${this.baseUrl}/comparisons`, {
      params: { limit, offset }
    });
    return response.data;
  }

  // ==================== 性能监控 ====================

  async getPerformanceMetrics(modelName: string, timeRange?: string): Promise<PerformanceMetrics> {
    const response = await apiClient.get(`${this.baseUrl}/performance/${modelName}`, {
      params: { time_range: timeRange }
    });
    return response.data;
  }

  async getPerformanceHistory(
    modelName: string,
    startTime?: string,
    endTime?: string
  ): Promise<PerformanceMetrics[]> {
    const response = await apiClient.get(`${this.baseUrl}/performance/${modelName}/history`, {
      params: { start_time: startTime, end_time: endTime }
    });
    return response.data;
  }

  async startPerformanceMonitoring(modelName: string, config?: {
    interval_seconds?: number;
    metrics_to_collect?: string[];
  }): Promise<{ monitor_id: string; status: string }> {
    const response = await apiClient.post(`${this.baseUrl}/performance/${modelName}/monitor`, config || {});
    return response.data;
  }

  async stopPerformanceMonitoring(modelName: string, monitorId: string): Promise<{ success: boolean }> {
    const response = await apiClient.delete(`${this.baseUrl}/performance/${modelName}/monitor/${monitorId}`);
    return response.data;
  }

  // ==================== 报告生成 ====================

  async generateReport(params: {
    evaluation_ids?: string[];
    benchmark_ids?: string[];
    comparison_ids?: string[];
    format?: 'json' | 'html' | 'pdf';
    include_visualizations?: boolean;
  }): Promise<EvaluationReport> {
    const response = await apiClient.post(`${this.baseUrl}/report`, params);
    return response.data;
  }

  async getReport(reportId: string): Promise<EvaluationReport> {
    const response = await apiClient.get(`${this.baseUrl}/report/${reportId}`);
    return response.data;
  }

  async downloadReport(reportId: string, format: 'json' | 'html' | 'pdf' = 'pdf'): Promise<Blob> {
    const response = await apiClient.get(`${this.baseUrl}/report/${reportId}/download`, {
      params: { format },
      responseType: 'blob'
    });
    return response.data;
  }

  async listReports(params?: {
    model_name?: string;
    limit?: number;
    offset?: number;
  }): Promise<{
    reports: EvaluationReport[];
    total: number;
  }> {
    const response = await apiClient.get(`${this.baseUrl}/reports`, { params });
    return response.data;
  }

  // ==================== 历史记录 ====================

  async getEvaluationHistory(params?: {
    model_name?: string;
    status?: string;
    limit?: number;
    offset?: number;
  }): Promise<EvaluationHistory> {
    const response = await apiClient.get(`${this.baseUrl}/history`, { params });
    return response.data;
  }

  async deleteEvaluation(evaluationId: string): Promise<{ success: boolean; message: string }> {
    const response = await apiClient.delete(`${this.baseUrl}/evaluation/${evaluationId}`);
    return response.data;
  }

  // ==================== 模型上传 ====================

  async uploadModel(file: File, metadata: {
    model_name: string;
    model_type: string;
    description?: string;
  }): Promise<{
    upload_id: string;
    model_path: string;
    status: string;
  }> {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('metadata', JSON.stringify(metadata));

    const response = await apiClient.post(`${this.baseUrl}/upload`, formData, {
      headers: {
        'Content-Type': 'multipart/form-data'
      }
    });
    return response.data;
  }

  // ==================== 实时监控 ====================

  subscribeToEvaluationUpdates(
    evaluationId: string,
    onUpdate: (data: any) => void
  ): () => void {
    // WebSocket连接实现
    const wsUrl = buildWsUrl(`${this.baseUrl}/evaluation/${evaluationId}/stream`);
    const ws = new WebSocket(wsUrl);

    ws.onopen = () => {
      logger.log(`评估流连接已建立: ${evaluationId}`);
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        onUpdate(data);
      } catch (error) {
        logger.error('解析评估更新失败:', error);
      }
    };

    ws.onerror = (error) => {
      logger.error('WebSocket错误:', error);
      if (ws.readyState !== WebSocket.CLOSING && ws.readyState !== WebSocket.CLOSED) {
        ws.close();
      }
    };

    // 返回取消订阅函数
    return () => {
      ws.close();
    };
  }

  // ==================== 配置管理 ====================

  async getConfiguration(): Promise<{
    default_device: string;
    default_batch_size: number;
    default_precision: string;
    available_models: string[];
    available_benchmarks: string[];
    max_parallel_evaluations: number;
  }> {
    const response = await apiClient.get(`${this.baseUrl}/config`);
    return response.data;
  }

  async updateConfiguration(config: Partial<{
    default_device?: string;
    default_batch_size?: number;
    default_precision?: string;
    max_parallel_evaluations?: number;
  }>): Promise<{ success: boolean; message: string }> {
    const response = await apiClient.put(`${this.baseUrl}/config`, config);
    return response.data;
  }
}

// ==================== 导出 ====================

export const modelEvaluationService = new ModelEvaluationService();
export default modelEvaluationService;
