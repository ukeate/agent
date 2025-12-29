import apiClient from './apiClient';

// 模型接口
export interface Model {
  id: string;
  name: string;
  version: string;
  type: 'classification' | 'regression' | 'nlp' | 'vision' | 'multimodal';
  status: 'active' | 'inactive' | 'training' | 'deployed';
  description?: string;
  created_at: string;
  updated_at?: string;
  size_mb: number;
  accuracy?: number;
  performance_metrics?: Record<string, number>;
}

// 部署配置接口
export interface DeploymentConfig {
  model_name: string;
  model_version?: string;
  deployment_type: 'docker' | 'kubernetes' | 'edge';
  replicas?: number;
  cpu_request?: string;
  cpu_limit?: string;
  memory_request?: string;
  memory_limit?: string;
  gpu_required?: boolean;
  gpu_count?: number;
  port?: number;
  environment_vars?: Record<string, string>;
}

// 部署状态接口
export interface Deployment {
  deployment_id: string;
  model_name: string;
  model_version: string;
  deployment_type: string;
  status: string;
  endpoint_url?: string;
  error_message?: string;
  created_at?: string;
  updated_at?: string;
}

// 学习会话接口
export interface LearningSession {
  session_id: string;
  model_name: string;
  model_version: string;
  status: 'active' | 'paused' | 'completed' | 'failed';
  created_at: string;
  updated_at: string;
  config?: Record<string, any>;
  feedback_count: number;
  update_count: number;
  performance_metrics: Record<string, number>;
  pending_feedback: number;
  buffer_usage: number;
  buffer_capacity: number;
}

// AB测试接口
export interface ABTest {
  test_id: string;
  name: string;
  control_model: string;
  treatment_models: string[];
  traffic_split: Record<string, number>;
  total_users: number;
  sample_counts: Record<string, number>;
}

// 预测请求接口
export interface PredictionRequest {
  model_name: string;
  version?: string;
  input_data: any;
  options?: {
    temperature?: number;
    max_tokens?: number;
    top_p?: number;
  };
}

// 预测响应接口
export interface PredictionResponse {
  prediction: any;
  confidence: number;
  processing_time_ms: number;
  model_info: {
    name: string;
    version: string;
  };
}

// 批量预测请求接口
export interface BatchPredictionRequest {
  model_name: string;
  version?: string;
  input_batch: any[];
  batch_size?: number;
  options?: Record<string, any>;
}

// 监控数据接口
export interface MonitoringOverview {
  total_models: number;
  active_deployments: number;
  total_requests: number;
  average_latency: number;
  error_rate: number;
  resource_utilization: {
    cpu_percent: number;
    memory_percent: number;
    gpu_percent?: number;
  };
}

// 警报接口
export interface Alert {
  id: string;
  type: 'performance' | 'error' | 'resource' | 'availability';
  severity: 'low' | 'medium' | 'high' | 'critical';
  message: string;
  model_name?: string;
  deployment_id?: string;
  created_at: string;
  resolved: boolean;
  resolved_at?: string;
}

// 模型统计接口
export interface ModelStatistics {
  total_models: number;
  models_by_type: Record<string, number>;
  models_by_status: Record<string, number>;
  total_deployments: number;
  active_deployments: number;
  total_predictions: number;
  average_model_size_mb: number;
  most_used_models: Array<{
    name: string;
    usage_count: number;
  }>;
}

class ModelService {
  private baseUrl = '/model-service';

  // ========== 模型管理 ==========
  async uploadModel(formData: FormData): Promise<Model> {
    const response = await apiClient.post(`${this.baseUrl}/models/upload`, formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      }
    });
    return response.data;
  }

  async registerModelFromHub(modelInfo: {
    hub_model_id: string;
    name: string;
    version: string;
    description?: string;
  }): Promise<Model> {
    const response = await apiClient.post(`${this.baseUrl}/models/register-from-hub`, modelInfo);
    return response.data;
  }

  async getModels(): Promise<Model[]> {
    const response = await apiClient.get(`${this.baseUrl}/models`);
    return response.data?.models || [];
  }

  async getModelVersion(modelName: string, version: string): Promise<Model> {
    const response = await apiClient.get(`${this.baseUrl}/models/${modelName}/versions/${version}`);
    return response.data;
  }

  async deleteModelVersion(modelName: string, version: string): Promise<void> {
    await apiClient.delete(`${this.baseUrl}/models/${modelName}/versions/${version}`);
  }

  async validateModel(modelName: string, version: string, validationData?: any): Promise<{
    valid: boolean;
    validation_score: number;
    issues: string[];
    recommendations: string[];
  }> {
    const response = await apiClient.post(`${this.baseUrl}/models/${modelName}/versions/${version}/validate`, {
      validation_data: validationData
    });
    return response.data;
  }

  // ========== 推理服务 ==========
  async predict(request: PredictionRequest): Promise<PredictionResponse> {
    const response = await apiClient.post(`${this.baseUrl}/inference/predict`, request);
    return response.data;
  }

  async batchPredict(request: BatchPredictionRequest): Promise<{
    batch_id: string;
    predictions: PredictionResponse[];
    total_processing_time_ms: number;
  }> {
    const response = await apiClient.post(`${this.baseUrl}/inference/batch-predict`, request);
    return response.data;
  }

  async getLoadedModels(): Promise<Array<{
    name: string;
    version: string;
    status: 'loading' | 'ready' | 'error';
    memory_usage_mb: number;
    last_used: string;
  }>> {
    const response = await apiClient.get(`${this.baseUrl}/inference/models/loaded`);
    return response.data?.loaded_models || [];
  }

  async loadModel(modelName: string, version?: string): Promise<{
    status: string;
    message: string;
    estimated_load_time_seconds: number;
  }> {
    const response = await apiClient.post(`${this.baseUrl}/inference/models/${modelName}/load`, {
      version
    });
    return response.data;
  }

  async unloadModel(modelName: string): Promise<void> {
    await apiClient.delete(`${this.baseUrl}/inference/models/${modelName}/unload`);
  }

  // ========== 部署管理 ==========
  async deployModel(config: DeploymentConfig): Promise<{
    deployment_id: string;
    message: string;
    deployment_type: string;
  }> {
    const response = await apiClient.post(`${this.baseUrl}/deployment/deploy`, config);
    return response.data;
  }

  async getDeployment(deploymentId: string): Promise<Deployment> {
    const response = await apiClient.get(`${this.baseUrl}/deployment/${deploymentId}`);
    return response.data;
  }

  async listDeployments(): Promise<Deployment[]> {
    const response = await apiClient.get(`${this.baseUrl}/deployment/list`);
    return response.data?.deployments || [];
  }

  async deleteDeployment(deploymentId: string): Promise<void> {
    await apiClient.delete(`${this.baseUrl}/deployment/${deploymentId}`);
  }

  // ========== 学习管理 ==========
  async startLearningSession(modelName: string, learningConfig: {
    dataset_id?: string;
    model_version?: string;
    learning_rate?: number;
    batch_size?: number;
    epochs?: number;
  }): Promise<{
    session_id: string;
    message: string;
  }> {
    const { model_version, ...config } = learningConfig || {};
    const response = await apiClient.post(
      `${this.baseUrl}/learning/start`,
      config,
      { params: { model_name: modelName, model_version: model_version || 'latest' } }
    );
    return response.data;
  }

  async provideFeedback(sessionId: string, feedback: {
    prediction_id: string;
    inputs: Record<string, any>;
    expected_output: any;
    actual_output: any;
    feedback_type: string;
    quality_score?: number;
    user_id?: string;
    metadata?: Record<string, any>;
  }): Promise<{
    message: string;
  }> {
    const response = await apiClient.post(`${this.baseUrl}/learning/${sessionId}/feedback`, feedback);
    return response.data;
  }

  async updateLearningSession(sessionId: string, updates?: Record<string, any>): Promise<Record<string, any>> {
    const response = await apiClient.post(`${this.baseUrl}/learning/${sessionId}/update`, updates);
    return response.data;
  }

  async getLearningSessionStats(sessionId: string): Promise<LearningSession> {
    const response = await apiClient.get(`${this.baseUrl}/learning/${sessionId}/stats`);
    return response.data;
  }

  async getLearningSessions(): Promise<LearningSession[]> {
    const response = await apiClient.get(`${this.baseUrl}/learning/sessions`);
    return response.data?.sessions || [];
  }

  async getLearningHistory(sessionId: string): Promise<Array<{ timestamp: string; update_count: number; metrics: Record<string, any> }>> {
    const response = await apiClient.get(`${this.baseUrl}/learning/${sessionId}/history`);
    return response.data?.history || [];
  }

  async pauseLearningSession(sessionId: string): Promise<{ session_id: string; status: string }> {
    const response = await apiClient.post(`${this.baseUrl}/learning/${sessionId}/pause`);
    return response.data;
  }

  async resumeLearningSession(sessionId: string): Promise<{ session_id: string; status: string }> {
    const response = await apiClient.post(`${this.baseUrl}/learning/${sessionId}/resume`);
    return response.data;
  }

  async stopLearningSession(sessionId: string): Promise<{ session_id: string; status: string }> {
    const response = await apiClient.post(`${this.baseUrl}/learning/${sessionId}/stop`);
    return response.data;
  }

  // ========== AB测试 ==========
  async createABTest(testConfig: {
    name: string;
    model_a: string;
    model_b: string;
    traffic_split: number;
    duration_hours?: number;
    success_metrics: string[];
  }): Promise<ABTest> {
    const trafficRatio = Math.min(100, Math.max(0, testConfig.traffic_split)) / 100;
    const payload = {
      name: testConfig.name,
      description: testConfig.name,
      control_model: testConfig.model_a,
      treatment_models: [testConfig.model_b],
      traffic_split: {
        [testConfig.model_a]: trafficRatio,
        [testConfig.model_b]: 1 - trafficRatio,
      },
      success_metrics: testConfig.success_metrics,
      max_duration_days: testConfig.duration_hours ? Math.ceil(testConfig.duration_hours / 24) : 30,
    };
    const response = await apiClient.post(`${this.baseUrl}/abtest/create`, payload);
    return response.data;
  }

  async assignABTestVariant(testId: string, userId?: string): Promise<{
    model_id: string;
    test_id: string;
    user_id: string;
  }> {
    const response = await apiClient.get(`${this.baseUrl}/abtest/${testId}/assign`, {
      params: { user_id: userId }
    });
    return response.data;
  }

  async recordABTestMetric(testId: string, userId: string, metrics: Record<string, number>): Promise<{ message: string }> {
    const response = await apiClient.post(`${this.baseUrl}/abtest/${testId}/record`, metrics, {
      params: { user_id: userId }
    });
    return response.data;
  }

  async getABTestResults(testId: string): Promise<Record<string, any>> {
    const response = await apiClient.get(`${this.baseUrl}/abtest/${testId}/results`);
    return response.data;
  }

  async listABTests(): Promise<ABTest[]> {
    const response = await apiClient.get(`${this.baseUrl}/abtest/list`);
    return response.data?.tests || [];
  }

  // ========== 监控管理 ==========
  async getMonitoringOverview(): Promise<MonitoringOverview> {
    const response = await apiClient.get(`${this.baseUrl}/monitoring/overview`);
    return response.data;
  }

  async getMonitoringDashboard(): Promise<{
    overview: MonitoringOverview;
    recent_deployments: Deployment[];
    active_learning_sessions: LearningSession[];
    recent_alerts: Alert[];
    performance_trends: Array<{
      timestamp: string;
      requests_per_second: number;
      average_latency_ms: number;
      error_rate: number;
    }>;
  }> {
    const response = await apiClient.get(`${this.baseUrl}/monitoring/dashboard`);
    return response.data;
  }

  async getAlerts(severity?: string): Promise<Alert[]> {
    const params = severity ? { severity } : {};
    const response = await apiClient.get(`${this.baseUrl}/monitoring/alerts`, { params });
    return response.data?.alerts || [];
  }

  async getMonitoringRecommendations(): Promise<Array<{
    type: 'performance' | 'cost' | 'scaling' | 'maintenance';
    priority: 'low' | 'medium' | 'high';
    title: string;
    description: string;
    estimated_impact: string;
    action_required: string;
  }>> {
    const response = await apiClient.get(`${this.baseUrl}/monitoring/recommendations`);
    return response.data;
  }

  async getMonitoringMetric(metricName: string, timeRange?: string): Promise<{
    metric_name: string;
    data_points: Array<{
      timestamp: string;
      value: number;
    }>;
    summary: {
      min: number;
      max: number;
      average: number;
      current: number;
    };
  }> {
    const params = timeRange ? { time_range: timeRange } : {};
    const response = await apiClient.get(`${this.baseUrl}/monitoring/metrics/${metricName}`, { params });
    return response.data;
  }

  // ========== 统计信息 ==========
  async getModelStatistics(): Promise<ModelStatistics> {
    const response = await apiClient.get(`${this.baseUrl}/statistics`);
    return response.data;
  }

  // ========== 健康检查 ==========
  async getHealthStatus(): Promise<{
    status: 'healthy' | 'degraded' | 'unhealthy';
    components: {
      model_registry: 'healthy' | 'unhealthy';
      inference_engine: 'healthy' | 'unhealthy';
      deployment_manager: 'healthy' | 'unhealthy';
      learning_engine: 'healthy' | 'unhealthy';
      monitoring_system: 'healthy' | 'unhealthy';
    };
    version: string;
    uptime_seconds: number;
  }> {
    const response = await apiClient.get(`${this.baseUrl}/health`);
    return response.data;
  }
}

export const modelService = new ModelService();
