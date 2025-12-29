import apiClient from './apiClient';

export interface DetectAnomaliesRequest {
  experiment_id: string;
  metric_name: string;
  values: number[];
  timestamps?: string[];
  variant?: string;
  methods?: string[];
}

export interface SRMCheckRequest {
  experiment_id: string;
  control_count: number;
  treatment_count: number;
  expected_ratio?: number;
}

export interface DataQualityCheckRequest {
  experiment_id: string;
  missing_rate: number;
  duplicate_rate: number;
  null_count: number;
  total_count: number;
}

export interface ConfigureDetectionRequest {
  methods: string[];
  sensitivity: number;
  window_size: number;
  min_samples: number;
  z_threshold: number;
  iqr_multiplier: number;
  enable_seasonal: boolean;
  enable_trend: boolean;
}

export interface RealTimeMonitorRequest {
  experiment_id: string;
  metrics: string[];
  check_interval: number;
  alert_threshold: string;
}

export interface Anomaly {
  timestamp: string;
  type: string;
  severity: string;
  metric: string;
  variant?: string;
  observed: number;
  expected: number;
  deviation: number;
  method: string;
  confidence: number;
  description: string;
  metadata: any;
}

export interface AnomalyDetectionResponse {
  success: boolean;
  anomalies: Anomaly[];
  total_count: number;
  methods_used: string[];
}

export interface SRMCheckResponse {
  success: boolean;
  has_srm: boolean;
  anomaly?: {
    severity: string;
    observed_ratio: number;
    expected_ratio: number;
    confidence: number;
    description: string;
    metadata: any;
  };
  message?: string;
}

export interface DataQualityResponse {
  success: boolean;
  quality_issues: Array<{
    type: string;
    severity: string;
    description: string;
    value: number;
    metadata: any;
  }>;
  has_issues: boolean;
  quality_score: number;
}

export interface AnomalySummaryResponse {
  success: boolean;
  summary: any;
}

export interface ConfigurationResponse {
  success: boolean;
  message: string;
  config: {
    methods: string[];
    sensitivity: number;
    window_size: number;
    min_samples: number;
    z_threshold: number;
    iqr_multiplier: number;
    enable_seasonal: boolean;
    enable_trend: boolean;
  };
}

export interface AnomalyTypesResponse {
  success: boolean;
  types: Array<{
    value: string;
    name: string;
    description: string;
  }>;
}

export interface DetectionMethodsResponse {
  success: boolean;
  methods: Array<{
    value: string;
    name: string;
    description: string;
  }>;
}

export interface BatchDetectionResponse {
  success: boolean;
  results: Record<string, Record<string, {
    anomaly_count: number;
    severities: string[];
  }>>;
}

export interface HealthResponse {
  success: boolean;
  service: string;
  status: string;
  config: {
    methods: string[];
    sensitivity: number;
  };
}

export const anomalyDetectionService = {
  // 检测异常
  async detectAnomalies(request: DetectAnomaliesRequest): Promise<AnomalyDetectionResponse> {
    const response = await apiClient.post<AnomalyDetectionResponse>('/anomalies/detect', request);
    return response.data;
  },

  // 检查样本比例不匹配
  async checkSampleRatioMismatch(request: SRMCheckRequest): Promise<SRMCheckResponse> {
    const response = await apiClient.post<SRMCheckResponse>('/anomalies/check-srm', request);
    return response.data;
  },

  // 检查数据质量
  async checkDataQuality(request: DataQualityCheckRequest): Promise<DataQualityResponse> {
    const response = await apiClient.post<DataQualityResponse>('/anomalies/check-data-quality', request);
    return response.data;
  },

  // 获取异常摘要
  async getAnomalySummary(
    experimentId: string,
    startTime?: string,
    endTime?: string
  ): Promise<AnomalySummaryResponse> {
    const params = new URLSearchParams();
    if (startTime) params.append('start_time', startTime);
    if (endTime) params.append('end_time', endTime);
    
    const response = await apiClient.get<AnomalySummaryResponse>(
      `/anomalies/summary/${experimentId}?${params.toString()}`
    );
    return response.data;
  },

  // 配置检测参数
  async configureDetection(request: ConfigureDetectionRequest): Promise<ConfigurationResponse> {
    const response = await apiClient.post<ConfigurationResponse>('/anomalies/configure', request);
    return response.data;
  },

  // 设置实时监控
  async setupRealTimeMonitoring(request: RealTimeMonitorRequest): Promise<{ success: boolean; message: string }> {
    const response = await apiClient.post('/anomalies/real-time-monitor', request);
    return response.data;
  },

  // 获取异常类型列表
  async getAnomalyTypes(): Promise<AnomalyTypesResponse> {
    const response = await apiClient.get<AnomalyTypesResponse>('/anomalies/types');
    return response.data;
  },

  // 获取检测方法列表
  async getDetectionMethods(): Promise<DetectionMethodsResponse> {
    const response = await apiClient.get<DetectionMethodsResponse>('/anomalies/methods');
    return response.data;
  },

  // 批量检测异常
  async batchDetectAnomalies(experiments: string[], metrics: string[]): Promise<BatchDetectionResponse> {
    const params = new URLSearchParams();
    experiments.forEach(exp => params.append('experiments', exp));
    metrics.forEach(metric => params.append('metrics', metric));
    
    const response = await apiClient.post<BatchDetectionResponse>(
      `/anomalies/batch-detect?${params.toString()}`
    );
    return response.data;
  },

  // 健康检查
  async healthCheck(): Promise<HealthResponse> {
    const response = await apiClient.get<HealthResponse>('/anomalies/health');
    return response.data;
  },
};

export default anomalyDetectionService;
