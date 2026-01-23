import { buildApiUrl } from '../utils/apiBase'
import { apiClient } from './apiClient'

// 继承现有的类型定义
export * from './fineTuningService'
import { fineTuningService as baseService } from './fineTuningService'

// 增强的接口定义
export interface TrainingMetrics {
  loss_curve: Array<{ epoch: number; loss: number; timestamp: string }>
  learning_rate_curve: Array<{ step: number; lr: number }>
  gpu_utilization: Array<{ timestamp: string; utilization: number }>
  memory_usage: Array<{ timestamp: string; allocated: number; cached: number }>
  throughput: Array<{ epoch: number; samples_per_second: number }>
}

export interface TrainingLogs {
  logs: string[]
  total_lines: number
  last_updated: string
}

export interface ModelValidationResult {
  valid: boolean
  warnings: string[]
  errors: string[]
  model_info: {
    architecture: string
    parameters: number
    max_seq_length: number
    target_modules: string[]
  }
  hardware_recommendations: {
    min_gpu_memory: number
    recommended_batch_size: number
    recommended_gradient_accumulation: number
    quantization_recommendation: string
  }
  compatibility_checks: {
    cuda_compatible: boolean
    memory_sufficient: boolean
    compute_capability: string
  }
}

export interface HardwareConfig {
  cuda_available: boolean
  gpu_count: number
  gpu_memory: number[]
  total_memory: number
  cpu_count: number
  compute_capability: string[]
}

export interface JobStatistics {
  total_jobs: number
  running_jobs: number
  completed_jobs: number
  failed_jobs: number
  avg_completion_time: number
  success_rate: number
  resource_utilization: {
    avg_gpu_usage: number
    avg_memory_usage: number
  }
}

export interface DatasetValidationResult {
  valid: boolean
  format: string
  total_samples: number
  errors: string[]
  warnings: string[]
  statistics: {
    avg_instruction_length: number
    avg_output_length: number
    max_seq_length: number
    unique_samples: number
  }
  sample_preview: Array<{
    instruction: string
    output: string
    input?: string
  }>
}

export interface ConfigurationTemplate {
  name: string
  description: string
  model_type: string
  recommended_use_case: string
  config: {
    training_mode: string
    learning_rate: number
    batch_size: number
    epochs: number
    lora_config?: any
    quantization_config?: any
  }
  hardware_requirements: {
    min_gpu_memory: number
    recommended_gpus: number
  }
}

export interface ExperimentComparison {
  jobs: Array<{
    job_id: string
    job_name: string
    final_loss: number
    training_time: number
    config_summary: any
  }>
  best_config: any
  performance_analysis: {
    fastest_convergence: string
    lowest_loss: string
    most_efficient: string
  }
}

export interface ResourceMonitoring {
  current_usage: {
    gpu_utilization: number[]
    memory_usage: number[]
    temperature: number[]
  }
  historical_data: Array<{
    timestamp: string
    gpu_util: number
    memory_util: number
    power_draw: number
  }>
  alerts: Array<{
    type: string
    message: string
    timestamp: string
    severity: 'info' | 'warning' | 'error'
  }>
}

export interface BatchJobManagement {
  batch_id: string
  name: string
  jobs: string[]
  status: 'pending' | 'running' | 'completed' | 'failed'
  progress: number
  created_at: string
  estimated_completion: string
}

// 增强的微调服务
export const fineTuningServiceEnhanced = {
  // 包含所有基础服务方法
  ...baseService,

  // 增强的训练监控功能
  async getDetailedTrainingMetrics(jobId: string): Promise<TrainingMetrics> {
    const response = await apiClient.get(
      `/fine-tuning/jobs/${jobId}/metrics/detailed`
    )
    return response.data
  },

  async getEnhancedTrainingLogs(
    jobId: string,
    lines = 100,
    level = 'all'
  ): Promise<TrainingLogs> {
    const response = await apiClient.get(
      `/fine-tuning/jobs/${jobId}/logs/enhanced?lines=${lines}&level=${level}`
    )
    return response.data
  },

  async streamTrainingProgress(jobId: string): Promise<EventSource> {
    return new EventSource(
      buildApiUrl(`/fine-tuning/jobs/${jobId}/progress/stream`)
    )
  },

  // 高级模型验证
  async performAdvancedModelValidation(
    config: any
  ): Promise<ModelValidationResult> {
    const response = await apiClient.post(
      '/fine-tuning/models/validate/advanced',
      config
    )
    return response.data
  },

  async getHardwareConfiguration(): Promise<HardwareConfig> {
    const response = await apiClient.get('/fine-tuning/hardware/config')
    return response.data
  },

  async optimizeTrainingConfiguration(
    config: any
  ): Promise<{ optimized_config: any; improvements: string[] }> {
    const response = await apiClient.post(
      '/fine-tuning/configs/optimize',
      config
    )
    return response.data
  },

  // 任务统计和分析
  async getJobStatistics(timeRange = '30d'): Promise<JobStatistics> {
    const response = await apiClient.get(
      `/fine-tuning/jobs/statistics?range=${timeRange}`
    )
    return response.data
  },

  async compareExperiments(jobIds: string[]): Promise<ExperimentComparison> {
    const response = await apiClient.post('/fine-tuning/experiments/compare', {
      job_ids: jobIds,
    })
    return response.data
  },

  async getPerformanceAnalysis(jobId: string): Promise<any> {
    const response = await apiClient.get(
      `/fine-tuning/jobs/${jobId}/analysis/performance`
    )
    return response.data
  },

  // 高级数据集功能
  async validateDatasetAdvanced(
    datasetId: string
  ): Promise<DatasetValidationResult> {
    const response = await apiClient.post(
      `/fine-tuning/datasets/${datasetId}/validate/advanced`
    )
    return response.data
  },

  async preprocessDataset(
    datasetId: string,
    options: any
  ): Promise<{ processed_dataset_id: string; stats: any }> {
    const response = await apiClient.post(
      `/fine-tuning/datasets/${datasetId}/preprocess`,
      options
    )
    return response.data
  },

  async analyzeDatasetQuality(datasetId: string): Promise<any> {
    const response = await apiClient.post(
      `/fine-tuning/datasets/${datasetId}/analyze/quality`
    )
    return response.data
  },

  // 配置模板管理
  async getAdvancedConfigTemplates(): Promise<{
    templates: ConfigurationTemplate[]
  }> {
    const response = await apiClient.get(
      '/fine-tuning/configs/templates/advanced'
    )
    return response.data
  },

  async createCustomTemplate(
    template: Omit<ConfigurationTemplate, 'name'>
  ): Promise<ConfigurationTemplate> {
    const response = await apiClient.post(
      '/fine-tuning/configs/templates/custom',
      template
    )
    return response.data
  },

  async applyConfigTemplate(
    templateName: string,
    overrides?: any
  ): Promise<any> {
    const response = await apiClient.post(
      `/fine-tuning/configs/templates/${templateName}/apply`,
      overrides
    )
    return response.data
  },

  // 资源监控
  async getResourceMonitoring(): Promise<ResourceMonitoring> {
    const response = await apiClient.get('/fine-tuning/resources/monitoring')
    return response.data
  },

  async setResourceAlerts(config: any): Promise<{ message: string }> {
    const response = await apiClient.post(
      '/fine-tuning/resources/alerts',
      config
    )
    return response.data
  },

  async getResourceRecommendations(jobConfig: any): Promise<any> {
    const response = await apiClient.post(
      '/fine-tuning/resources/recommendations',
      jobConfig
    )
    return response.data
  },

  // 批量任务管理
  async createBatchJob(batchConfig: any): Promise<BatchJobManagement> {
    const response = await apiClient.post(
      '/fine-tuning/batch/create',
      batchConfig
    )
    return response.data
  },

  async getBatchJobs(): Promise<BatchJobManagement[]> {
    const response = await apiClient.get('/fine-tuning/batch/list')
    return response.data
  },

  async getBatchJobStatus(batchId: string): Promise<BatchJobManagement> {
    const response = await apiClient.get(`/fine-tuning/batch/${batchId}`)
    return response.data
  },

  async cancelBatchJob(batchId: string): Promise<{ message: string }> {
    const response = await apiClient.post(
      `/fine-tuning/batch/${batchId}/cancel`
    )
    return response.data
  },

  // 模型导出和部署
  async exportTrainedModel(
    jobId: string,
    format = 'huggingface'
  ): Promise<{ download_url: string; expires_at: string }> {
    const response = await apiClient.post(`/fine-tuning/jobs/${jobId}/export`, {
      format,
    })
    return response.data
  },

  async prepareModelDeployment(
    jobId: string,
    deploymentConfig: any
  ): Promise<any> {
    const response = await apiClient.post(
      `/fine-tuning/jobs/${jobId}/deploy/prepare`,
      deploymentConfig
    )
    return response.data
  },

  // 实验记录和版本管理
  async createExperiment(
    experimentConfig: any
  ): Promise<{ experiment_id: string }> {
    const response = await apiClient.post(
      '/fine-tuning/experiments/create',
      experimentConfig
    )
    return response.data
  },

  async getExperimentHistory(experimentId: string): Promise<any> {
    const response = await apiClient.get(
      `/fine-tuning/experiments/${experimentId}/history`
    )
    return response.data
  },

  async tagExperimentVersion(
    experimentId: string,
    version: string,
    tag: string
  ): Promise<{ message: string }> {
    const response = await apiClient.post(
      `/fine-tuning/experiments/${experimentId}/versions/${version}/tag`,
      { tag }
    )
    return response.data
  },

  // 性能基准测试
  async runPerformanceBenchmark(
    modelId: string,
    testSuite: string
  ): Promise<any> {
    const response = await apiClient.post(
      `/fine-tuning/models/${modelId}/benchmark`,
      { test_suite: testSuite }
    )
    return response.data
  },

  async getBenchmarkResults(benchmarkId: string): Promise<any> {
    const response = await apiClient.get(
      `/fine-tuning/benchmarks/${benchmarkId}/results`
    )
    return response.data
  },

  async compareModelPerformance(modelIds: string[]): Promise<any> {
    const response = await apiClient.post(
      '/fine-tuning/models/compare/performance',
      { model_ids: modelIds }
    )
    return response.data
  },
}

export default fineTuningServiceEnhanced
