import { apiClient } from './apiClient'

export interface LoRAConfig {
  rank: number
  alpha: number
  dropout: number
  target_modules?: string[]
  bias: string
}

export interface QuantizationConfig {
  quantization_type: string
  bits: number
  use_double_quant: boolean
  quant_type: string
  compute_dtype: string
}

export interface TrainingJobRequest {
  job_name: string
  model_name: string
  model_architecture?: string
  training_mode: string
  dataset_path: string
  output_dir?: string
  learning_rate: number
  num_train_epochs: number
  per_device_train_batch_size: number
  gradient_accumulation_steps: number
  warmup_steps: number
  max_seq_length: number
  lora_config?: LoRAConfig
  quantization_config?: QuantizationConfig
  use_distributed: boolean
  use_deepspeed: boolean
  use_flash_attention: boolean
  use_gradient_checkpointing: boolean
  fp16: boolean
  bf16: boolean
}

export interface TrainingJob {
  job_id: string
  job_name: string
  status: string
  created_at: string
  started_at?: string
  completed_at?: string
  progress: number
  current_epoch: number
  total_epochs: number
  current_loss?: number
  best_loss?: number
  error_message?: string
  config?: Record<string, any>
}

export interface ModelInfo {
  models: Array<{
    models: string[]
    architecture: string
    max_seq_length: number
  }>
  architectures: string[]
}

export interface ValidationResult {
  valid: boolean
  warnings: string[]
  model_info: {
    architecture: string
    max_seq_length: number
    target_modules: string[]
  }
  hardware_info: any
  recommendations: any
}

export interface ConfigTemplate {
  templates: Record<string, any>
}

export interface Dataset {
  filename: string
  path: string
  size: number
  created_at: string
}

export const fineTuningService = {
  // 任务管理
  async createTrainingJob(config: TrainingJobRequest): Promise<TrainingJob> {
    const response = await apiClient.post('/fine-tuning/jobs', config)
    return response.data
  },

  async getTrainingJobs(
    status?: string,
    limit = 50,
    offset = 0
  ): Promise<TrainingJob[]> {
    const params = new URLSearchParams()
    if (status) params.append('status', status)
    params.append('limit', limit.toString())
    params.append('offset', offset.toString())

    const response = await apiClient.get(`/fine-tuning/jobs?${params}`)
    return response.data
  },

  async getTrainingJob(jobId: string): Promise<TrainingJob> {
    const response = await apiClient.get(`/fine-tuning/jobs/${jobId}`)
    return response.data
  },

  async cancelTrainingJob(jobId: string): Promise<{ message: string }> {
    const response = await apiClient.put(`/fine-tuning/jobs/${jobId}/cancel`)
    return response.data
  },

  async deleteTrainingJob(jobId: string): Promise<{ message: string }> {
    const response = await apiClient.delete(`/fine-tuning/jobs/${jobId}`)
    return response.data
  },

  async pauseTrainingJob(jobId: string): Promise<{ message: string }> {
    const response = await apiClient.post(`/fine-tuning/jobs/${jobId}/pause`)
    return response.data
  },

  async resumeTrainingJob(jobId: string): Promise<{ message: string }> {
    const response = await apiClient.post(`/fine-tuning/jobs/${jobId}/resume`)
    return response.data
  },

  // 监控和日志
  async getTrainingLogs(
    jobId: string,
    lines = 100
  ): Promise<{ logs: string[] }> {
    const response = await apiClient.get(
      `/fine-tuning/jobs/${jobId}/logs?lines=${lines}`
    )
    return response.data
  },

  async getTrainingMetrics(jobId: string): Promise<{ metrics: any }> {
    const response = await apiClient.get(`/fine-tuning/jobs/${jobId}/metrics`)
    return response.data
  },

  async getTrainingProgress(jobId: string): Promise<any> {
    const response = await apiClient.get(`/fine-tuning/jobs/${jobId}/progress`)
    return response.data
  },

  // 模型和配置管理
  async getSupportedModels(): Promise<ModelInfo> {
    const response = await apiClient.get('/fine-tuning/models')
    return response.data
  },

  async validateModelConfig(
    config: TrainingJobRequest
  ): Promise<ValidationResult> {
    const response = await apiClient.post(
      '/fine-tuning/models/validate',
      config
    )
    return response.data
  },

  async getConfigTemplates(): Promise<ConfigTemplate> {
    const response = await apiClient.get('/fine-tuning/configs/templates')
    return response.data
  },

  async validateTrainingConfig(
    config: TrainingJobRequest
  ): Promise<{ valid: boolean; errors: string[] }> {
    const response = await apiClient.post(
      '/fine-tuning/configs/validate',
      config
    )
    return response.data
  },

  // 数据集管理
  async uploadDataset(file: File, name: string): Promise<any> {
    const formData = new FormData()
    formData.append('file', file)
    formData.append('name', name)

    const response = await apiClient.post('/fine-tuning/datasets', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    })
    return response.data
  },

  async getDatasets(): Promise<{ datasets: Dataset[] }> {
    const response = await apiClient.get('/fine-tuning/datasets')
    return response.data
  },

  async getDatasetInfo(datasetId: string): Promise<any> {
    const response = await apiClient.get(`/fine-tuning/datasets/${datasetId}`)
    return response.data
  },

  async validateDatasetFormat(datasetId: string): Promise<any> {
    const response = await apiClient.post(
      `/fine-tuning/datasets/${datasetId}/validate`
    )
    return response.data
  },
}

export default fineTuningService
