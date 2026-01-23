import apiClient from './apiClient'

export type ModelFormat =
  | 'pytorch'
  | 'tensorflow'
  | 'onnx'
  | 'jax'
  | 'huggingface'
  | 'mlflow'
  | 'triton'
  | 'custom'
export type ModelType =
  | 'transformer'
  | 'cnn'
  | 'rnn'
  | 'gan'
  | 'vae'
  | 'diffusion'
  | 'rl'
  | 'custom'
export type CompressionType =
  | 'none'
  | 'quantization'
  | 'pruning'
  | 'distillation'
  | 'mixed'

export interface ModelMetadataRequest {
  name: string
  version?: string
  format: ModelFormat
  model_type?: ModelType
  description?: string
  author?: string
  parameters_count?: number
  model_size_mb?: number
  input_shape?: number[]
  output_shape?: number[]
  training_framework?: string
  training_dataset?: string
  training_epochs?: number
  performance_metrics?: Record<string, number>
  compression_type?: CompressionType
  compression_ratio?: number
  original_size_mb?: number
  tags?: string[]
  license?: string
  repository_url?: string
  paper_url?: string
}

export interface ModelMetadata extends ModelMetadataRequest {
  created_at: string
  updated_at: string
  dependencies: string[]
  python_version?: string
  framework_versions?: Record<string, string>
}

export interface ModelEntry {
  metadata: ModelMetadata
  model_path: string
  config_path?: string
  tokenizer_path?: string
  checksum?: string
}

export interface ModelSearchParams {
  name?: string
  version?: string
  format?: ModelFormat
  model_type?: ModelType
  tags?: string[]
  author?: string
  min_performance?: number
  max_size_mb?: number
}

export interface ModelValidationResult {
  is_valid: boolean
  errors: string[]
  warnings: string[]
  metadata?: ModelMetadata
}

export interface ModelComparisonResult {
  model1: ModelMetadata
  model2: ModelMetadata
  differences: {
    field: string
    value1: any
    value2: any
  }[]
  performance_comparison: Record<
    string,
    {
      model1: number
      model2: number
      improvement: number
    }
  >
}

export interface ModelStatistics {
  total_models: number
  total_size_gb: number
  active_deployments: number
  recent_uploads: number
  by_format: Record<string, number>
  by_type: Record<string, number>
}

class ModelRegistryService {
  private baseUrl = '/model-registry'

  // Model Management
  async registerModel(
    metadata: ModelMetadataRequest,
    file?: File
  ): Promise<ModelEntry> {
    const formData = new FormData()
    formData.append('metadata', JSON.stringify(metadata))
    if (file) {
      formData.append('model_file', file)
    }

    const response = await apiClient.post(
      `${this.baseUrl}/register`,
      formData,
      {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      }
    )
    return response.data
  }

  async getModel(name: string, version?: string): Promise<ModelEntry> {
    const params = version ? { version } : {}
    const response = await apiClient.get(`${this.baseUrl}/models/${name}`, {
      params,
    })
    return response.data
  }

  async listModels(params?: ModelSearchParams): Promise<ModelEntry[]> {
    const response = await apiClient.get(`${this.baseUrl}/models`, { params })
    return response.data.models
  }

  async updateModel(
    name: string,
    version: string,
    metadata: Partial<ModelMetadataRequest>
  ): Promise<ModelEntry> {
    const response = await apiClient.put(
      `${this.baseUrl}/models/${name}/${version}`,
      metadata
    )
    return response.data
  }

  async deleteModel(name: string, version: string): Promise<void> {
    await apiClient.delete(`${this.baseUrl}/models/${name}/${version}`)
  }

  // Model Search and Discovery
  async searchModels(query: ModelSearchParams): Promise<ModelEntry[]> {
    const response = await apiClient.post(`${this.baseUrl}/search`, query)
    return response.data.models
  }

  async getModelsByTag(tag: string): Promise<ModelEntry[]> {
    const response = await apiClient.get(`${this.baseUrl}/tags/${tag}`)
    return response.data.models
  }

  async getModelsByAuthor(author: string): Promise<ModelEntry[]> {
    const response = await apiClient.get(`${this.baseUrl}/authors/${author}`)
    return response.data.models
  }

  // Model Versions
  async listVersions(name: string): Promise<string[]> {
    const response = await apiClient.get(
      `${this.baseUrl}/models/${name}/versions`
    )
    return response.data.versions
  }

  async getLatestVersion(name: string): Promise<ModelEntry> {
    const response = await apiClient.get(
      `${this.baseUrl}/models/${name}/latest`
    )
    return response.data
  }

  // Model Validation
  async validateModel(
    name: string,
    version: string
  ): Promise<ModelValidationResult> {
    const response = await apiClient.post(
      `${this.baseUrl}/models/${name}/${version}/validate`
    )
    return response.data
  }

  // Model Comparison
  async compareModels(
    name1: string,
    version1: string,
    name2: string,
    version2: string
  ): Promise<ModelComparisonResult> {
    const response = await apiClient.post(`${this.baseUrl}/compare`, {
      model1: { name: name1, version: version1 },
      model2: { name: name2, version: version2 },
    })
    return response.data
  }

  // Model Download
  async downloadModel(name: string, version: string): Promise<Blob> {
    const response = await apiClient.get(
      `${this.baseUrl}/models/${name}/${version}/download`,
      {
        responseType: 'blob',
      }
    )
    return response.data
  }

  // Model Deployment
  async deployModel(
    name: string,
    version: string,
    deploymentConfig?: {
      target_environment?: 'production' | 'staging' | 'development'
      replicas?: number
      resources?: {
        cpu?: string
        memory?: string
        gpu?: number
      }
    }
  ): Promise<{
    deployment_id: string
    status: string
    endpoint_url?: string
  }> {
    const response = await apiClient.post(
      `${this.baseUrl}/models/${name}/${version}/deploy`,
      deploymentConfig || {}
    )
    return response.data
  }

  // Model Statistics
  async getModelStats(
    name: string,
    version?: string
  ): Promise<{
    download_count: number
    deployment_count: number
    average_rating: number
    last_accessed: string
    usage_trends: Array<{
      date: string
      downloads: number
      deployments: number
    }>
  }> {
    const params = version ? { version } : {}
    const response = await apiClient.get(
      `${this.baseUrl}/models/${name}/stats`,
      { params }
    )
    return response.data
  }

  // Model Tags
  async getAllTags(): Promise<string[]> {
    const response = await apiClient.get(`${this.baseUrl}/tags`)
    return response.data.tags
  }

  async addTags(name: string, version: string, tags: string[]): Promise<void> {
    await apiClient.post(`${this.baseUrl}/models/${name}/${version}/tags`, {
      tags,
    })
  }

  async removeTags(
    name: string,
    version: string,
    tags: string[]
  ): Promise<void> {
    await apiClient.delete(`${this.baseUrl}/models/${name}/${version}/tags`, {
      data: { tags },
    })
  }

  // Registry Statistics
  async getStatistics(): Promise<{
    total_models: number
    total_size_gb: number
    active_deployments: number
    recent_uploads: number
    by_format: Record<string, number>
    by_type: Record<string, number>
  }> {
    const response = await apiClient.get(`${this.baseUrl}/stats`)
    const data = response.data || {}
    return {
      total_models: data.total_models ?? 0,
      total_size_gb: Number(((data.total_size_mb ?? 0) / 1024).toFixed(2)),
      active_deployments: 0,
      recent_uploads: 0,
      by_format: data.formats || {},
      by_type: data.types || {},
    }
  }
}

export const modelRegistryService = new ModelRegistryService()
