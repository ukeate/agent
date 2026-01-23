import apiClient from './apiClient'

export interface KnowledgeModel {
  key: string
  name: string
  type: string
  accuracy: number
  speed: number
  size: string
  description?: string
  created_at?: string
  updated_at?: string
  status?: 'active' | 'inactive' | 'testing'
  features?: string[]
  languages?: string[]
}

export interface ComparisonResult {
  model1: string
  model2: string
  metrics: {
    accuracy_diff: number
    speed_diff: number
    size_ratio: number
    feature_overlap: number
  }
  recommendation: string
  details: Record<string, any>
}

export interface ModelBenchmark {
  model_name: string
  dataset: string
  metrics: {
    precision: number
    recall: number
    f1_score: number
    accuracy: number
    latency_ms: number
  }
  timestamp: string
}

class KnowledgeModelService {
  private baseUrl = '/knowledge-models'

  async listModels(type?: string): Promise<KnowledgeModel[]> {
    const params = type ? { type } : {}
    const response = await apiClient.get(`${this.baseUrl}/models`, { params })
    return response.data
  }

  async getModel(modelId: string): Promise<KnowledgeModel> {
    const response = await apiClient.get(`${this.baseUrl}/models/${modelId}`)
    return response.data
  }

  async compareModels(
    model1Id: string,
    model2Id: string
  ): Promise<ComparisonResult> {
    const response = await apiClient.post(`${this.baseUrl}/compare`, {
      model1: model1Id,
      model2: model2Id,
    })
    return response.data
  }

  async getBenchmarks(modelId: string): Promise<ModelBenchmark[]> {
    const response = await apiClient.get(
      `${this.baseUrl}/models/${modelId}/benchmarks`
    )
    return response.data
  }

  async runBenchmark(
    modelId: string,
    dataset: string
  ): Promise<ModelBenchmark> {
    const response = await apiClient.post(
      `${this.baseUrl}/models/${modelId}/benchmark`,
      {
        dataset,
      }
    )
    return response.data
  }

  async getRecommendation(requirements: {
    task_type: string
    accuracy_requirement: number
    speed_requirement: number
    max_size?: string
  }): Promise<KnowledgeModel[]> {
    const response = await apiClient.post(
      `${this.baseUrl}/recommend`,
      requirements
    )
    return response.data
  }

  async exportComparison(
    modelIds: string[],
    format: 'json' | 'csv' | 'pdf' = 'json'
  ): Promise<Blob> {
    const response = await apiClient.post(
      `${this.baseUrl}/export-comparison`,
      { model_ids: modelIds, format },
      { responseType: 'blob' }
    )
    return response.data
  }
}

export const knowledgeModelService = new KnowledgeModelService()
