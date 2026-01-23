/**
 * 实验增强服务
 * 提供实验搜索、分析、监控等高级功能，全部依赖真实API。
 */
import apiClient from './apiClient'

// 实验元数据接口
export interface ExperimentMetadata {
  id: string
  name: string
  description: string
  status: 'draft' | 'running' | 'paused' | 'completed' | 'failed'
  type: string
  startDate?: Date
  endDate?: Date
  owner: string
  tags: string[]
  sampleSize: {
    current: number
    required: number
  }
  variants: Array<{
    id: string
    name: string
    traffic: number
    isControl: boolean
  }>
  metrics: Array<{
    id: string
    name: string
    type: 'conversion' | 'revenue' | 'engagement'
    target: number
  }>
}

// 实验分析数据接口
export interface ExperimentAnalysis {
  experimentId: string
  status: 'analyzing' | 'ready' | 'inconclusive'
  confidence: number
  significance: number
  statisticalPower: number
  variants: Array<{
    id: string
    name: string
    sampleSize: number
    conversionRate: number
    lift: number
    pValue: number
    confidenceInterval: [number, number]
  }>
  metrics: Array<{
    id: string
    name: string
    primaryVariant: string
    winner?: string
    significance: number
    effect: number
    revenue?: number
  }>
  recommendations: Array<{
    type: 'continue' | 'stop' | 'extend' | 'modify'
    reason: string
    action: string
    priority: 'high' | 'medium' | 'low'
  }>
  timeline: Array<{
    date: string
    event: string
    description: string
  }>
}

// 搜索参数接口
export interface SearchParams {
  filters?: {
    status?: string[]
    type?: string[]
    owner?: string[]
    tags?: string[]
    dateRange?: [Date, Date]
  }
  sort?: {
    field: string
    order: 'asc' | 'desc'
  }
  pagination?: {
    page: number
    limit: number
  }
}

class ExperimentServiceEnhanced {
  private baseUrl = '/experiments'

  async searchExperiments(params: SearchParams): Promise<{
    experiments: ExperimentMetadata[]
    total: number
    page: number
  }> {
    const response = await apiClient.post(`${this.baseUrl}/search`, params)
    return response.data
  }

  async getExperimentAnalysis(
    experimentId: string
  ): Promise<ExperimentAnalysis> {
    const response = await apiClient.get(
      `${this.baseUrl}/${experimentId}/analysis`
    )
    return response.data
  }

  async getTemplates(): Promise<any[]> {
    const response = await apiClient.get(`${this.baseUrl}/templates`)
    return response.data
  }

  async getMonitoring(experimentId: string): Promise<any> {
    const response = await apiClient.get(
      `${this.baseUrl}/${experimentId}/monitoring`
    )
    return response.data
  }

  async getCostAnalysis(experimentId: string): Promise<any> {
    const response = await apiClient.get(
      `${this.baseUrl}/${experimentId}/cost-analysis`
    )
    return response.data
  }

  async getAuditLog(experimentId: string): Promise<any[]> {
    const response = await apiClient.get(
      `${this.baseUrl}/${experimentId}/audit`
    )
    return response.data
  }

  async addComment(
    experimentId: string,
    comment: { text: string; type: string }
  ): Promise<void> {
    await apiClient.post(`${this.baseUrl}/${experimentId}/comments`, comment)
  }

  async updateSettings(experimentId: string, settings: any): Promise<void> {
    await apiClient.put(`${this.baseUrl}/${experimentId}/settings`, settings)
  }

  async exportData(
    experimentId: string,
    format: 'csv' | 'xlsx' | 'json'
  ): Promise<Blob> {
    const response = await apiClient.get(
      `${this.baseUrl}/${experimentId}/export`,
      {
        params: { format },
        responseType: 'blob',
      }
    )
    return response.data
  }

  async shareExperiment(experimentId: string, users: string[]): Promise<void> {
    await apiClient.post(`${this.baseUrl}/${experimentId}/share`, { users })
  }

  async cloneExperiment(
    experimentId: string,
    name: string
  ): Promise<ExperimentMetadata> {
    const response = await apiClient.post(
      `${this.baseUrl}/${experimentId}/clone`,
      { name }
    )
    return response.data
  }
}

export const experimentServiceEnhanced = new ExperimentServiceEnhanced()
