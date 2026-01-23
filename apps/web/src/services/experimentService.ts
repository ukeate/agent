/**
 * 实验服务
 */
import apiClient from './apiClient'

// 实验配置接口
export interface ExperimentConfig {
  name: string
  description: string
  type: string
  status: string
  startDate?: Date
  endDate?: Date
  variants: any[]
  metrics: any[]
  targetingRules: any[]
  sampleSize?: number
  confidenceLevel: number
  mutuallyExclusiveGroup?: string
  layer?: string
  tags: string[]
  enableDataQualityChecks: boolean
  enableAutoStop: boolean
  autoStopThreshold?: number
}

// 列表参数
export interface ListExperimentsParams {
  search?: string
  status?: string
  type?: string
  owner?: string
  startDateFrom?: string
  startDateTo?: string
  tags?: string[]
  sortBy?: string
  sortOrder?: string
  page?: number
  pageSize?: number
}

// 实验数据
export interface ExperimentData {
  id: string
  name: string
  description: string
  type: string
  status: string
  startDate?: Date | string
  endDate?: Date | string
  variants: any[]
  metrics: any[]
  sampleSize?: {
    current: number
    required: number
  }
  participants?: number
  conversion_rate?: number
  lift?: number
  confidenceLevel?: number
  created_at?: string
  updated_at?: string
  owner?: string
  owners?: string[]
  tags?: string[]
  healthStatus?: string
  healthMessage?: string
}

// 列表响应
export interface ListExperimentsResponse {
  experiments: ExperimentData[]
  total: number
  page: number
  pageSize: number
}

class ExperimentService {
  private baseUrl = '/experiments'

  /**
   * 获取实验列表
   */
  async listExperiments(
    params: ListExperimentsParams = {}
  ): Promise<ListExperimentsResponse> {
    const response = await apiClient.get<ListExperimentsResponse>(
      this.baseUrl,
      { params }
    )
    return response.data
  }

  /**
   * 获取实验模板 - 使用实际API
   */
  async getExperimentTemplates(): Promise<any[]> {
    const response = await apiClient.get(`${this.baseUrl}/templates`)
    return response.data
  }

  /**
   * 计算样本量 - 使用实际API
   */
  async calculateSampleSize(params: {
    baselineRate: number
    minimumDetectableEffect: number
    confidenceLevel: number
    power: number
  }): Promise<any> {
    const response = await apiClient.post(
      `${this.baseUrl}/calculate-sample-size`,
      params
    )
    return response.data
  }

  /**
   * 获取实验详情
   */
  async getExperiment(id: string): Promise<ExperimentData> {
    const response = await apiClient.get(`${this.baseUrl}/${id}`)
    return response.data
  }

  /**
   * 创建实验
   */
  async createExperiment(config: ExperimentConfig): Promise<ExperimentData> {
    const response = await apiClient.post(this.baseUrl, config)
    return response.data
  }

  /**
   * 更新实验
   */
  async updateExperiment(
    id: string,
    config: Partial<ExperimentConfig>
  ): Promise<ExperimentData> {
    const response = await apiClient.put(`${this.baseUrl}/${id}`, config)
    return response.data
  }

  /**
   * 删除实验
   */
  async deleteExperiment(id: string): Promise<void> {
    await apiClient.delete(`${this.baseUrl}/${id}`)
  }

  /**
   * 启动实验
   */
  async startExperiment(id: string): Promise<void> {
    await apiClient.post(`${this.baseUrl}/${id}/start`)
  }

  /**
   * 暂停实验
   */
  async pauseExperiment(id: string): Promise<void> {
    await apiClient.post(`${this.baseUrl}/${id}/pause`)
  }

  /**
   * 恢复实验
   */
  async resumeExperiment(id: string): Promise<void> {
    await apiClient.post(`${this.baseUrl}/${id}/resume`)
  }

  /**
   * 停止实验
   */
  async stopExperiment(id: string): Promise<void> {
    await apiClient.post(`${this.baseUrl}/${id}/stop`)
  }

  /**
   * 克隆实验
   */
  async cloneExperiment(id: string): Promise<ExperimentData> {
    const response = await apiClient.post(`${this.baseUrl}/${id}/clone`)
    return response.data
  }

  /**
   * 归档实验
   */
  async archiveExperiment(id: string): Promise<void> {
    await apiClient.post(`${this.baseUrl}/${id}/archive`)
  }

  /**
   * 导出实验
   */
  async exportExperiments(ids: string[]): Promise<Blob> {
    const response = await apiClient.post(
      `${this.baseUrl}/export`,
      { ids },
      { responseType: 'blob' }
    )
    return response.data
  }

  /**
   * 导入实验
   */
  async importExperiments(
    file: File
  ): Promise<{ imported: number; failed: number }> {
    const formData = new FormData()
    formData.append('file', file)
    const response = await apiClient.post(`${this.baseUrl}/import`, formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    })
    return response.data
  }

  /**
   * 获取实验报告
   */
  async getExperimentReport(id: string): Promise<any> {
    const response = await apiClient.get(`${this.baseUrl}/${id}/report`)
    return response.data
  }

  /**
   * 获取实验指标
   */
  async getExperimentMetrics(id: string): Promise<any> {
    const response = await apiClient.get(`${this.baseUrl}/${id}/metrics`)
    return response.data
  }

  /**
   * 获取实验事件
   */
  async getExperimentEvents(
    id: string,
    params?: { limit?: number }
  ): Promise<any> {
    const response = await apiClient.get(`${this.baseUrl}/${id}/events`, {
      params,
    })
    return response.data
  }

  /**
   * 验证实验配置
   */
  async validateConfig(
    config: ExperimentConfig
  ): Promise<{ valid: boolean; errors?: string[] }> {
    const response = await apiClient.post(`${this.baseUrl}/validate`, config)
    return response.data
  }

  /**
   * 获取实验模板
   */
  async getTemplates(): Promise<any[]> {
    const response = await apiClient.get(`${this.baseUrl}/templates`)
    return response.data
  }

  /**
   * 从模板创建实验
   */
  async createFromTemplate(
    templateId: string,
    overrides?: Partial<ExperimentConfig>
  ): Promise<ExperimentData> {
    const response = await apiClient.post(`${this.baseUrl}/from-template`, {
      templateId,
      overrides,
    })
    return response.data
  }
}

export const experimentService = new ExperimentService()
