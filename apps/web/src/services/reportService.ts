/**
 * 报告服务
 */
import apiClient from './apiClient'

// 报告格式
export type ReportFormat = 'pdf' | 'html' | 'markdown' | 'json'

// 报告数据
export interface ReportData {
  experiment: any
  summary: {
    status: string
    startDate: Date
    endDate?: Date
    duration: number
    totalUsers: number
    totalEvents: number
    recommendation: string
  }
  metrics: {
    primary: MetricResult[]
    secondary: MetricResult[]
    guardrail: GuardrailResult[]
  }
  statistics: StatisticsData
  variants: VariantData[]
  segments: SegmentData[]
  timeline: TimelineData[]
  insights: string[]
  recommendations: string[]
}

// 指标结果
export interface MetricResult {
  name: string
  control: number
  treatment: number
  improvement: number
  pValue: number
  ci_lower: number
  ci_upper: number
  significant: boolean
}

// 护栏指标结果
export interface GuardrailResult {
  name: string
  control: number
  treatment: number
  change: number
  threshold: number
  violated: boolean
}

// 统计数据
export interface StatisticsData {
  confidence: number
  power: number
  sampleSize: {
    required: number
    achieved: number
  }
  srm: {
    passed: boolean
    pValue: number
  }
  multipleTestingCorrection?: string
}

// 变体数据
export interface VariantData {
  id: string
  name: string
  users: number
  conversion: number
  revenue: number
  improvement?: number
}

// 分段数据
export interface SegmentData {
  segment: string
  control: number
  treatment: number
  improvement: number
  significant: boolean
}

// 时间线数据
export interface TimelineData {
  date: string
  control: number
  treatment: number
  cumulative_control: number
  cumulative_treatment: number
}

// 报告参数
export interface GenerateReportParams {
  includeRawData?: boolean
  includeSegments?: boolean
  includeTimeline?: boolean
  confidenceLevel?: number
}

class ReportService {
  private baseUrl = '/report-generation'

  /**
   * 生成实验报告
   */
  async generateReport(
    experimentId: string,
    params: GenerateReportParams = {}
  ): Promise<ReportData> {
    const response = await apiClient.post(`${this.baseUrl}/generate`, {
      experiment_id: experimentId,
      ...params,
    })
    return this.parseReportData(response.data)
  }

  /**
   * 导出报告
   */
  async exportReport(
    experimentId: string,
    format: ReportFormat = 'pdf'
  ): Promise<Blob> {
    const response = await apiClient.post(
      `${this.baseUrl}/export`,
      {
        experiment_id: experimentId,
        format,
      },
      { responseType: 'blob' }
    )
    return response.data
  }

  /**
   * 创建分享链接
   */
  async createShareLink(
    experimentId: string,
    expiresIn: number = 7 * 24 * 60 * 60 // 默认7天
  ): Promise<string> {
    const response = await apiClient.post(`${this.baseUrl}/share`, {
      experiment_id: experimentId,
      expires_in: expiresIn,
    })
    return response.data.share_url
  }

  /**
   * 获取报告模板
   */
  async getTemplates(): Promise<any[]> {
    const response = await apiClient.get(`${this.baseUrl}/templates`)
    return response.data.templates
  }

  /**
   * 使用模板生成报告
   */
  async generateFromTemplate(
    experimentId: string,
    templateId: string,
    customizations?: any
  ): Promise<ReportData> {
    const response = await apiClient.post(`${this.baseUrl}/from-template`, {
      experiment_id: experimentId,
      template_id: templateId,
      customizations,
    })
    return this.parseReportData(response.data)
  }

  /**
   * 获取报告历史
   */
  async getReportHistory(experimentId: string): Promise<any[]> {
    const response = await apiClient.get(`${this.baseUrl}/history`, {
      params: { experiment_id: experimentId },
    })
    return response.data.reports
  }

  /**
   * 安排定期报告
   */
  async scheduleReport(
    experimentId: string,
    schedule: {
      frequency: 'daily' | 'weekly' | 'monthly'
      recipients: string[]
      format: ReportFormat
    }
  ): Promise<void> {
    await apiClient.post(`${this.baseUrl}/schedule`, {
      experiment_id: experimentId,
      ...schedule,
    })
  }

  /**
   * 获取实验洞察
   */
  async getInsights(experimentId: string): Promise<string[]> {
    const response = await apiClient.get(`${this.baseUrl}/insights`, {
      params: { experiment_id: experimentId },
    })
    return response.data.insights
  }

  /**
   * 获取建议
   */
  async getRecommendations(experimentId: string): Promise<string[]> {
    const response = await apiClient.get(`${this.baseUrl}/recommendations`, {
      params: { experiment_id: experimentId },
    })
    return response.data.recommendations
  }

  /**
   * 比较多个实验
   */
  async compareExperiments(experimentIds: string[]): Promise<any> {
    const response = await apiClient.post(`${this.baseUrl}/compare`, {
      experiment_ids: experimentIds,
    })
    return response.data
  }

  /**
   * 解析报告数据
   */
  private parseReportData(data: any): ReportData {
    return {
      ...data,
      summary: {
        ...data.summary,
        startDate: new Date(data.summary.startDate),
        endDate: data.summary.endDate
          ? new Date(data.summary.endDate)
          : undefined,
      },
    }
  }

  /**
   * 生成执行摘要
   */
  async generateExecutiveSummary(experimentId: string): Promise<string> {
    const response = await apiClient.post(`${this.baseUrl}/executive-summary`, {
      experiment_id: experimentId,
    })
    return response.data.summary
  }

  /**
   * 获取报告状态
   */
  async getReportStatus(reportId: string): Promise<{
    status: 'pending' | 'generating' | 'completed' | 'failed'
    progress?: number
    error?: string
  }> {
    const response = await apiClient.get(`${this.baseUrl}/status/${reportId}`)
    return response.data
  }

  /**
   * 取消报告生成
   */
  async cancelReport(reportId: string): Promise<void> {
    await apiClient.post(`${this.baseUrl}/cancel/${reportId}`)
  }

  /**
   * 验证报告数据
   */
  async validateReportData(experimentId: string): Promise<{
    valid: true
    issues?: string[]
  }> {
    const response = await apiClient.post(`${this.baseUrl}/validate`, {
      experiment_id: experimentId,
    })
    return response.data
  }

  // 新增：未使用的API方法 - 根据api-ui.md

  /**
   * 生成模型评估报告 - POST /model-evaluation/generate-report
   */
  async generateModelEvaluationReport(evaluationId: string): Promise<any> {
    const response = await apiClient.post(`/model-evaluation/generate-report`, {
      evaluation_id: evaluationId,
    })
    return response.data
  }

  /**
   * 获取实验报告 - GET /experiments/{experiment_id}/report
   */
  async getExperimentReport(experimentId: string): Promise<any> {
    const response = await apiClient.get(`/experiments/${experimentId}/report`)
    return response.data
  }

  /**
   * 获取知识图谱质量报告 - GET /knowledge-graph/quality/report
   */
  async getKnowledgeGraphQualityReport(): Promise<any> {
    const response = await apiClient.get(`/knowledge-graph/quality/report`)
    return response.data
  }

  /**
   * 获取安全合规报告 - GET /security/compliance-report
   */
  async getSecurityComplianceReport(): Promise<any> {
    const response = await apiClient.get(`/security/compliance-report`)
    return response.data
  }

  /**
   * 获取情感智能质量报告 - GET /emotion-intelligence/quality-report
   */
  async getEmotionIntelligenceQualityReport(): Promise<any> {
    const response = await apiClient.get(`/emotion-intelligence/quality-report`)
    return response.data
  }

  /**
   * 获取平台监控报告 - GET /platform/monitoring/report
   */
  async getPlatformMonitoringReport(): Promise<any> {
    const response = await apiClient.get(`/platform/monitoring/report`)
    return response.data
  }

  /**
   * 获取平台优化报告 - GET /platform/optimization/report
   */
  async getPlatformOptimizationReport(): Promise<any> {
    const response = await apiClient.get(`/platform/optimization/report`)
    return response.data
  }

  /**
   * 获取PGVector性能报告 - GET /pgvector/monitoring/performance-report
   */
  async getPGVectorPerformanceReport(): Promise<any> {
    const response = await apiClient.get(
      `/pgvector/monitoring/performance-report`
    )
    return response.data
  }

  /**
   * 获取模型压缩结果报告 - GET /model-compression/results/{job_id}/report
   */
  async getModelCompressionReport(jobId: string): Promise<any> {
    const response = await apiClient.get(
      `/model-compression/results/${jobId}/report`
    )
    return response.data
  }

  /**
   * 获取集成测试报告 - GET /testing/integration/reports
   */
  async getIntegrationTestReports(): Promise<any> {
    const response = await apiClient.get(`/testing/integration/reports`)
    return response.data
  }

  /**
   * 获取训练数据标注任务质量报告 - GET /training-data/annotation-tasks/{task_id}/quality-report
   */
  async getAnnotationTaskQualityReport(taskId: string): Promise<any> {
    const response = await apiClient.get(
      `/training-data/annotation-tasks/${taskId}/quality-report`
    )
    return response.data
  }

  /**
   * 获取社交情感合规报告 - GET /social-emotion/compliance/report
   */
  async getSocialEmotionComplianceReport(): Promise<any> {
    const response = await apiClient.get(`/social-emotion/compliance/report`)
    return response.data
  }

  /**
   * 生成汇总报告 - POST /reports/generate-summary
   */
  async generateSummaryReport(params: any): Promise<any> {
    const response = await apiClient.post(`/reports/generate-summary`, params)
    return response.data
  }

  /**
   * 批量生成报告 - POST /reports/batch-generate
   */
  async batchGenerateReports(reportRequests: any[]): Promise<any> {
    const response = await apiClient.post(`/reports/batch-generate`, {
      reports: reportRequests,
    })
    return response.data
  }

  /**
   * 预览实验报告 - GET /reports/preview/{experiment_id}
   */
  async previewExperimentReport(experimentId: string): Promise<any> {
    const response = await apiClient.get(`/reports/preview/${experimentId}`)
    return response.data
  }

  /**
   * 比较实验报告 - POST /reports/compare
   */
  async compareExperimentReports(experimentIds: string[]): Promise<any> {
    const response = await apiClient.post(`/reports/compare`, {
      experiment_ids: experimentIds,
    })
    return response.data
  }

  /**
   * 创建报告模板 - POST /reports/templates
   */
  async createReportTemplate(template: any): Promise<any> {
    const response = await apiClient.post(`/reports/templates`, template)
    return response.data
  }

  /**
   * 获取报告健康状态 - GET /reports/health
   */
  async getReportHealth(): Promise<any> {
    const response = await apiClient.get(`/reports/health`)
    return response.data
  }

  /**
   * 安排报告生成 - POST /reports/schedule
   */
  async scheduleReportGeneration(scheduleParams: any): Promise<any> {
    const response = await apiClient.post(`/reports/schedule`, scheduleParams)
    return response.data
  }

  /**
   * 导出实验报告 - GET /reports/export/{experiment_id}
   */
  async exportExperimentReport(
    experimentId: string,
    format?: string
  ): Promise<Blob> {
    const response = await apiClient.get(`/reports/export/${experimentId}`, {
      params: { format },
      responseType: 'blob',
    })
    return response.data
  }
}

export const reportService = new ReportService()
