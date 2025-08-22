/**
 * 报告服务
 */
import { apiClient } from './apiClient';

// 报告格式
export type ReportFormat = 'pdf' | 'html' | 'markdown' | 'json';

// 报告数据
export interface ReportData {
  experiment: any;
  summary: {
    status: string;
    startDate: Date;
    endDate?: Date;
    duration: number;
    totalUsers: number;
    totalEvents: number;
    recommendation: string;
  };
  metrics: {
    primary: MetricResult[];
    secondary: MetricResult[];
    guardrail: GuardrailResult[];
  };
  statistics: StatisticsData;
  variants: VariantData[];
  segments: SegmentData[];
  timeline: TimelineData[];
  insights: string[];
  recommendations: string[];
}

// 指标结果
export interface MetricResult {
  name: string;
  control: number;
  treatment: number;
  improvement: number;
  pValue: number;
  ci_lower: number;
  ci_upper: number;
  significant: boolean;
}

// 护栏指标结果
export interface GuardrailResult {
  name: string;
  control: number;
  treatment: number;
  change: number;
  threshold: number;
  violated: boolean;
}

// 统计数据
export interface StatisticsData {
  confidence: number;
  power: number;
  sampleSize: {
    required: number;
    achieved: number;
  };
  srm: {
    passed: boolean;
    pValue: number;
  };
  multipleTestingCorrection?: string;
}

// 变体数据
export interface VariantData {
  id: string;
  name: string;
  users: number;
  conversion: number;
  revenue: number;
  improvement?: number;
}

// 分段数据
export interface SegmentData {
  segment: string;
  control: number;
  treatment: number;
  improvement: number;
  significant: boolean;
}

// 时间线数据
export interface TimelineData {
  date: string;
  control: number;
  treatment: number;
  cumulative_control: number;
  cumulative_treatment: number;
}

// 报告参数
export interface GenerateReportParams {
  includeRawData?: boolean;
  includeSegments?: boolean;
  includeTimeline?: boolean;
  confidenceLevel?: number;
}

class ReportService {
  private baseUrl = '/api/v1/report-generation';

  /**
   * 生成实验报告
   */
  async generateReport(
    experimentId: string,
    params: GenerateReportParams = {}
  ): Promise<ReportData> {
    const response = await apiClient.post(`${this.baseUrl}/generate`, {
      experiment_id: experimentId,
      ...params
    });
    return this.parseReportData(response.data);
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
        format
      },
      { responseType: 'blob' }
    );
    return response.data;
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
      expires_in: expiresIn
    });
    return response.data.share_url;
  }

  /**
   * 获取报告模板
   */
  async getTemplates(): Promise<any[]> {
    const response = await apiClient.get(`${this.baseUrl}/templates`);
    return response.data.templates;
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
      customizations
    });
    return this.parseReportData(response.data);
  }

  /**
   * 获取报告历史
   */
  async getReportHistory(experimentId: string): Promise<any[]> {
    const response = await apiClient.get(`${this.baseUrl}/history`, {
      params: { experiment_id: experimentId }
    });
    return response.data.reports;
  }

  /**
   * 安排定期报告
   */
  async scheduleReport(
    experimentId: string,
    schedule: {
      frequency: 'daily' | 'weekly' | 'monthly';
      recipients: string[];
      format: ReportFormat;
    }
  ): Promise<void> {
    await apiClient.post(`${this.baseUrl}/schedule`, {
      experiment_id: experimentId,
      ...schedule
    });
  }

  /**
   * 获取实验洞察
   */
  async getInsights(experimentId: string): Promise<string[]> {
    const response = await apiClient.get(
      `${this.baseUrl}/insights`,
      { params: { experiment_id: experimentId } }
    );
    return response.data.insights;
  }

  /**
   * 获取建议
   */
  async getRecommendations(experimentId: string): Promise<string[]> {
    const response = await apiClient.get(
      `${this.baseUrl}/recommendations`,
      { params: { experiment_id: experimentId } }
    );
    return response.data.recommendations;
  }

  /**
   * 比较多个实验
   */
  async compareExperiments(experimentIds: string[]): Promise<any> {
    const response = await apiClient.post(`${this.baseUrl}/compare`, {
      experiment_ids: experimentIds
    });
    return response.data;
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
        endDate: data.summary.endDate ? new Date(data.summary.endDate) : undefined
      }
    };
  }

  /**
   * 生成执行摘要
   */
  async generateExecutiveSummary(experimentId: string): Promise<string> {
    const response = await apiClient.post(`${this.baseUrl}/executive-summary`, {
      experiment_id: experimentId
    });
    return response.data.summary;
  }

  /**
   * 获取报告状态
   */
  async getReportStatus(reportId: string): Promise<{
    status: 'pending' | 'generating' | 'completed' | 'failed';
    progress?: number;
    error?: string;
  }> {
    const response = await apiClient.get(`${this.baseUrl}/status/${reportId}`);
    return response.data;
  }

  /**
   * 取消报告生成
   */
  async cancelReport(reportId: string): Promise<void> {
    await apiClient.post(`${this.baseUrl}/cancel/${reportId}`);
  }

  /**
   * 验证报告数据
   */
  async validateReportData(experimentId: string): Promise<{
    valid: boolean;
    issues?: string[];
  }> {
    const response = await apiClient.post(`${this.baseUrl}/validate`, {
      experiment_id: experimentId
    });
    return response.data;
  }
}

export const reportService = new ReportService();