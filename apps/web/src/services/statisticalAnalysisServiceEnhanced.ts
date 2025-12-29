import { apiClient } from './apiClient'

// 从现有服务继承类型
export * from './statisticalAnalysisService'
import statisticalAnalysisService from './statisticalAnalysisService'

// 符合实际API端点的接口定义
export enum MetricType {
  NUMERIC = 'numeric',
  CONVERSION = 'conversion'
}

export interface BasicStatsRequest {
  values: number[]
  calculate_advanced?: boolean
}

export interface ConversionStatsRequest {
  conversions: number
  total_users: number
}

export interface GroupData {
  name: string
  values?: number[]
  conversions?: number
  total_users?: number
}

export interface MultipleGroupsStatsRequest {
  groups: Record<string, GroupData>
  metric_type: MetricType
}

export interface PercentileRequest {
  values: number[]
  percentiles: number[]
}

// 响应接口
export interface BasicStatsResponse {
  stats: {
    count: number
    mean: number
    std_dev: number
    variance: number
    min: number
    max: number
    median: number
    mode?: number[]
    skewness?: number
    kurtosis?: number
    quartiles?: {
      q1: number
      q2: number
      q3: number
    }
    range: number
    coefficient_of_variation: number
  }
  message: string
}

export interface ConversionStatsResponse {
  conversion_rate: number
  stats: {
    conversion_rate: number
    confidence_interval: {
      lower: number
      upper: number
      confidence_level: number
    }
    sample_size: number
    standard_error: number
    z_score: number
    statistical_significance: boolean
  }
  message: string
}

export interface MultipleGroupsStatsResponse {
  groups_stats: Record<string, {
    group_name: string
    stats: any
    sample_size: number
    metric_type: string
  }>
  summary: {
    total_groups: number
    successful_groups: number
    metric_type: string
    groups_overview: Record<string, {
      name: string
      sample_size: number
      mean: number
    }>
  }
  message: string
}

export interface PercentileResponse {
  percentiles: Record<string, number>
  message: string
}

export interface QuickSummary {
  count: number
  mean: number
  std_dev: number
  min: number
  max: number
  median: number
  q25: number
  q75: number
  range: number
  message: string
}

export interface VarianceResponse {
  variance: number
  std_deviation: number
  count: number
  sample: boolean
  message: string
}

export interface MeanResponse {
  mean: number
  count: number
  message: string
}

export interface HealthCheckResponse {
  status: 'healthy' | 'unhealthy'
  service: string
  test_calculation?: {
    input: number[]
    mean: number
    expected: number
    passed: boolean
  }
  error?: string
  message: string
}

// 增强的统计分析服务 - 对应实际API端点
export const statisticalAnalysisServiceEnhanced = {
  // 包含所有基础服务方法
  ...statisticalAnalysisService,

  // 符合实际API端点的方法
  
  // 基础统计计算 (/statistical-analysis/basic-stats)
  async calculateBasicStatistics(request: BasicStatsRequest): Promise<BasicStatsResponse> {
    const response = await apiClient.post('/statistical-analysis/basic-stats', request)
    return response.data
  },

  // 转化率统计 (/statistical-analysis/conversion-stats)
  async calculateConversionStatistics(request: ConversionStatsRequest): Promise<ConversionStatsResponse> {
    const response = await apiClient.post('/statistical-analysis/conversion-stats', request)
    return response.data
  },

  // 分位数计算 (/statistical-analysis/percentiles)
  async calculatePercentiles(request: PercentileRequest): Promise<PercentileResponse> {
    const response = await apiClient.post('/statistical-analysis/percentiles', request)
    return response.data
  },

  // 多分组统计 (/statistical-analysis/multiple-groups-stats)
  async calculateMultipleGroupsStatistics(request: MultipleGroupsStatsRequest): Promise<MultipleGroupsStatsResponse> {
    const response = await apiClient.post('/statistical-analysis/multiple-groups-stats', request)
    return response.data
  },

  // 快速均值计算 (/statistical-analysis/mean)
  async calculateMean(values: number[]): Promise<MeanResponse> {
    const params = new URLSearchParams()
    values.forEach(value => params.append('values', value.toString()))
    const response = await apiClient.get(`/statistical-analysis/mean?${params}`)
    return response.data
  },

  // 方差计算 (/statistical-analysis/variance)
  async calculateVariance(values: number[], sample = true): Promise<VarianceResponse> {
    const params = new URLSearchParams()
    values.forEach(value => params.append('values', value.toString()))
    params.append('sample', sample.toString())
    const response = await apiClient.get(`/statistical-analysis/variance?${params}`)
    return response.data
  },

  // 快速摘要 (/statistical-analysis/summary)
  async getQuickSummary(values: number[]): Promise<QuickSummary> {
    const params = new URLSearchParams()
    values.forEach(value => params.append('values', value.toString()))
    const response = await apiClient.get(`/statistical-analysis/summary?${params}`)
    return response.data
  },

  // 健康检查 (/statistical-analysis/health)
  async healthCheck(): Promise<HealthCheckResponse> {
    const response = await apiClient.get('/statistical-analysis/health')
    return response.data
  }
}

export default statisticalAnalysisServiceEnhanced
