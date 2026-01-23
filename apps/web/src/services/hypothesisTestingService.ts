/**
 * 假设检验统计分析API服务
 * 提供t检验、卡方检验、A/B测试比较等统计推断功能
 */

import { apiClient } from './apiClient'

const API_BASE = '/hypothesis-testing'

// 枚举类型
export enum HypothesisType {
  TWO_SIDED = 'two-sided',
  LESS = 'less',
  GREATER = 'greater',
}

export enum MetricType {
  CONVERSION = 'conversion',
  CONTINUOUS = 'continuous',
  COUNT = 'count',
  RATIO = 'ratio',
}

export interface OneSampleTTestRequest {
  sample: number[]
  population_mean: number
  hypothesis_type: HypothesisType
  alpha: number
}

export interface TwoSampleTTestRequest {
  sample1: number[]
  sample2: number[]
  equal_variances: boolean
  hypothesis_type: HypothesisType
  alpha: number
}

export interface PairedTTestRequest {
  sample1: number[]
  sample2: number[]
  hypothesis_type: HypothesisType
  alpha: number
}

export interface ChiSquareGoodnessOfFitRequest {
  observed: number[]
  expected: number[]
  alpha: number
}

export interface ChiSquareIndependenceRequest {
  contingency_table: number[][]
  alpha: number
}

export interface TwoProportionTestRequest {
  successes1: number
  total1: number
  successes2: number
  total2: number
  hypothesis_type: HypothesisType
  alpha: number
}

export interface ABTestComparisonRequest {
  control_group: Record<string, any>
  treatment_group: Record<string, any>
  metric_type: MetricType
  hypothesis_type: HypothesisType
  alpha: number
  equal_variances: boolean
}

export interface HypothesisTestResult {
  test_type: string
  statistic: number
  p_value: number
  critical_value?: number
  degrees_of_freedom?: number
  is_significant: boolean
  effect_size?: number
  confidence_interval?: [number, number]
  alpha: number
  power?: number
}

export interface HypothesisTestResponse {
  result: HypothesisTestResult
  interpretation: Record<string, string>
  recommendations: string[]
  message: string
}

export interface ABTestComparisonResponse {
  test_result: HypothesisTestResult
  control_stats: GroupStats
  treatment_stats: GroupStats
  practical_significance: Record<string, number>
  interpretation: Record<string, string>
  recommendations: string[]
  message: string
}

export interface GroupStats {
  group_id: string
  group_name: string
  stats: {
    count: number
    mean: number
    std: number
    min: number
    max: number
    median?: number
    q25?: number
    q75?: number
  }
  confidence_interval?: [number, number]
  metric_type: string
}

export interface HealthCheckResponse {
  status: string
  service: string
  test_calculation?: {
    test_type: string
    p_value: number
    is_significant: boolean
    passed: boolean
  }
  error?: string
  message: string
}

/**
 * 假设检验服务类 - 适配实际后端API
 */
export class HypothesisTestingService {
  /**
   * 单样本t检验 - 适配到统一API
   */
  async oneSampleTTest(
    request: OneSampleTTestRequest
  ): Promise<HypothesisTestResponse> {
    const response = await apiClient.post(
      `${API_BASE}/t-test/one-sample`,
      request
    )
    return response.data
  }

  /**
   * 双样本t检验 - 适配到统一API
   */
  async twoSampleTTest(
    request: TwoSampleTTestRequest
  ): Promise<HypothesisTestResponse> {
    const response = await apiClient.post(
      `${API_BASE}/t-test/two-sample`,
      request
    )
    return response.data
  }

  /**
   * 配对t检验 - 适配到统一API
   */
  async pairedTTest(
    request: PairedTTestRequest
  ): Promise<HypothesisTestResponse> {
    const response = await apiClient.post(`${API_BASE}/t-test/paired`, request)
    return response.data
  }

  /**
   * 卡方拟合优度检验
   */
  async chiSquareGoodnessOfFit(
    request: ChiSquareGoodnessOfFitRequest
  ): Promise<HypothesisTestResponse> {
    const response = await apiClient.post(
      `${API_BASE}/chi-square/goodness-of-fit`,
      request
    )
    return response.data
  }

  /**
   * 卡方独立性检验
   */
  async chiSquareIndependence(
    request: ChiSquareIndependenceRequest
  ): Promise<HypothesisTestResponse> {
    const response = await apiClient.post(
      `${API_BASE}/chi-square/independence`,
      request
    )
    return response.data
  }

  /**
   * 两比例检验
   */
  async twoProportionTest(
    request: TwoProportionTestRequest
  ): Promise<HypothesisTestResponse> {
    const response = await apiClient.post(
      `${API_BASE}/proportion-test`,
      request
    )
    return response.data
  }

  /**
   * A/B测试比较分析
   */
  async abTestComparison(
    request: ABTestComparisonRequest
  ): Promise<ABTestComparisonResponse> {
    const response = await apiClient.post(
      `${API_BASE}/ab-test-comparison`,
      request
    )
    return response.data
  }

  /**
   * 服务健康检查
   */
  async healthCheck(): Promise<HealthCheckResponse> {
    const response = await apiClient.get(`${API_BASE}/health`)
    return response.data
  }

  // 辅助方法
  /**
   * 创建单样本t检验请求
   */
  createOneSampleTTestRequest(
    sample: number[],
    populationMean: number,
    hypothesisType: HypothesisType = HypothesisType.TWO_SIDED,
    alpha: number = 0.05
  ): OneSampleTTestRequest {
    return {
      sample,
      population_mean: populationMean,
      hypothesis_type: hypothesisType,
      alpha,
    }
  }

  /**
   * 创建双样本t检验请求
   */
  createTwoSampleTTestRequest(
    sample1: number[],
    sample2: number[],
    equalVariances: boolean = true,
    hypothesisType: HypothesisType = HypothesisType.TWO_SIDED,
    alpha: number = 0.05
  ): TwoSampleTTestRequest {
    return {
      sample1,
      sample2,
      equal_variances: equalVariances,
      hypothesis_type: hypothesisType,
      alpha,
    }
  }

  /**
   * 创建A/B测试请求（转化率比较）
   */
  createConversionABTestRequest(
    controlConversions: number,
    controlTotal: number,
    treatmentConversions: number,
    treatmentTotal: number,
    hypothesisType: HypothesisType = HypothesisType.TWO_SIDED,
    alpha: number = 0.05
  ): ABTestComparisonRequest {
    return {
      control_group: {
        conversions: controlConversions,
        total_users: controlTotal,
      },
      treatment_group: {
        conversions: treatmentConversions,
        total_users: treatmentTotal,
      },
      metric_type: MetricType.CONVERSION,
      hypothesis_type: hypothesisType,
      alpha,
      equal_variances: true,
    }
  }

  /**
   * 创建A/B测试请求（连续指标比较）
   */
  createContinuousABTestRequest(
    controlValues: number[],
    treatmentValues: number[],
    metricType: MetricType,
    equalVariances: boolean = true,
    hypothesisType: HypothesisType = HypothesisType.TWO_SIDED,
    alpha: number = 0.05
  ): ABTestComparisonRequest {
    return {
      control_group: {
        values: controlValues,
      },
      treatment_group: {
        values: treatmentValues,
      },
      metric_type: metricType,
      hypothesis_type: hypothesisType,
      alpha,
      equal_variances: equalVariances,
    }
  }

  /**
   * 解释p值
   */
  interpretPValue(pValue: number): string {
    if (pValue < 0.001) {
      return '极强证据 (p < 0.001)'
    } else if (pValue < 0.01) {
      return '很强证据 (p < 0.01)'
    } else if (pValue < 0.05) {
      return '中等证据 (p < 0.05)'
    } else if (pValue < 0.1) {
      return '较弱证据 (p < 0.1)'
    } else {
      return '证据不足 (p ≥ 0.1)'
    }
  }

  /**
   * 解释效应量
   */
  interpretEffectSize(effectSize: number): string {
    if (effectSize < 0.2) {
      return '很小'
    } else if (effectSize < 0.5) {
      return '小'
    } else if (effectSize < 0.8) {
      return '中等'
    } else {
      return '大'
    }
  }

  /**
   * 计算样本统计信息
   */
  calculateSampleStats(data: number[]): {
    count: number
    mean: number
    std: number
    min: number
    max: number
    median: number
  } {
    const sorted = [...data].sort((a, b) => a - b)
    const count = data.length
    const mean = data.reduce((sum, val) => sum + val, 0) / count
    const variance =
      data.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / (count - 1)
    const std = Math.sqrt(variance)
    const min = sorted[0]
    const max = sorted[count - 1]
    const median =
      count % 2 === 0
        ? (sorted[count / 2 - 1] + sorted[count / 2]) / 2
        : sorted[Math.floor(count / 2)]

    return { count, mean, std, min, max, median }
  }

  /**
   * 生成列联表
   */
  generateContingencyTable(
    group1_category1: number,
    group1_category2: number,
    group2_category1: number,
    group2_category2: number
  ): number[][] {
    return [
      [group1_category1, group1_category2],
      [group2_category1, group2_category2],
    ]
  }

  /**
   * 检查样本量是否足够
   */
  checkSampleSizeAdequacy(sampleSize: number, minSize: number = 30): boolean {
    return sampleSize >= minSize
  }

  /**
   * 估算所需样本量（简单估算）
   */
  estimateRequiredSampleSize(
    effectSize: number,
    alpha: number = 0.05,
    power: number = 0.8
  ): number {
    // 简化的样本量估算公式（仅供参考）
    const zAlpha = alpha === 0.05 ? 1.96 : 2.58 // 近似值
    const zBeta = power === 0.8 ? 0.84 : 1.28 // 近似值

    return Math.ceil(
      (2 * Math.pow(zAlpha + zBeta, 2)) / Math.pow(effectSize, 2)
    )
  }
}

// 导出服务实例
export const hypothesisTestingService = new HypothesisTestingService()
