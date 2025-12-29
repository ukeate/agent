import apiClient from './apiClient';

import { logger } from '../utils/logger'
// ========== Statistical Analysis Types ==========
export interface DescriptiveStats {
  mean: number;
  median: number;
  std_dev: number;
  variance: number;
  min: number;
  max: number;
  count: number;
  percentiles: {
    p25: number;
    p50: number;
    p75: number;
    p95: number;
    p99: number;
  };
}

export interface CorrelationResult {
  correlation: number;
  p_value: number;
  confidence_interval: [number, number];
  method: string;
}

export interface RegressionResult {
  coefficients: Record<string, number>;
  intercept: number;
  r_squared: number;
  p_values: Record<string, number>;
  predictions?: number[];
  residuals?: number[];
}

// ========== Hypothesis Testing Types ==========
export enum TestType {
  T_TEST = 't_test',
  CHI_SQUARE = 'chi_square',
  ANOVA = 'anova',
  MANN_WHITNEY = 'mann_whitney',
  FISHER_EXACT = 'fisher_exact'
}

export interface HypothesisTestResult {
  test_type: TestType;
  statistic: number;
  p_value: number;
  confidence_level: number;
  reject_null: boolean;
  effect_size?: number;
  confidence_interval?: [number, number];
  sample_sizes: {
    control: number;
    treatment: number;
  };
  means?: {
    control: number;
    treatment: number;
  };
}

export interface MultipleTestingCorrection {
  original_p_values: number[];
  corrected_p_values: number[];
  method: string;
  alpha: number;
  rejected: boolean[];
}

// ========== Power Analysis Types ==========
export interface PowerAnalysisResult {
  power: number;
  sample_size?: number;
  effect_size?: number;
  alpha: number;
  alternative: string;
  test_type: string;
  recommendations: string[];
}

export interface SampleSizeCalculation {
  required_sample_size: number;
  per_variant: number;
  total_users: number;
  duration_days: number;
  confidence: number;
}

export interface MinimumDetectableEffect {
  mde: number;
  baseline_rate: number;
  relative_change: number;
  absolute_change: number;
}

// ========== Service Class ==========
class StatisticalAnalysisService {
  private statsUrl = '/statistical-analysis';
  private hypothesisUrl = '/hypothesis-testing';
  private powerUrl = '/power-analysis';

  // ========== Statistical Analysis Methods ==========
  
  async getDescriptiveStats(data: number[]): Promise<DescriptiveStats> {
    try {
      const response = await apiClient.post(`${this.statsUrl}/descriptive`, { data });
      return response.data.stats;
    } catch (error) {
      logger.error('获取描述性统计失败:', error);
      throw error;
    }
  }

  async calculateCorrelation(x: number[], y: number[], method: string = 'pearson'): Promise<CorrelationResult> {
    try {
      const response = await apiClient.post(`${this.statsUrl}/correlation`, { x, y, method });
      return response.data.result;
    } catch (error) {
      logger.error('计算相关性失败:', error);
      throw error;
    }
  }

  async performRegression(x: number[][], y: number[]): Promise<RegressionResult> {
    try {
      const response = await apiClient.post(`${this.statsUrl}/regression`, { x, y });
      return response.data.result;
    } catch (error) {
      logger.error('执行回归分析失败:', error);
      throw error;
    }
  }

  async detectOutliers(data: number[], method: string = 'iqr'): Promise<{
    outliers: number[];
    indices: number[];
    cleaned_data: number[];
  }> {
    try {
      const response = await apiClient.post(`${this.statsUrl}/outliers`, { data, method });
      return response.data;
    } catch (error) {
      logger.error('检测异常值失败:', error);
      throw error;
    }
  }

  async performTimeSeries(data: number[], periods: number = 7): Promise<{
    trend: number[];
    seasonal: number[];
    residual: number[];
    forecast: number[];
  }> {
    try {
      const response = await apiClient.post(`${this.statsUrl}/time-series`, { data, periods });
      return response.data;
    } catch (error) {
      logger.error('时间序列分析失败:', error);
      throw error;
    }
  }

  // ========== Hypothesis Testing Methods ==========
  
  async performHypothesisTest(
    controlData: number[],
    treatmentData: number[],
    testType: TestType,
    confidenceLevel: number = 0.95
  ): Promise<HypothesisTestResult> {
    try {
      const response = await apiClient.post(`${this.hypothesisUrl}/test`, {
        control_data: controlData,
        treatment_data: treatmentData,
        test_type: testType,
        confidence_level: confidenceLevel
      });
      return response.data.result;
    } catch (error) {
      logger.error('假设检验失败:', error);
      throw error;
    }
  }

  async performMultipleTestingCorrection(
    pValues: number[],
    method: string = 'bonferroni',
    alpha: number = 0.05
  ): Promise<MultipleTestingCorrection> {
    try {
      const response = await apiClient.post(`${this.hypothesisUrl}/multiple-testing`, {
        p_values: pValues,
        method,
        alpha
      });
      return response.data.result;
    } catch (error) {
      logger.error('多重检验校正失败:', error);
      throw error;
    }
  }

  async performSequentialTesting(
    experimentId: string,
    metric: string,
    alpha: number = 0.05
  ): Promise<{
    should_stop: boolean;
    p_value: number;
    z_score: number;
    boundaries: {
      upper: number;
      lower: number;
    };
  }> {
    try {
      const response = await apiClient.post(`${this.hypothesisUrl}/sequential`, {
        experiment_id: experimentId,
        metric,
        alpha
      });
      return response.data;
    } catch (error) {
      logger.error('序贯检验失败:', error);
      throw error;
    }
  }

  async calculateConfidenceInterval(
    data: number[],
    confidenceLevel: number = 0.95
  ): Promise<[number, number]> {
    try {
      const response = await apiClient.post(`${this.hypothesisUrl}/confidence-interval`, {
        data,
        confidence_level: confidenceLevel
      });
      return response.data.interval;
    } catch (error) {
      logger.error('计算置信区间失败:', error);
      throw error;
    }
  }

  // ========== Power Analysis Methods ==========
  
  async calculatePower(
    sampleSize: number,
    effectSize: number,
    alpha: number = 0.05,
    testType: string = 't_test'
  ): Promise<PowerAnalysisResult> {
    try {
      const response = await apiClient.post(`${this.powerUrl}/calculate`, {
        sample_size: sampleSize,
        effect_size: effectSize,
        alpha,
        test_type: testType
      });
      return response.data.result;
    } catch (error) {
      logger.error('功效计算失败:', error);
      throw error;
    }
  }

  async calculateSampleSize(
    effectSize: number,
    power: number = 0.8,
    alpha: number = 0.05,
    testType: string = 't_test'
  ): Promise<SampleSizeCalculation> {
    try {
      const response = await apiClient.post(`${this.powerUrl}/sample-size`, {
        effect_size: effectSize,
        power,
        alpha,
        test_type: testType
      });
      return response.data.result;
    } catch (error) {
      logger.error('样本量计算失败:', error);
      throw error;
    }
  }

  async calculateMDE(
    sampleSize: number,
    baselineRate: number,
    power: number = 0.8,
    alpha: number = 0.05
  ): Promise<MinimumDetectableEffect> {
    try {
      const response = await apiClient.post(`${this.powerUrl}/mde`, {
        sample_size: sampleSize,
        baseline_rate: baselineRate,
        power,
        alpha
      });
      return response.data.result;
    } catch (error) {
      logger.error('MDE计算失败:', error);
      throw error;
    }
  }

  async performPowerSimulation(
    experimentConfig: {
      baseline_rate: number;
      expected_lift: number;
      daily_traffic: number;
      variants: number;
    }
  ): Promise<{
    power_curve: Array<{ days: number; power: number }>;
    recommended_duration: number;
    risk_analysis: {
      type_i_error: number;
      type_ii_error: number;
      false_positive_rate: number;
      false_negative_rate: number;
    };
  }> {
    try {
      const response = await apiClient.post(`${this.powerUrl}/simulate`, experimentConfig);
      return response.data;
    } catch (error) {
      logger.error('功效模拟失败:', error);
      throw error;
    }
  }

  async getOptimalAllocation(
    totalSampleSize: number,
    numVariants: number,
    priorData?: number[][]
  ): Promise<{
    allocation: number[];
    efficiency_gain: number;
    method: string;
  }> {
    try {
      const response = await apiClient.post(`${this.powerUrl}/optimal-allocation`, {
        total_sample_size: totalSampleSize,
        num_variants: numVariants,
        prior_data: priorData
      });
      return response.data;
    } catch (error) {
      logger.error('优化分配失败:', error);
      throw error;
    }
  }

  // ========== Experiment Analysis Methods ==========
  
  async analyzeExperiment(experimentId: string): Promise<{
    summary: {
      start_date: string;
      duration_days: number;
      total_users: number;
      variants: string[];
    };
    metrics: Record<string, {
      control: DescriptiveStats;
      treatment: DescriptiveStats;
      test_result: HypothesisTestResult;
      power_analysis: PowerAnalysisResult;
    }>;
    recommendations: string[];
  }> {
    try {
      const response = await apiClient.get(`${this.statsUrl}/experiment/${experimentId}/analysis`);
      return response.data;
    } catch (error) {
      logger.error('实验分析失败:', error);
      throw error;
    }
  }

  async getMetricTrends(
    experimentId: string,
    metric: string
  ): Promise<{
    dates: string[];
    control: number[];
    treatment: number[];
    cumulative_effect: number[];
    confidence_bands: {
      lower: number[];
      upper: number[];
    };
  }> {
    try {
      const response = await apiClient.get(`${this.statsUrl}/experiment/${experimentId}/trends`, {
        params: { metric }
      });
      return response.data;
    } catch (error) {
      logger.error('获取指标趋势失败:', error);
      throw error;
    }
  }

  async validateSRM(experimentId: string): Promise<{
    is_valid: boolean;
    chi_square_statistic: number;
    p_value: number;
    expected_ratio: number[];
    observed_ratio: number[];
    message: string;
  }> {
    try {
      const response = await apiClient.post(`${this.hypothesisUrl}/srm-check`, {
        experiment_id: experimentId
      });
      return response.data;
    } catch (error) {
      logger.error('SRM验证失败:', error);
      throw error;
    }
  }
}

// 导出服务实例
export const statisticalAnalysisService = new StatisticalAnalysisService();
export default statisticalAnalysisService;