import { apiClient } from './apiClient';

export enum TestType {
  ONE_SAMPLE_T = 'one_sample_t',
  TWO_SAMPLE_T = 'two_sample_t',
  PAIRED_T = 'paired_t',
  ONE_PROPORTION = 'one_proportion',
  TWO_PROPORTIONS = 'two_proportions',
  CHI_SQUARE = 'chi_square',
  ANOVA = 'anova'
}

export enum AlternativeHypothesis {
  TWO_SIDED = 'two_sided',
  GREATER = 'greater',
  LESS = 'less'
}

export interface PowerCalculationRequest {
  test_type: TestType;
  effect_size: number;
  sample_size: number | number[];
  alpha: number;
  alternative: AlternativeHypothesis;
}

export class PowerAnalysisService {
  private baseUrl = '/power-analysis';

  async calculatePower(request: PowerCalculationRequest): Promise<any> {
    const response = await apiClient.post(`${this.baseUrl}/calculate`, request);
    return response.data;
  }

  async calculateSampleSize(request: any): Promise<any> {
    const response = await apiClient.post(`${this.baseUrl}/calculate`, {
      ...request,
      calculate_type: 'sample_size'
    });
    return response.data;
  }

  async calculateEffectSize(request: any): Promise<any> {
    const response = await apiClient.post(`${this.baseUrl}/calculate`, {
      ...request,
      calculate_type: 'effect_size'
    });
    return response.data;
  }

  async calculateABTestSampleSize(request: any): Promise<any> {
    const response = await apiClient.post('/experiments/calculate-sample-size', {
      baselineRate: request.baseline_conversion_rate,
      minimumDetectableEffect: request.minimum_detectable_effect,
      power: request.power,
      confidenceLevel: 1 - request.alpha
    });
    return response.data;
  }

  async getRecommendations(testType: TestType): Promise<any> {
    const response = await apiClient.get(`${this.baseUrl}/recommendations/${testType}`);
    return response.data;
  }
}

export const powerAnalysisService = new PowerAnalysisService();
