import { apiClient } from './apiClient'

export enum CorrectionMethod {
  BONFERRONI = 'bonferroni',
  HOLM = 'holm',
  HOCHBERG = 'hochberg',
  BENJAMINI_HOCHBERG = 'benjamini_hochberg',
  BENJAMINI_YEKUTIELI = 'benjamini_yekutieli',
}

export interface CorrectionRequest {
  pvalues: number[]
  method: CorrectionMethod
  alpha: number
}

export class MultipleTestingCorrectionService {
  private baseUrl = '/multiple-testing'

  async correctPValues(request: CorrectionRequest): Promise<any> {
    const response = await apiClient.post(`${this.baseUrl}/correct`, request)
    return response.data
  }

  async compareMethods(pvalues: number[], alpha: number = 0.05): Promise<any> {
    const response = await apiClient.post(`${this.baseUrl}/compare`, {
      pvalues,
      alpha,
    })
    return response.data
  }

  async getMethodRecommendation(request: any): Promise<any> {
    const response = await apiClient.post(
      `${this.baseUrl}/recommendation`,
      request
    )
    return response.data
  }

  async adjustPowerForMultipleTesting(request: any): Promise<any> {
    const response = await apiClient.post(
      `${this.baseUrl}/power-adjustment`,
      request
    )
    return response.data
  }

  async analyzeABTestMultipleComparisons(request: any): Promise<any> {
    const response = await apiClient.post(
      `${this.baseUrl}/ab-test-multiple`,
      request
    )
    return response.data
  }
}

export const multipleTestingCorrectionService =
  new MultipleTestingCorrectionService()
