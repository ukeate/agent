import apiClient from './apiClient'

export interface GraphValidationViolation {
  rule_id: string
  rule_type: string
  message: string
  count: number
  details?: any[]
}

export interface GraphValidationResult {
  valid: boolean
  violations: GraphValidationViolation[]
  checked_rules: number
  execution_time_ms: number
}

class KnowledgeManagementService {
  private baseUrl = '/kg'

  async validateGraph(rules?: any[]): Promise<GraphValidationResult> {
    const response = await apiClient.post(`${this.baseUrl}/validate`, rules || null)
    return response.data
  }
}

export const knowledgeManagementService = new KnowledgeManagementService()
