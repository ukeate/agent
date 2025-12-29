import apiClient from './apiClient'

export interface ReasoningRule {
  id: string
  name: string
  rule_text: string
  status: string
  confidence: number
  priority: number
  execution_count?: number
  success_count?: number
  success_rate?: number
  created_at?: string
  updated_at?: string
}

class KgReasoningService {
  private baseUrl = '/kg-reasoning'

  async listRules(status?: string): Promise<ReasoningRule[]> {
    const params = status ? { status } : {}
    const response = await apiClient.get(`${this.baseUrl}/rules`, { params })
    return response.data.rules || []
  }
}

export const kgReasoningService = new KgReasoningService()
