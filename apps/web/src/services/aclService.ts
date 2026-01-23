import apiClient from './apiClient'

export interface ACLRule {
  id: string
  name: string
  description: string
  source: string
  target: string
  action: 'allow' | 'deny'
  conditions: string[]
  priority: number
  status: 'active' | 'inactive'
  created_at: string
  updated_at: string
}

export interface CreateACLRuleRequest {
  name: string
  description: string
  source: string
  target: string
  action: 'allow' | 'deny'
  conditions: string[]
  priority: number
}

export interface UpdateACLRuleRequest {
  name?: string
  description?: string
  source?: string
  target?: string
  action?: 'allow' | 'deny'
  conditions?: string[]
  priority?: number
  status?: 'active' | 'inactive'
}

export interface SecurityMetrics {
  total_rules: number
  active_rules: number
  blocked_requests: number
  allowed_requests: number
  violation_count: number
  last_violation_time: string | null
}

export interface ValidationResult {
  is_valid: boolean
  errors: string[]
  warnings: string[]
}

export interface TestRuleRequest {
  source: string
  target: string
  action: string
  conditions?: string[]
}

class ACLService {
  private baseUrl = '/acl'

  async listRules(): Promise<ACLRule[]> {
    const response = await apiClient.get(`${this.baseUrl}/rules`)
    return response.data
  }

  async getRule(ruleId: string): Promise<ACLRule> {
    const response = await apiClient.get(`${this.baseUrl}/rules/${ruleId}`)
    return response.data
  }

  async createRule(data: CreateACLRuleRequest): Promise<ACLRule> {
    const response = await apiClient.post(`${this.baseUrl}/rules`, data)
    return response.data
  }

  async updateRule(
    ruleId: string,
    data: UpdateACLRuleRequest
  ): Promise<ACLRule> {
    const response = await apiClient.put(
      `${this.baseUrl}/rules/${ruleId}`,
      data
    )
    return response.data
  }

  async deleteRule(ruleId: string): Promise<void> {
    await apiClient.delete(`${this.baseUrl}/rules/${ruleId}`)
  }

  async validateRule(data: CreateACLRuleRequest): Promise<ValidationResult> {
    const response = await apiClient.post(`${this.baseUrl}/validate`, data)
    return response.data
  }

  async getSecurityMetrics(): Promise<SecurityMetrics> {
    const response = await apiClient.get(`${this.baseUrl}/metrics`)
    return response.data
  }

  async testRule(
    ruleId: string,
    testData: { source: string; target: string; context?: any }
  ): Promise<{
    allowed: boolean
    matched_rule: ACLRule | null
    reason: string
  }> {
    const response = await apiClient.post(
      `${this.baseUrl}/rules/${ruleId}/test`,
      testData
    )
    return response.data
  }

  async exportRules(format: 'json' | 'yaml' | 'xml' = 'json'): Promise<Blob> {
    const response = await apiClient.get(`${this.baseUrl}/export`, {
      params: { format },
      responseType: 'blob',
    })
    return response.data
  }

  async importRules(
    file: File
  ): Promise<{ imported: number; failed: number; errors: string[] }> {
    const formData = new FormData()
    formData.append('file', file)
    const response = await apiClient.post(`${this.baseUrl}/import`, formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    })
    return response.data
  }
}

export const aclService = new ACLService()
