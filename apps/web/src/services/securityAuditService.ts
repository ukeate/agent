import apiClient from './apiClient'

export interface AuditLog {
  id: string
  timestamp: string
  event_type: string
  severity: 'info' | 'warning' | 'error' | 'critical'
  user_id: string
  user_name: string
  user_role: string
  ip_address: string
  user_agent: string
  resource: string
  action: string
  result: 'success' | 'failure' | 'blocked'
  details: Record<string, any>
  session_id: string
  correlation_id?: string
}

export interface AuditStatistics {
  total_events: number
  events_by_type: Record<string, number>
  events_by_severity: Record<string, number>
  events_by_result: Record<string, number>
  top_users: Array<{ user_id: string; user_name: string; event_count: number }>
  top_resources: Array<{ resource: string; access_count: number }>
  failed_attempts: number
  blocked_attempts: number
  time_range: {
    start: string
    end: string
  }
}

export interface SecurityIncident {
  id: string
  timestamp: string
  type: string
  severity: 'low' | 'medium' | 'high' | 'critical'
  status: 'open' | 'investigating' | 'resolved' | 'closed'
  affected_resources: string[]
  description: string
  detection_method: string
  response_actions: string[]
  assigned_to?: string
  resolution?: string
  related_audit_logs: string[]
}

export interface AuditFilterParams {
  start_time?: string
  end_time?: string
  event_type?: string
  severity?: string
  user_id?: string
  resource?: string
  result?: string
  page?: number
  page_size?: number
}

class SecurityAuditService {
  private baseUrl = '/security-audit'

  async getAuditLogs(filters?: AuditFilterParams): Promise<{
    logs: AuditLog[]
    total: number
    page: number
    page_size: number
  }> {
    const response = await apiClient.get(`${this.baseUrl}/logs`, {
      params: filters,
    })
    return response.data
  }

  async getAuditLog(logId: string): Promise<AuditLog> {
    const response = await apiClient.get(`${this.baseUrl}/logs/${logId}`)
    return response.data
  }

  async getStatistics(
    startTime?: string,
    endTime?: string
  ): Promise<AuditStatistics> {
    const params = { start_time: startTime, end_time: endTime }
    const response = await apiClient.get(`${this.baseUrl}/statistics`, {
      params,
    })
    return response.data
  }

  async getSecurityIncidents(status?: string): Promise<SecurityIncident[]> {
    const params = status ? { status } : {}
    const response = await apiClient.get(`${this.baseUrl}/incidents`, {
      params,
    })
    return response.data
  }

  async getIncident(incidentId: string): Promise<SecurityIncident> {
    const response = await apiClient.get(
      `${this.baseUrl}/incidents/${incidentId}`
    )
    return response.data
  }

  async updateIncidentStatus(
    incidentId: string,
    status: string,
    resolution?: string
  ): Promise<SecurityIncident> {
    const response = await apiClient.put(
      `${this.baseUrl}/incidents/${incidentId}/status`,
      {
        status,
        resolution,
      }
    )
    return response.data
  }

  async exportAuditLogs(
    format: 'json' | 'csv' | 'pdf',
    filters?: AuditFilterParams
  ): Promise<Blob> {
    const response = await apiClient.get(`${this.baseUrl}/export`, {
      params: { ...filters, format },
      responseType: 'blob',
    })
    return response.data
  }

  async searchAuditLogs(query: string): Promise<AuditLog[]> {
    const response = await apiClient.get(`${this.baseUrl}/search`, {
      params: { q: query },
    })
    return response.data
  }

  async getComplianceReport(
    standard: 'SOC2' | 'ISO27001' | 'GDPR' | 'HIPAA'
  ): Promise<{
    standard: string
    compliance_score: number
    findings: Array<{
      requirement: string
      status: 'compliant' | 'non_compliant' | 'partial'
      evidence: string[]
      recommendations?: string[]
    }>
    generated_at: string
  }> {
    const response = await apiClient.get(
      `${this.baseUrl}/compliance/${standard}`
    )
    return response.data
  }

  async getRiskAssessment(): Promise<{
    overall_risk_level: 'low' | 'medium' | 'high' | 'critical'
    risk_score: number
    risk_factors: Array<{
      factor: string
      level: string
      impact: number
      likelihood: number
      mitigation?: string
    }>
    recommendations: string[]
    assessment_date: string
  }> {
    const response = await apiClient.get(`${this.baseUrl}/risk-assessment`)
    return response.data
  }
}

export const securityAuditService = new SecurityAuditService()
