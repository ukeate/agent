/**
 * 安全API服务
 */

import apiClient from './apiClient'

export interface SecurityStats {
  total_requests: number
  blocked_requests: number
  active_threats: number
  api_keys_count: number
  high_risk_events: number
  medium_risk_events: number
  low_risk_events: number
}

export interface SecurityAlert {
  id: string
  alert_type: string
  severity: 'low' | 'medium' | 'high' | 'critical'
  description: string
  affected_resource: string
  source_ip: string
  user_id?: string
  timestamp: string
  status: 'active' | 'investigating' | 'resolved' | 'false_positive'
  auto_blocked: boolean
  action_taken: string[]
  resolution_time?: string
}

export interface APIKey {
  id: string
  name: string
  key: string
  created_at: string
  last_used_at?: string
  expires_at?: string
  permissions: string[]
  rate_limits: {
    requests_per_minute: number
    requests_per_hour: number
  }
  status: 'active' | 'expired' | 'revoked'
}

export interface ToolPermission {
  tool_name: string
  description: string
  category: string
  enabled: boolean
  whitelist_only: boolean
  requires_approval: boolean
  risk_level: 'low' | 'medium' | 'high' | 'critical'
  allowed_roles: string[]
  usage_count: number
  last_used?: string
}

export interface ToolWhitelist {
  tool_name: string
  users: string[]
  roles: string[]
}

class SecurityApi {
  // 安全统计
  async getSecurityStats(): Promise<SecurityStats> {
    const response = await apiClient.get('/security/metrics')
    const metrics = response.data?.security_metrics || {}
    const criticalAlerts = metrics.critical_alerts ?? 0
    const alertsLastHour = metrics.alerts_last_hour ?? 0
    return {
      total_requests: metrics.total_requests_last_hour ?? 0,
      blocked_requests: metrics.blocked_ips ?? 0,
      active_threats: metrics.active_alerts ?? 0,
      api_keys_count: response.data?.api_keys_count ?? 0,
      high_risk_events: criticalAlerts,
      medium_risk_events: Math.max(alertsLastHour - criticalAlerts, 0),
      low_risk_events: 0,
    }
  }

  async getSecurityConfig(): Promise<any> {
    const response = await apiClient.get('/security/config')
    return response.data
  }

  async updateSecurityConfig(config: any): Promise<void> {
    await apiClient.put('/security/config', config)
  }

  // 安全告警
  async getSecurityAlerts(): Promise<SecurityAlert[]> {
    const response = await apiClient.get('/security/alerts')
    return response.data?.alerts || []
  }

  async resolveAlert(alertId: string): Promise<void> {
    await apiClient.post(`/security/alerts/${alertId}/resolve`)
  }

  async updateAlertStatus(alertId: string, status: string): Promise<void> {
    await apiClient.put(`/security/alerts/${alertId}`, { status })
  }

  // API密钥管理
  async getAPIKeys(): Promise<APIKey[]> {
    const response = await apiClient.get('/security/api-keys')
    return response.data?.api_keys || []
  }

  async createAPIKey(data: {
    name: string
    permissions: string[]
    expires_in_days: number
  }): Promise<{ key: string; id: string }> {
    const response = await apiClient.post('/security/api-keys', data)
    return response.data
  }

  async revokeAPIKey(keyId: string): Promise<void> {
    await apiClient.delete(`/security/api-keys/${keyId}`)
  }

  // MCP工具权限
  async getToolPermissions(): Promise<ToolPermission[]> {
    const response = await apiClient.get('/security/mcp-tools/permissions')
    const permissions = response.data?.permissions || {}
    return Object.entries(permissions).map(([toolName, perm]) => ({
      tool_name: toolName,
      ...(perm as ToolPermission),
    }))
  }

  async updateToolPermission(
    toolName: string,
    updates: Partial<ToolPermission>
  ): Promise<void> {
    await apiClient.put('/security/mcp-tools/permissions', updates, {
      params: { tool_name: toolName },
    })
  }

  async getToolWhitelist(): Promise<ToolWhitelist[]> {
    const response = await apiClient.get('/security/mcp-tools/whitelist')
    const whitelist =
      response.data?.whitelist || response.data?.current_whitelist || []
    return whitelist.map((toolName: string) => ({
      tool_name: toolName,
      users: [],
      roles: [],
    }))
  }

  async updateToolWhitelist(
    toolName: string,
    whitelist: { users: string[]; roles: string[] }
  ): Promise<void> {
    const action =
      whitelist.users.length || whitelist.roles.length ? 'add' : 'remove'
    await apiClient.post('/security/mcp-tools/whitelist', [toolName], {
      params: { action },
    })
  }

  // 审计日志
  async getAuditLogs(params?: {
    start_time?: string
    end_time?: string
    user_id?: string
    tool_name?: string
  }): Promise<any[]> {
    const response = await apiClient.get('/security/mcp-tools/audit', {
      params,
    })
    return response.data?.logs || []
  }

  // 风险评估
  async getRiskAssessment(): Promise<any> {
    const response = await apiClient.get('/security/risk-assessment')
    return response.data
  }

  // 合规报告
  async getComplianceReport(
    startDate: string,
    endDate: string,
    reportType?: string
  ): Promise<any> {
    const params = {
      start_date: startDate,
      end_date: endDate,
      report_type: reportType,
    }
    const response = await apiClient.get('/security/compliance-report', {
      params,
    })
    return response.data
  }

  // MCP工具待审批请求
  async getPendingApprovals(): Promise<any[]> {
    const response = await apiClient.get(
      '/security/mcp-tools/pending-approvals'
    )
    return response.data?.pending_approvals || []
  }

  // 审批工具调用
  async approveToolCall(
    requestId: string,
    approved: boolean,
    reason?: string
  ): Promise<void> {
    await apiClient.post(`/security/mcp-tools/approve/${requestId}`, {
      approved,
      reason,
    })
  }
}

export const securityApi = new SecurityApi()
