/**
 * 安全API服务
 */

import { apiClient } from './apiClient';

export interface SecurityStats {
  total_requests: number;
  blocked_requests: number;
  active_threats: number;
  api_keys_count: number;
  high_risk_events: number;
  medium_risk_events: number;
  low_risk_events: number;
}

export interface SecurityAlert {
  id: string;
  alert_type: string;
  severity: 'low' | 'medium' | 'high' | 'critical';
  description: string;
  affected_resource: string;
  source_ip: string;
  user_id?: string;
  timestamp: string;
  status: 'active' | 'investigating' | 'resolved' | 'false_positive';
  auto_blocked: boolean;
  action_taken: string[];
  resolution_time?: string;
}

export interface APIKey {
  id: string;
  name: string;
  key: string;
  created_at: string;
  last_used_at?: string;
  expires_at?: string;
  permissions: string[];
  rate_limits: {
    requests_per_minute: number;
    requests_per_hour: number;
  };
  status: 'active' | 'expired' | 'revoked';
}

export interface ToolPermission {
  tool_name: string;
  description: string;
  category: string;
  enabled: boolean;
  whitelist_only: boolean;
  requires_approval: boolean;
  risk_level: 'low' | 'medium' | 'high' | 'critical';
  allowed_roles: string[];
  usage_count: number;
  last_used?: string;
}

export interface ToolWhitelist {
  tool_name: string;
  users: string[];
  roles: string[];
}

class SecurityApi {
  // 安全统计
  async getSecurityStats(): Promise<SecurityStats> {
    const response = await apiClient.get('/api/v1/security/metrics');
    return response.data || {
      total_requests: 0,
      blocked_requests: 0,
      active_threats: 0,
      api_keys_count: 0,
      audit_logs_count: 0,
      permission_rules_count: 0,
      whitelist_entries_count: 0
    };
  }

  async getSecurityConfig(): Promise<any> {
    const response = await apiClient.get('/api/v1/security/config');
    return response.data;
  }

  async updateSecurityConfig(config: any): Promise<void> {
    await apiClient.put('/api/v1/security/config', config);
  }

  // 安全告警
  async getSecurityAlerts(): Promise<SecurityAlert[]> {
    const response = await apiClient.get('/api/v1/security/alerts');
    return response.data;
  }

  async resolveAlert(alertId: string): Promise<void> {
    await apiClient.post(`/api/v1/security/alerts/${alertId}/resolve`);
  }

  async updateAlertStatus(alertId: string, status: string): Promise<void> {
    await apiClient.put(`/api/v1/security/alerts/${alertId}`, { status });
  }

  // API密钥管理
  async getAPIKeys(): Promise<APIKey[]> {
    const response = await apiClient.get('/api/v1/security/api-keys');
    return response.data;
  }

  async createAPIKey(data: {
    name: string;
    permissions: string[];
    expires_in_days: number;
  }): Promise<{ key: string; id: string }> {
    const response = await apiClient.post('/api/v1/security/api-keys', data);
    return response.data;
  }

  async revokeAPIKey(keyId: string): Promise<void> {
    await apiClient.delete(`/api/v1/security/api-keys/${keyId}`);
  }

  // MCP工具权限
  async getToolPermissions(): Promise<ToolPermission[]> {
    const response = await apiClient.get('/api/v1/security/mcp-tools/permissions');
    return response.data;
  }

  async updateToolPermission(toolName: string, updates: Partial<ToolPermission>): Promise<void> {
    await apiClient.put(`/api/v1/security/mcp-tools/permissions/${toolName}`, updates);
  }

  async getToolWhitelist(): Promise<ToolWhitelist[]> {
    const response = await apiClient.get('/api/v1/security/mcp-tools/whitelist');
    return response.data;
  }

  async updateToolWhitelist(toolName: string, whitelist: { users: string[]; roles: string[] }): Promise<void> {
    await apiClient.post('/api/v1/security/mcp-tools/whitelist', {
      tool_name: toolName,
      ...whitelist
    });
  }

  // 审计日志
  async getAuditLogs(params?: {
    start_time?: string;
    end_time?: string;
    user_id?: string;
    tool_name?: string;
  }): Promise<any[]> {
    const response = await apiClient.get('/api/v1/security/mcp-tools/audit', { params });
    return response.data;
  }

  // 风险评估
  async getRiskAssessment(): Promise<any> {
    const response = await apiClient.get('/api/v1/security/risk-assessment');
    return response.data;
  }
}

export const securityApi = new SecurityApi();