import apiClient from './apiClient';

// ========== 增强的安全服务，补充8个未覆盖端点功能 ==========

class SecurityEnhancedService {
  private baseUrl = '/security';
  private distributedUrl = '/distributed-security';

  // ========== 安全配置管理 ==========
  
  async updateSecurityConfig(config: {
    authentication?: {
      session_timeout?: number;
      max_failed_attempts?: number;
      password_policy?: {
        min_length?: number;
        require_uppercase?: boolean;
        require_lowercase?: boolean;
        require_numbers?: boolean;
        require_special_chars?: boolean;
      };
    };
    authorization?: {
      default_permissions?: string[];
      role_hierarchy?: Record<string, string[]>;
      resource_access_rules?: Record<string, any>;
    };
    encryption?: {
      algorithm?: string;
      key_rotation_interval?: number;
      data_at_rest?: boolean;
      data_in_transit?: boolean;
    };
    audit?: {
      log_level?: 'basic' | 'detailed' | 'verbose';
      retention_days?: number;
      real_time_alerts?: boolean;
      compliance_standards?: string[];
    };
    mcp_security?: {
      tool_execution_timeout?: number;
      resource_limits?: {
        memory_mb?: number;
        cpu_percent?: number;
        disk_mb?: number;
      };
      sandbox_mode?: boolean;
    };
  }): Promise<{
    success: boolean;
    message: string;
    updated_sections: string[];
    validation_errors?: string[];
    requires_restart?: boolean;
    backup_id?: string;
  }> {
    const response = await apiClient.put(`${this.baseUrl}/config`, config);
    return response.data;
  }

  // ========== MCP工具安全管理 ==========

  async updateMcpToolWhitelist(whitelist: {
    allowed_tools: string[];
    blocked_tools: string[];
    restricted_tools: Array<{
      tool_name: string;
      restrictions: {
        max_executions_per_hour?: number;
        allowed_users?: string[];
        allowed_roles?: string[];
        allowed_time_windows?: Array<{
          start: string;
          end: string;
          days: string[];
        }>;
        require_approval?: boolean;
        monitoring_level?: 'basic' | 'detailed' | 'verbose';
      };
    }>;
    global_settings: {
      default_allow?: boolean;
      logging_enabled?: boolean;
      alert_on_violations?: boolean;
      auto_block_suspicious?: boolean;
    };
  }): Promise<{
    status: 'updated' | 'partial' | 'failed';
    message: string;
    updated_tools: number;
    invalid_tools?: string[];
    warnings?: string[];
    effective_date: string;
  }> {
    const response = await apiClient.post(`${this.baseUrl}/mcp-tools/whitelist`, whitelist);
    return response.data;
  }

  async updateMcpToolPermissions(permissions: {
    tool_permissions: Array<{
      tool_name: string;
      permission_level: 'read' | 'write' | 'execute' | 'admin';
      users: Array<{
        user_id: string;
        permissions: string[];
        expires_at?: string;
        conditions?: {
          ip_whitelist?: string[];
          time_restrictions?: Array<{
            start: string;
            end: string;
            timezone: string;
          }>;
          session_limits?: {
            max_concurrent?: number;
            max_duration_minutes?: number;
          };
        };
      }>;
      roles: Array<{
        role_name: string;
        permissions: string[];
        inherited_from?: string[];
      }>;
    }>;
    audit_settings: {
      log_permission_changes: boolean;
      notify_users_on_change: boolean;
      require_approval_for_elevation: boolean;
    };
  }): Promise<{
    success: boolean;
    updated_permissions: number;
    affected_users: number;
    affected_roles: number;
    validation_errors?: Array<{
      tool_name: string;
      error: string;
      suggestion?: string;
    }>;
    change_summary: {
      added_permissions: number;
      removed_permissions: number;
      modified_permissions: number;
    };
  }> {
    const response = await apiClient.put(`${this.baseUrl}/mcp-tools/permissions`, permissions);
    return response.data;
  }

  // ========== 合规报告生成 ==========

  async generateComplianceReport(params: {
    compliance_standards: Array<'SOC2' | 'ISO27001' | 'GDPR' | 'HIPAA' | 'PCI_DSS' | 'NIST'>;
    report_format: 'json' | 'pdf' | 'html' | 'excel';
    time_range: {
      start_date: string;
      end_date: string;
    };
    include_sections: {
      executive_summary?: boolean;
      control_assessments?: boolean;
      audit_logs?: boolean;
      risk_analysis?: boolean;
      remediation_plans?: boolean;
      compliance_gaps?: boolean;
      evidence_documentation?: boolean;
    };
    filters?: {
      severity_levels?: string[];
      departments?: string[];
      systems?: string[];
    };
    customization?: {
      company_logo?: boolean;
      custom_branding?: boolean;
      additional_notes?: string;
    };
  }): Promise<{
    report_id: string;
    status: 'generating' | 'completed' | 'failed';
    generation_started_at: string;
    estimated_completion_time?: string;
    file_size_mb?: number;
    download_url?: string;
    compliance_scores: Record<string, {
      overall_score: number;
      passing_controls: number;
      failing_controls: number;
      not_applicable: number;
      compliance_percentage: number;
    }>;
    summary: {
      total_controls_assessed: number;
      critical_findings: number;
      high_risk_findings: number;
      medium_risk_findings: number;
      low_risk_findings: number;
      recommendations_count: number;
    };
    next_assessment_due?: string;
  }> {
    const response = await apiClient.get(`${this.baseUrl}/compliance-report`, { params });
    return response.data;
  }

  // ========== 分布式安全策略管理 ==========

  async addSecurityPolicy(policy: {
    policy_name: string;
    description: string;
    policy_type: 'access_control' | 'data_protection' | 'communication' | 'audit' | 'compliance';
    priority: 'low' | 'medium' | 'high' | 'critical';
    scope: {
      agents?: string[];
      users?: string[];
      roles?: string[];
      resources?: string[];
      operations?: string[];
    };
    rules: Array<{
      rule_id: string;
      name: string;
      condition: {
        field: string;
        operator: 'equals' | 'not_equals' | 'contains' | 'not_contains' | 'regex' | 'in' | 'not_in';
        value: any;
        case_sensitive?: boolean;
      };
      action: 'allow' | 'deny' | 'require_approval' | 'log_only' | 'rate_limit';
      parameters?: Record<string, any>;
    }>;
    enforcement: {
      mode: 'enforcing' | 'permissive' | 'disabled';
      effective_date: string;
      expiry_date?: string;
      grace_period_hours?: number;
    };
    notifications: {
      on_violation: boolean;
      on_policy_change: boolean;
      notification_channels: Array<'email' | 'slack' | 'webhook' | 'sms'>;
      recipients: string[];
    };
  }): Promise<{
    policy_id: string;
    status: 'created' | 'validation_failed' | 'conflict_detected';
    message: string;
    validation_errors?: string[];
    conflicting_policies?: Array<{
      policy_id: string;
      policy_name: string;
      conflict_description: string;
    }>;
    effective_from: string;
    estimated_impact: {
      affected_agents: number;
      affected_users: number;
      performance_impact: 'minimal' | 'moderate' | 'significant';
    };
  }> {
    const response = await apiClient.post(`${this.distributedUrl}/policies`, policy);
    return response.data;
  }

  async getSecurityPolicies(params?: {
    policy_type?: string;
    status?: 'active' | 'inactive' | 'expired' | 'pending';
    scope_filter?: {
      agent_id?: string;
      user_id?: string;
      resource?: string;
    };
    search_term?: string;
    sort_by?: 'name' | 'created_date' | 'priority' | 'last_modified';
    sort_order?: 'asc' | 'desc';
    limit?: number;
    offset?: number;
  }): Promise<{
    policies: Array<{
      policy_id: string;
      policy_name: string;
      description: string;
      policy_type: string;
      priority: string;
      status: 'active' | 'inactive' | 'expired' | 'pending';
      created_date: string;
      last_modified: string;
      created_by: string;
      rule_count: number;
      violation_count_last_30_days: number;
      enforcement_mode: string;
      effective_date: string;
      expiry_date?: string;
      compliance_standards: string[];
      tags: string[];
    }>;
    pagination: {
      total: number;
      page: number;
      per_page: number;
      has_more: boolean;
    };
    filters_applied: Record<string, any>;
    summary_stats: {
      total_policies: number;
      active_policies: number;
      expired_policies: number;
      policies_by_type: Record<string, number>;
      policies_by_priority: Record<string, number>;
    };
  }> {
    const response = await apiClient.get(`${this.distributedUrl}/policies`, { params });
    return response.data;
  }

  // ========== 分布式通信加密 ==========

  async encryptCommunication(data: {
    sender_agent_id: string;
    recipient_agent_ids: string[];
    message: {
      content: any;
      message_type: 'command' | 'data' | 'status' | 'heartbeat' | 'alert';
      priority: 'low' | 'normal' | 'high' | 'urgent';
    };
    encryption_settings: {
      algorithm?: 'AES-256' | 'ChaCha20' | 'RSA-4096';
      key_derivation?: 'PBKDF2' | 'Argon2' | 'scrypt';
      compression?: boolean;
      integrity_check?: boolean;
    };
    delivery_options: {
      delivery_method: 'direct' | 'relay' | 'broadcast';
      require_confirmation?: boolean;
      timeout_seconds?: number;
      retry_attempts?: number;
      store_and_forward?: boolean;
    };
    metadata?: {
      session_id?: string;
      correlation_id?: string;
      custom_headers?: Record<string, string>;
    };
  }): Promise<{
    message_id: string;
    encryption_status: 'success' | 'partial' | 'failed';
    encrypted_size_bytes: number;
    original_size_bytes: number;
    compression_ratio?: number;
    delivery_status: Record<string, {
      agent_id: string;
      status: 'delivered' | 'pending' | 'failed' | 'timeout';
      delivery_time?: string;
      error_message?: string;
    }>;
    encryption_metadata: {
      algorithm_used: string;
      key_id: string;
      cipher_mode: string;
      iv: string;
      auth_tag?: string;
    };
    performance_metrics: {
      encryption_time_ms: number;
      transmission_time_ms: number;
      total_time_ms: number;
    };
  }> {
    const response = await apiClient.post(`${this.distributedUrl}/communication/encrypt`, data);
    return response.data;
  }

  // ========== 访问权限撤销管理 ==========

  async revokeAgentAccess(request: {
    target_agent_id: string;
    revocation_scope: {
      revoke_all?: boolean;
      specific_permissions?: string[];
      specific_resources?: string[];
      specific_operations?: string[];
    };
    revocation_reason: {
      reason_code: 'security_breach' | 'policy_violation' | 'role_change' | 'termination' | 'maintenance' | 'other';
      description: string;
      initiated_by: string;
      supporting_evidence?: string[];
    };
    immediate_action: boolean;
    effective_time?: string;
    notification_settings: {
      notify_agent: boolean;
      notify_administrators: boolean;
      notify_affected_users: boolean;
      notification_message?: string;
    };
    cleanup_options: {
      revoke_active_sessions?: boolean;
      clear_cached_permissions?: boolean;
      audit_trail_retention?: boolean;
      backup_agent_state?: boolean;
    };
  }): Promise<{
    revocation_id: string;
    status: 'completed' | 'partial' | 'failed' | 'pending';
    timestamp: string;
    affected_permissions: string[];
    active_sessions_terminated: number;
    cleanup_actions_performed: string[];
    rollback_information?: {
      rollback_id: string;
      rollback_available_until: string;
      rollback_instructions: string;
    };
    impact_assessment: {
      affected_operations: string[];
      dependent_agents: string[];
      estimated_downtime_minutes?: number;
      business_impact: 'minimal' | 'moderate' | 'significant' | 'critical';
    };
    next_steps: string[];
    compliance_notes?: string;
  }> {
    const response = await apiClient.post(`${this.distributedUrl}/revoke-access`, request);
    return response.data;
  }

  // ========== 辅助功能方法 ==========

  async getSecurityConfigSchema(): Promise<{
    schema: Record<string, any>;
    version: string;
    supported_standards: string[];
    default_values: Record<string, any>;
    validation_rules: Record<string, any>;
  }> {
    const response = await apiClient.get(`${this.baseUrl}/config/schema`);
    return response.data;
  }

  async getMcpToolsList(): Promise<{
    available_tools: Array<{
      tool_name: string;
      description: string;
      risk_level: 'low' | 'medium' | 'high' | 'critical';
      required_permissions: string[];
      resource_requirements: Record<string, any>;
      security_notes: string[];
    }>;
    total_tools: number;
    last_updated: string;
  }> {
    const response = await apiClient.get(`${this.baseUrl}/mcp-tools/list`);
    return response.data;
  }

  async validateSecurityPolicy(policy: any): Promise<{
    is_valid: boolean;
    validation_errors: string[];
    warnings: string[];
    suggestions: string[];
    estimated_performance_impact: string;
  }> {
    const response = await apiClient.post(`${this.distributedUrl}/policies/validate`, policy);
    return response.data;
  }
}

export const securityEnhancedService = new SecurityEnhancedService();