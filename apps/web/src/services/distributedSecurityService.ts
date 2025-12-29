import apiClient from './apiClient';

export type ThreatLevel = 'low' | 'medium' | 'high' | 'critical';
export type AlertStatus = 'active' | 'acknowledged' | 'resolved';
export type AuthenticationMethod = 'password' | 'token' | 'certificate' | 'biometric' | 'multi_factor';

export interface AuthenticationRequest {
  agent_id: string;
  credentials: Record<string, any>;
  authentication_methods: AuthenticationMethod[];
}

export interface AuthenticationResponse {
  authenticated: boolean;
  session_token?: string;
  trust_score: number;
  error_message?: string;
}

export interface AccessControlRequest {
  subject_id: string;
  resource_id: string;
  action: string;
  resource_type: string;
  context?: Record<string, any>;
}

export interface AccessControlResponse {
  decision: 'allow' | 'deny';
  reason: string;
  request_id: string;
  evaluation_time_ms: number;
}

export interface SecurityEventRequest {
  event_type: string;
  source_agent_id: string;
  target_resource?: string;
  action: string;
  result: string;
  details?: Record<string, any>;
}

export interface SecurityEventResponse {
  event_id: string;
  logged: boolean;
  message: string;
}

export interface SecurityMetrics {
  authentication: {
    total_attempts_24h: number;
    successful_attempts_24h: number;
    failed_attempts_24h: number;
    success_rate: number;
    average_response_time_ms: number;
  };
  access_control: {
    total_requests_24h: number;
    granted_requests_24h: number;
    denied_requests_24h: number;
    approval_rate: number;
    policy_evaluation_time_ms: number;
  };
  communication: {
    active_sessions: number;
    total_messages_24h: number;
    encryption_overhead_ms: number;
    integrity_violations: number;
  };
  threat_detection: {
    events_processed_24h: number;
    threats_detected_24h: number;
    false_positives_24h: number;
    alert_response_time_minutes: number;
  };
}

export interface SecurityAlert {
  id: string;
  alert_type: string;
  title: string;
  threat_level: ThreatLevel;
  status: AlertStatus;
  agent_id?: string;
  created_at: string;
  indicators: Record<string, any>;
  resolution?: {
    resolved_at: string;
    resolved_by: string;
    resolution_notes: string;
  };
}

export interface AgentIdentity {
  id: string;
  agent_id: string;
  display_name: string;
  trust_score: number;
  last_authentication: string;
  failed_attempts: number;
  is_locked: boolean;
  authentication_methods: AuthenticationMethod[];
  permissions: string[];
  last_activity?: string;
}

export interface CommunicationSession {
  session_id: string;
  participants: string[];
  established_at: string;
  last_activity: string;
  encryption_level: 'none' | 'standard' | 'high' | 'military';
  message_count: number;
  status: 'active' | 'idle' | 'terminated';
}

class DistributedSecurityService {
  private baseUrl = '/distributed-security';

  // Authentication
  async authenticate(request: AuthenticationRequest): Promise<AuthenticationResponse> {
    const response = await apiClient.post(`${this.baseUrl}/authenticate`, request);
    return response.data;
  }

  async validateToken(token: string): Promise<{ valid: boolean; agent_id?: string; expires_at?: string }> {
    const response = await apiClient.post(`${this.baseUrl}/validate-token`, { token });
    return response.data;
  }

  async revokeToken(token: string): Promise<{ success: boolean }> {
    const response = await apiClient.post(`${this.baseUrl}/revoke-token`, { token });
    return response.data;
  }

  // Access Control
  async checkAccess(request: AccessControlRequest): Promise<AccessControlResponse> {
    const response = await apiClient.post(`${this.baseUrl}/check-access`, request);
    return response.data;
  }

  async getPermissions(agent_id: string): Promise<string[]> {
    const response = await apiClient.get(`${this.baseUrl}/agents/${agent_id}/permissions`);
    return response.data.permissions;
  }

  async updatePermissions(agent_id: string, permissions: string[]): Promise<{ success: boolean }> {
    const response = await apiClient.put(`${this.baseUrl}/agents/${agent_id}/permissions`, { permissions });
    return response.data;
  }

  // Security Metrics
  async getMetrics(): Promise<SecurityMetrics> {
    const response = await apiClient.get(`${this.baseUrl}/metrics`);
    return response.data;
  }

  async getMetricsHistory(timeRange: { start: string; end: string }): Promise<SecurityMetrics[]> {
    const response = await apiClient.get(`${this.baseUrl}/metrics/history`, {
      params: timeRange
    });
    return response.data.metrics;
  }

  // Security Alerts
  async getAlerts(params?: {
    status?: AlertStatus;
    threat_level?: ThreatLevel;
    limit?: number;
    offset?: number;
  }): Promise<SecurityAlert[]> {
    const response = await apiClient.get(`${this.baseUrl}/alerts`, { params });
    return response.data.alerts;
  }

  async getAlert(alertId: string): Promise<SecurityAlert> {
    const response = await apiClient.get(`${this.baseUrl}/alerts/${alertId}`);
    return response.data;
  }

  async acknowledgeAlert(alertId: string): Promise<{ success: boolean }> {
    const response = await apiClient.post(`${this.baseUrl}/alerts/${alertId}/acknowledge`);
    return response.data;
  }

  async resolveAlert(alertId: string, resolution: {
    notes: string;
    resolved_by: string;
  }): Promise<{ success: boolean }> {
    const response = await apiClient.post(`${this.baseUrl}/alerts/${alertId}/resolve`, resolution);
    return response.data;
  }

  // Agent Management
  async getAgents(): Promise<AgentIdentity[]> {
    const response = await apiClient.get(`${this.baseUrl}/agents`);
    return response.data.agents;
  }

  async getAgent(agentId: string): Promise<AgentIdentity> {
    const response = await apiClient.get(`${this.baseUrl}/agents/${agentId}`);
    return response.data;
  }

  async lockAgent(agentId: string, reason: string): Promise<{ success: boolean }> {
    const response = await apiClient.post(`${this.baseUrl}/agents/${agentId}/lock`, { reason });
    return response.data;
  }

  async unlockAgent(agentId: string): Promise<{ success: boolean }> {
    const response = await apiClient.post(`${this.baseUrl}/agents/${agentId}/unlock`);
    return response.data;
  }

  async resetAgentCredentials(agentId: string): Promise<{ 
    success: boolean; 
    new_credentials?: Record<string, any>;
  }> {
    const response = await apiClient.post(`${this.baseUrl}/agents/${agentId}/reset-credentials`);
    return response.data;
  }

  // Security Events
  async logEvent(event: SecurityEventRequest): Promise<SecurityEventResponse> {
    const response = await apiClient.post(`${this.baseUrl}/events`, event);
    return response.data;
  }

  async getEvents(params?: {
    event_type?: string;
    agent_id?: string;
    start_time?: string;
    end_time?: string;
    limit?: number;
  }): Promise<Array<SecurityEventRequest & { event_id: string; timestamp: string }>> {
    const response = await apiClient.get(`${this.baseUrl}/events`, { params });
    return response.data.events;
  }

  // Communication Security
  async getCommunicationSessions(): Promise<CommunicationSession[]> {
    const response = await apiClient.get(`${this.baseUrl}/communication/sessions`);
    return response.data.sessions;
  }

  async establishSecureChannel(participant1: string, participant2: string): Promise<{
    session_id: string;
    encryption_key: string;
  }> {
    const response = await apiClient.post(`${this.baseUrl}/communication/establish`, {
      participant1,
      participant2
    });
    return response.data;
  }

  async terminateSession(sessionId: string): Promise<{ success: boolean }> {
    const response = await apiClient.post(`${this.baseUrl}/communication/sessions/${sessionId}/terminate`);
    return response.data;
  }

  // Threat Detection
  async getThreatIntelligence(): Promise<{
    known_threats: Array<{
      threat_id: string;
      threat_type: string;
      description: string;
      indicators: string[];
      mitigation: string;
    }>;
    risk_level: ThreatLevel;
  }> {
    const response = await apiClient.get(`${this.baseUrl}/threat-intelligence`);
    return response.data;
  }

  async scanForThreats(): Promise<{
    scan_id: string;
    threats_found: number;
    details: Array<{
      threat_type: string;
      severity: ThreatLevel;
      affected_resources: string[];
    }>;
  }> {
    const response = await apiClient.post(`${this.baseUrl}/threat-scan`);
    return response.data;
  }
}

export const distributedSecurityService = new DistributedSecurityService();