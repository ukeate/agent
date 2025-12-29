import apiClient from './apiClient';

export interface AgentInfo {
  id: string;
  name: string;
  type: string;
  status: 'online' | 'offline' | 'error' | 'registering';
  version: string;
  endpoint: string;
  capabilities: string[];
  metadata: {
    description: string;
    tags: string[];
    owner: string;
    environment: string;
    region: string;
    created: string;
    lastSeen: string;
  };
  metrics: {
    uptime: number;
    responseTime: number;
    requestCount: number;
    errorRate: number;
    memoryUsage: number;
    cpuUsage: number;
  };
  healthCheck: {
    enabled: boolean;
    interval: number;
    timeout: number;
    retries: number;
    lastCheck: string;
    status: 'healthy' | 'unhealthy' | 'unknown';
  };
}

export interface RegisterAgentRequest {
  name: string;
  type: string;
  endpoint: string;
  version: string;
  capabilities: string[];
  description?: string;
  tags?: string[];
  owner?: string;
  environment?: string;
  region?: string;
  healthCheckInterval?: number;
  healthCheckTimeout?: number;
  healthCheckRetries?: number;
}

export interface UpdateAgentRequest {
  name?: string;
  endpoint?: string;
  capabilities?: string[];
  metadata?: Partial<AgentInfo['metadata']>;
  healthCheck?: Partial<AgentInfo['healthCheck']>;
}

class AgentRegistryService {
  private baseUrl = '/service-discovery/agents';

  private normalizeUsage(value: any): number {
    const num = typeof value === 'number' && !Number.isNaN(value) ? value : 0;
    if (num <= 1) {
      return Math.round(num * 100);
    }
    return Math.round(num);
  }

  private buildCapabilities(capabilities: string[], description: string, version: string) {
    return capabilities.map((capability) => ({
      name: capability,
      description,
      version,
      input_schema: {},
      output_schema: {},
      performance_metrics: {},
      constraints: {},
    }));
  }

  private mapAgent(agent: any): AgentInfo {
    const resources = agent.resources || {};
    const createdAt = agent.created_at || '';
    const lastHeartbeat = agent.last_heartbeat || '';
    const uptimeHours = createdAt ? Math.max(0, (Date.now() - Date.parse(createdAt)) / 3600000) : 0;
    const requestCount = Number(agent.request_count || 0);
    const errorCount = Number(agent.error_count || 0);
    const errorRate = requestCount > 0 ? (errorCount / requestCount) * 100 : 0;
    const healthConfig = resources.health_check || {};
    const status = agent.status || '';
    const healthStatus =
      status === 'unhealthy'
        ? 'unhealthy'
        : status === 'active'
          ? 'healthy'
          : 'unknown';

    return {
      id: agent.agent_id,
      name: agent.name,
      type: agent.agent_type,
      status:
        status === 'active'
          ? 'online'
          : status === 'unhealthy'
            ? 'error'
            : 'offline',
      version: agent.version,
      endpoint: agent.endpoint,
      capabilities: Array.isArray(agent.capabilities)
        ? agent.capabilities.map((cap: any) =>
            typeof cap === 'string' ? cap : cap.name || cap.type || String(cap),
          )
        : [],
      metadata: {
        description: resources.description || '',
        tags: Array.isArray(agent.tags) ? agent.tags : [],
        owner: resources.owner || '',
        environment: agent.group || '',
        region: agent.region || '',
        created: createdAt,
        lastSeen: lastHeartbeat,
      },
      metrics: {
        uptime: Number(uptimeHours.toFixed(1)),
        responseTime: Number(agent.avg_response_time || 0),
        requestCount,
        errorRate: Number(errorRate.toFixed(2)),
        memoryUsage: this.normalizeUsage(resources.memory_usage),
        cpuUsage: this.normalizeUsage(resources.cpu_usage),
      },
      healthCheck: {
        enabled: Boolean(agent.health_endpoint),
        interval: Number(healthConfig.interval || 0),
        timeout: Number(healthConfig.timeout || 0),
        retries: Number(healthConfig.retries || 0),
        lastCheck: lastHeartbeat,
        status: healthStatus,
      },
    };
  }

  async listAgents(): Promise<AgentInfo[]> {
    const response = await apiClient.get(this.baseUrl);
    // 后端返回 {agents: AgentMetadataResponse[], total_count: number, query_time: number}
    if (response.data && Array.isArray(response.data.agents)) {
      return response.data.agents.map((agent: any) => this.mapAgent(agent));
    }
    // 如果格式不对，返回空数组
    return [];
  }

  async getAgent(agentId: string): Promise<AgentInfo> {
    const response = await apiClient.get(`${this.baseUrl}/${agentId}`);
    return this.mapAgent(response.data);
  }

  async registerAgent(data: RegisterAgentRequest): Promise<AgentInfo> {
    if (!data.version) {
      throw new Error('版本号不能为空');
    }
    if (!data.capabilities || data.capabilities.length === 0) {
      throw new Error('至少需要一个核心能力');
    }
    const url = new URL(data.endpoint);
    const port = url.port ? Number(url.port) : url.protocol === 'https:' ? 443 : 80;
    const healthEndpoint = new URL('/health', url).toString();
    const payload = {
      agent_id: data.name,
      agent_type: data.type,
      name: data.name,
      version: data.version,
      capabilities: this.buildCapabilities(data.capabilities, data.description || '', data.version),
      host: url.hostname,
      port,
      endpoint: data.endpoint,
      health_endpoint: healthEndpoint,
      resources: {
        description: data.description || '',
        owner: data.owner || '',
        environment: data.environment || '',
        health_check: {
          interval: data.healthCheckInterval || 0,
          timeout: data.healthCheckTimeout || 0,
          retries: data.healthCheckRetries || 0,
        },
      },
      tags: data.tags || [],
      group: data.environment || 'default',
      region: data.region || 'default',
    };
    const response = await apiClient.post(this.baseUrl, payload);
    const agentId = response.data?.agent_id || data.name;
    const detail = await apiClient.get(`${this.baseUrl}/${agentId}`);
    return this.mapAgent(detail.data);
  }

  async updateAgent(agentId: string, data: UpdateAgentRequest): Promise<AgentInfo> {
    const response = await apiClient.put(`${this.baseUrl}/${agentId}`, data);
    return response.data;
  }

  async deleteAgent(agentId: string): Promise<void> {
    await apiClient.delete(`${this.baseUrl}/${agentId}`);
  }

  async getAgentMetrics(agentId: string): Promise<AgentInfo['metrics']> {
    const agent = await this.getAgent(agentId);
    return agent.metrics;
  }

  async getAgentHealth(agentId: string): Promise<AgentInfo['healthCheck']> {
    const agent = await this.getAgent(agentId);
    return agent.healthCheck;
  }

  async triggerHealthCheck(agentId: string): Promise<AgentInfo['healthCheck']> {
    return this.getAgentHealth(agentId);
  }

  async restartAgent(agentId: string): Promise<void> {
    await apiClient.post(`${this.baseUrl}/${agentId}/restart`);
  }

  async getAgentLogs(agentId: string, limit: number = 100): Promise<string[]> {
    const response = await apiClient.get(`${this.baseUrl}/${agentId}/logs`, {
      params: { limit }
    });
    return response.data;
  }
}

export const agentRegistryService = new AgentRegistryService();
