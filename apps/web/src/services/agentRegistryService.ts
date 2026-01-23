import apiClient from './apiClient'

export interface AgentInfo {
  id: string
  name: string
  type: string
  status: 'online' | 'offline' | 'error' | 'registering'
  version: string
  endpoint: string
  capabilities: string[]
  metadata: {
    description: string
    tags: string[]
    owner: string
    environment: string
    region: string
    created: string
    lastSeen: string
  }
  metrics: {
    uptime: number
    responseTime: number
    requestCount: number
    errorRate: number
    memoryUsage: number
    cpuUsage: number
  }
  healthCheck: {
    enabled: boolean
    interval: number
    timeout: number
    retries: number
    lastCheck: string
    status: 'healthy' | 'unhealthy' | 'unknown'
  }
}

export interface RegisterAgentRequest {
  name: string
  type: string
  endpoint: string
  version: string
  capabilities: string[]
  description?: string
  tags?: string[]
  owner?: string
  environment?: string
  region?: string
  healthCheckInterval?: number
  healthCheckTimeout?: number
  healthCheckRetries?: number
}

export interface UpdateAgentRequest {
  name?: string
  endpoint?: string
  capabilities?: string[]
  metadata?: Partial<AgentInfo['metadata']>
  healthCheck?: Partial<AgentInfo['healthCheck']>
}

class AgentRegistryService {
  private baseUrl = '/service-discovery/agents'

  private normalizeString(value: unknown): string {
    if (typeof value !== 'string') return ''
    return value.trim()
  }

  private normalizeStringList(values?: string[]): string[] {
    if (!Array.isArray(values)) return []
    const cleaned = values
      .map(item => String(item).trim())
      .filter(item => item.length > 0)
    return Array.from(new Set(cleaned))
  }

  private normalizePositiveNumber(value?: number | string | null): number {
    if (value === null || value === undefined) return 0
    const parsed = typeof value === 'string' ? Number(value) : value
    if (!Number.isFinite(parsed) || parsed < 0) return 0
    return parsed
  }

  private normalizeEndpoint(endpoint: string): URL {
    const trimmed = endpoint.trim()
    if (!trimmed) throw new Error('服务端点不能为空')
    const withProtocol = /^https?:\/\//i.test(trimmed)
      ? trimmed
      : `http://${trimmed}`
    try {
      return new URL(withProtocol)
    } catch {
      throw new Error('服务端点格式不正确')
    }
  }

  private normalizeUsage(value: any): number {
    const num = typeof value === 'number' && !Number.isNaN(value) ? value : 0
    if (num <= 1) {
      return Math.round(num * 100)
    }
    return Math.round(num)
  }

  private buildCapabilities(
    capabilities: string[],
    description: string,
    version: string
  ) {
    return capabilities.map(capability => ({
      name: capability,
      description,
      version,
      input_schema: {},
      output_schema: {},
      performance_metrics: {},
      constraints: {},
    }))
  }

  private mapAgent(agent: any): AgentInfo {
    const resources = agent.resources || {}
    const createdAt = agent.created_at || ''
    const lastHeartbeat = agent.last_heartbeat || ''
    const uptimeHours = createdAt
      ? Math.max(0, (Date.now() - Date.parse(createdAt)) / 3600000)
      : 0
    const requestCount = Number(agent.request_count || 0)
    const errorCount = Number(agent.error_count || 0)
    const errorRate = requestCount > 0 ? (errorCount / requestCount) * 100 : 0
    const healthConfig = resources.health_check || {}
    const status = agent.status || ''
    const healthStatus =
      status === 'unhealthy'
        ? 'unhealthy'
        : status === 'active'
          ? 'healthy'
          : 'unknown'

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
            typeof cap === 'string' ? cap : cap.name || cap.type || String(cap)
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
    }
  }

  async listAgents(): Promise<AgentInfo[]> {
    const response = await apiClient.get(this.baseUrl)
    // 后端返回 {agents: AgentMetadataResponse[], total_count: number, query_time: number}
    if (response.data && Array.isArray(response.data.agents)) {
      return response.data.agents.map((agent: any) => this.mapAgent(agent))
    }
    // 如果格式不对，返回空数组
    return []
  }

  async getAgent(agentId: string): Promise<AgentInfo> {
    const response = await apiClient.get(`${this.baseUrl}/${agentId}`)
    return this.mapAgent(response.data)
  }

  async registerAgent(data: RegisterAgentRequest): Promise<AgentInfo> {
    const name = this.normalizeString(data.name)
    if (!name) {
      throw new Error('智能体名称不能为空')
    }
    const type = this.normalizeString(data.type)
    if (!type) {
      throw new Error('智能体类型不能为空')
    }
    if (!data.version) {
      throw new Error('版本号不能为空')
    }
    const capabilities = this.normalizeStringList(data.capabilities)
    if (capabilities.length === 0) {
      throw new Error('至少需要一个核心能力')
    }
    const endpoint = this.normalizeString(data.endpoint)
    const url = this.normalizeEndpoint(endpoint)
    const port = url.port
      ? Number(url.port)
      : url.protocol === 'https:'
        ? 443
        : 80
    const healthEndpoint = new URL('/health', url).toString()
    const description = this.normalizeString(data.description)
    const owner = this.normalizeString(data.owner)
    const environment = this.normalizeString(data.environment)
    const region = this.normalizeString(data.region)
    const tags = this.normalizeStringList(data.tags)
    const healthCheckInterval = this.normalizePositiveNumber(
      data.healthCheckInterval
    )
    const healthCheckTimeout = this.normalizePositiveNumber(
      data.healthCheckTimeout
    )
    const healthCheckRetries = this.normalizePositiveNumber(
      data.healthCheckRetries
    )
    const payload = {
      agent_id: name,
      agent_type: type,
      name,
      version: data.version,
      capabilities: this.buildCapabilities(
        capabilities,
        description,
        data.version
      ),
      host: url.hostname,
      port,
      endpoint: url.toString(),
      health_endpoint: healthEndpoint,
      resources: {
        description,
        owner,
        environment,
        health_check: {
          interval: healthCheckInterval,
          timeout: healthCheckTimeout,
          retries: healthCheckRetries,
        },
      },
      tags,
      group: environment || 'default',
      region: region || 'default',
    }
    const response = await apiClient.post(this.baseUrl, payload)
    const agentId = response.data?.agent_id || name
    const detail = await apiClient.get(`${this.baseUrl}/${agentId}`)
    return this.mapAgent(detail.data)
  }

  async updateAgent(
    agentId: string,
    data: UpdateAgentRequest
  ): Promise<AgentInfo> {
    const response = await apiClient.put(`${this.baseUrl}/${agentId}`, data)
    return response.data
  }

  async deleteAgent(agentId: string): Promise<void> {
    await apiClient.delete(`${this.baseUrl}/${agentId}`)
  }

  async getAgentMetrics(agentId: string): Promise<AgentInfo['metrics']> {
    const agent = await this.getAgent(agentId)
    return agent.metrics
  }

  async getAgentHealth(agentId: string): Promise<AgentInfo['healthCheck']> {
    const agent = await this.getAgent(agentId)
    return agent.healthCheck
  }

  async triggerHealthCheck(agentId: string): Promise<AgentInfo['healthCheck']> {
    return this.getAgentHealth(agentId)
  }

  async restartAgent(agentId: string): Promise<void> {
    await apiClient.post(`${this.baseUrl}/${agentId}/restart`)
  }

  async getAgentLogs(agentId: string, limit: number = 100): Promise<string[]> {
    const response = await apiClient.get(`${this.baseUrl}/${agentId}/logs`, {
      params: { limit },
    })
    return response.data
  }
}

export const agentRegistryService = new AgentRegistryService()
