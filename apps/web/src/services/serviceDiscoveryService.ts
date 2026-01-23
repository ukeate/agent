import { apiClient } from './apiClient'

export interface AgentCapability {
  name: string
  description: string
  version: string
  input_schema?: any
  output_schema?: any
  performance_metrics?: any
  constraints?: any
}

export interface AgentRegistrationRequest {
  agent_id: string
  agent_type: string
  name: string
  version: string
  capabilities: AgentCapability[]
  host: string
  port: number
  endpoint: string
  health_endpoint?: string
  resources?: any
  tags?: string[]
  group?: string
  region?: string
}

export interface AgentMetadataResponse {
  agent_id: string
  agent_type: string
  name: string
  version: string
  capabilities: AgentCapability[]
  host: string
  port: number
  endpoint: string
  health_endpoint?: string
  resources?: any
  tags?: string[]
  group?: string
  region?: string
  status: string
  created_at: string
  last_heartbeat?: string
  request_count: number
  error_count: number
  avg_response_time: number
}

export interface AgentDiscoveryResponse {
  agents: AgentMetadataResponse[]
  total_count: number
  query_time: number
}

export interface LoadBalancerRequest {
  capability: string
  strategy: string
  tags?: string[]
  requirements?: any
}

export interface LoadBalancerResponse {
  selected_agent: AgentMetadataResponse | null
  selection_time: number
  strategy_used: string
}

export interface ServiceStats {
  registry: any
  load_balancer: any
  system_status: any
}

export interface HealthCheckResponse {
  status: string
  timestamp: number
  version: string
  uptime_seconds: number
}

export class ServiceDiscoveryService {
  private baseUrl = '/service-discovery'

  async registerAgent(request: AgentRegistrationRequest): Promise<any> {
    const response = await apiClient.post(`${this.baseUrl}/agents`, request)
    return response.data
  }

  async discoverAgents(params: {
    capability?: string
    tags?: string
    status_filter?: string
    group?: string
    region?: string
    limit?: number
  }): Promise<AgentDiscoveryResponse> {
    const response = await apiClient.get(`${this.baseUrl}/agents`, { params })
    return response.data
  }

  async getAgent(agentId: string): Promise<AgentMetadataResponse> {
    const response = await apiClient.get(`${this.baseUrl}/agents/${agentId}`)
    return response.data
  }

  async updateAgentStatus(agentId: string, status: string): Promise<any> {
    const response = await apiClient.put(
      `${this.baseUrl}/agents/${agentId}/status`,
      {
        status: { value: status },
      }
    )
    return response.data
  }

  async updateAgentMetrics(
    agentId: string,
    metrics: {
      request_count?: number
      error_count?: number
      avg_response_time?: number
    }
  ): Promise<any> {
    const response = await apiClient.put(
      `${this.baseUrl}/agents/${agentId}/metrics`,
      metrics
    )
    return response.data
  }

  async deregisterAgent(agentId: string): Promise<any> {
    const response = await apiClient.delete(`${this.baseUrl}/agents/${agentId}`)
    return response.data
  }

  async selectAgent(
    request: LoadBalancerRequest
  ): Promise<LoadBalancerResponse> {
    const response = await apiClient.post(
      `${this.baseUrl}/load-balancer/select`,
      request
    )
    return response.data
  }

  async getSystemStats(): Promise<ServiceStats> {
    const response = await apiClient.get(`${this.baseUrl}/stats`)
    return response.data
  }

  async healthCheck(): Promise<HealthCheckResponse> {
    const response = await apiClient.get(`${this.baseUrl}/health`)
    return response.data
  }

  async getConfiguration(): Promise<any> {
    const response = await apiClient.get(`${this.baseUrl}/config`)
    return response.data
  }

  async getAgentsByCapability(
    capability: string
  ): Promise<AgentMetadataResponse[]> {
    const response = await this.discoverAgents({ capability })
    return response.agents
  }

  async getActiveAgents(): Promise<AgentMetadataResponse[]> {
    const response = await this.discoverAgents({ status_filter: 'active' })
    return response.agents
  }

  async getAgentsByGroup(group: string): Promise<AgentMetadataResponse[]> {
    const response = await this.discoverAgents({ group })
    return response.agents
  }

  async getAgentsByRegion(region: string): Promise<AgentMetadataResponse[]> {
    const response = await this.discoverAgents({ region })
    return response.agents
  }
}

export const serviceDiscoveryService = new ServiceDiscoveryService()
