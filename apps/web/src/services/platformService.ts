/**
 * 平台集成 API（/api/v1/platform/*）
 * 只保留后端真实存在的接口。
 */

import apiClient from './apiClient'

export interface PlatformComponentInfo {
  component_id: string
  component_type: string
  name: string
  version: string
  status: string
  health_endpoint: string
  api_endpoint: string
  metadata: Record<string, any>
  registered_at: string
  last_heartbeat: string
}

export interface PlatformHealthStatus {
  overall_status: string
  healthy_components: number
  total_components: number
  components: Record<string, Record<string, any>>
  timestamp: string
}

export interface RegisterComponentRequest {
  component_id: string
  component_type: string
  name: string
  version: string
  health_endpoint: string
  api_endpoint: string
  metadata?: Record<string, any>
}

export interface RegisterComponentResponse {
  status: string
  component_id: string
  message: string
  component_status?: string
}

export interface UnregisterComponentResponse {
  status: string
  component_id: string
  message: string
}

export interface WorkflowRunRequest {
  workflow_type: string
  parameters?: Record<string, any>
  priority?: number
}

export interface WorkflowRunResponse {
  status: string
  workflow_id: string
  workflow_type: string
  message: string
  estimated_duration?: string
}

class PlatformService {
  private baseUrl = '/platform'

  async getHealth(): Promise<PlatformHealthStatus> {
    const response = await apiClient.get(`${this.baseUrl}/health`)
    return response.data
  }

  async getComponents(): Promise<PlatformComponentInfo[]> {
    const response = await apiClient.get(`${this.baseUrl}/components`)
    const data: any = response.data
    const components = data?.components
    if (!components || typeof components !== 'object') return []
    return Object.values(components) as PlatformComponentInfo[]
  }

  async registerComponent(
    request: RegisterComponentRequest
  ): Promise<RegisterComponentResponse> {
    const response = await apiClient.post(
      `${this.baseUrl}/components/register`,
      {
        ...request,
        metadata: request.metadata || {},
      }
    )
    return response.data
  }

  async deleteComponent(
    componentId: string
  ): Promise<UnregisterComponentResponse> {
    const response = await apiClient.delete(
      `${this.baseUrl}/components/${componentId}`
    )
    return response.data
  }

  async runWorkflow(request: WorkflowRunRequest): Promise<WorkflowRunResponse> {
    const response = await apiClient.post(`${this.baseUrl}/workflows/run`, {
      workflow_type: request.workflow_type,
      parameters: request.parameters || {},
      priority: request.priority || 0,
    })
    return response.data
  }

  async getWorkflowStatus(workflowId: string): Promise<any> {
    const response = await apiClient.get(
      `${this.baseUrl}/workflows/${workflowId}/status`
    )
    return response.data
  }
}

export const platformService = new PlatformService()
export default platformService
