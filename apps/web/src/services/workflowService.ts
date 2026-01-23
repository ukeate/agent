/**
 * 工作流服务
 */
import apiClient from './apiClient'

export interface WorkflowCreateRequest {
  name: string
  description?: string
  workflow_type: 'simple' | 'conditional' | 'custom'
  definition?: Record<string, any>
}

export interface WorkflowExecuteRequest {
  input_data?: Record<string, any>
}

export interface WorkflowControlRequest {
  action: 'pause' | 'resume' | 'cancel'
  reason?: string
}

export interface WorkflowResponse {
  id: string
  name?: string
  description?: string
  workflow_type?: string
  status: string
  current_state?: any
  created_at?: string
  started_at?: string
  paused_at?: string
  resumed_at?: string
  completed_at?: string
  failed_at?: string
  cancelled_at?: string
  error_message?: string
}

export interface CheckpointResponse {
  id: string
  workflow_id: string
  created_at: string
  version: number
  metadata: Record<string, any>
}

class WorkflowService {
  private baseUrl = '/workflows'

  async createWorkflow(data: WorkflowCreateRequest): Promise<WorkflowResponse> {
    const response = await apiClient.post(`${this.baseUrl}/`, data)
    return response.data
  }

  async listWorkflows(params?: {
    status?: string
    limit?: number
    offset?: number
  }): Promise<WorkflowResponse[]> {
    const response = await apiClient.get(`${this.baseUrl}/`, { params })
    return response.data
  }

  async getWorkflow(workflowId: string): Promise<WorkflowResponse> {
    const response = await apiClient.get(`${this.baseUrl}/${workflowId}`)
    return response.data
  }

  async startWorkflow(
    workflowId: string,
    request?: WorkflowExecuteRequest
  ): Promise<WorkflowResponse> {
    const response = await apiClient.post(
      `${this.baseUrl}/${workflowId}/start`,
      request || {}
    )
    return response.data
  }

  async controlWorkflow(
    workflowId: string,
    request: WorkflowControlRequest
  ): Promise<{ message: string; workflow_id: string }> {
    const response = await apiClient.put(
      `${this.baseUrl}/${workflowId}/control`,
      request
    )
    return response.data
  }

  async deleteWorkflow(
    workflowId: string
  ): Promise<{ message: string; workflow_id: string }> {
    const response = await apiClient.delete(`${this.baseUrl}/${workflowId}`)
    return response.data
  }

  async getCheckpoints(workflowId: string): Promise<CheckpointResponse[]> {
    const response = await apiClient.get(
      `${this.baseUrl}/${workflowId}/checkpoints`
    )
    return response.data
  }
}

export const workflowService = new WorkflowService()
