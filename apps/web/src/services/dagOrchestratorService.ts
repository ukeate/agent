import apiClient from './apiClient';

export interface DAGNode {
  id: string;
  name: string;
  type: 'task' | 'condition' | 'start' | 'end';
  status: 'pending' | 'running' | 'completed' | 'failed' | 'skipped';
  dependencies: string[];
  duration?: number;
  start_time?: string;
  end_time?: string;
  metadata?: Record<string, any>;
  config?: {
    retry_count?: number;
    timeout?: number;
    executor?: string;
  };
}

export interface DAGWorkflow {
  id: string;
  name: string;
  description: string;
  status: 'draft' | 'active' | 'paused' | 'completed' | 'failed';
  created_at: string;
  updated_at?: string;
  nodes: DAGNode[];
  total_nodes: number;
  completed_nodes: number;
  failed_nodes: number;
  execution_time: number;
  progress: number;
  schedule?: string;
  tags?: string[];
}

export interface DAGExecution {
  id: string;
  workflow_id: string;
  workflow_name: string;
  status: string;
  start_time: string;
  end_time?: string;
  duration: number;
  total_tasks: number;
  completed_tasks: number;
  failed_tasks: number;
  execution_log: string[];
  error_message?: string;
  retry_count?: number;
}

export interface CreateWorkflowRequest {
  name: string;
  description: string;
  template?: string;
  nodes?: DAGNode[];
  schedule?: string;
  tags?: string[];
}

export interface WorkflowStatistics {
  total_workflows: number;
  active_workflows: number;
  total_executions: number;
  success_rate: number;
  average_execution_time: number;
  failed_executions: number;
  pending_executions: number;
}

class DagOrchestratorService {
  private baseUrl = '/dag-orchestrator';

  async listWorkflows(status?: string): Promise<DAGWorkflow[]> {
    const params = status ? { status } : {};
    const response = await apiClient.get(`${this.baseUrl}/workflows`, { params });
    return response.data;
  }

  async getWorkflow(workflowId: string): Promise<DAGWorkflow> {
    const response = await apiClient.get(`${this.baseUrl}/workflows/${workflowId}`);
    return response.data;
  }

  async createWorkflow(data: CreateWorkflowRequest): Promise<DAGWorkflow> {
    const response = await apiClient.post(`${this.baseUrl}/workflows`, data);
    return response.data;
  }

  async updateWorkflow(workflowId: string, data: Partial<CreateWorkflowRequest>): Promise<DAGWorkflow> {
    const response = await apiClient.put(`${this.baseUrl}/workflows/${workflowId}`, data);
    return response.data;
  }

  async deleteWorkflow(workflowId: string): Promise<void> {
    await apiClient.delete(`${this.baseUrl}/workflows/${workflowId}`);
  }

  async executeWorkflow(workflowId: string, params?: Record<string, any>): Promise<DAGExecution> {
    const response = await apiClient.post(`${this.baseUrl}/workflows/${workflowId}/execute`, { params });
    return response.data;
  }

  async pauseWorkflow(workflowId: string): Promise<void> {
    await apiClient.post(`${this.baseUrl}/workflows/${workflowId}/pause`);
  }

  async resumeWorkflow(workflowId: string): Promise<void> {
    await apiClient.post(`${this.baseUrl}/workflows/${workflowId}/resume`);
  }

  async stopWorkflow(workflowId: string): Promise<void> {
    await apiClient.post(`${this.baseUrl}/workflows/${workflowId}/stop`);
  }

  async listExecutions(workflowId?: string): Promise<DAGExecution[]> {
    const params = workflowId ? { workflow_id: workflowId } : {};
    const response = await apiClient.get(`${this.baseUrl}/executions`, { params });
    return response.data;
  }

  async getExecution(executionId: string): Promise<DAGExecution> {
    const response = await apiClient.get(`${this.baseUrl}/executions/${executionId}`);
    return response.data;
  }

  async getExecutionLogs(executionId: string): Promise<string[]> {
    const response = await apiClient.get(`${this.baseUrl}/executions/${executionId}/logs`);
    return response.data;
  }

  async retryExecution(executionId: string): Promise<DAGExecution> {
    const response = await apiClient.post(`${this.baseUrl}/executions/${executionId}/retry`);
    return response.data;
  }

  async getStatistics(): Promise<WorkflowStatistics> {
    const response = await apiClient.get(`${this.baseUrl}/statistics`);
    return response.data;
  }

  async validateWorkflow(data: CreateWorkflowRequest): Promise<{
    valid: boolean;
    errors?: string[];
    warnings?: string[];
  }> {
    const response = await apiClient.post(`${this.baseUrl}/validate`, data);
    return response.data;
  }

  async exportWorkflow(workflowId: string, format: 'json' | 'yaml' | 'xml' = 'json'): Promise<Blob> {
    const response = await apiClient.get(`${this.baseUrl}/workflows/${workflowId}/export`, {
      params: { format },
      responseType: 'blob'
    });
    return response.data;
  }

  async importWorkflow(file: File): Promise<DAGWorkflow> {
    const formData = new FormData();
    formData.append('file', file);
    const response = await apiClient.post(`${this.baseUrl}/import`, formData, {
      headers: { 'Content-Type': 'multipart/form-data' }
    });
    return response.data;
  }

  async getWorkflowTemplates(): Promise<Array<{
    id: string;
    name: string;
    description: string;
    category: string;
    nodes_count: number;
  }>> {
    const response = await apiClient.get(`${this.baseUrl}/templates`);
    return response.data;
  }

  async cloneWorkflow(workflowId: string, newName: string): Promise<DAGWorkflow> {
    const response = await apiClient.post(`${this.baseUrl}/workflows/${workflowId}/clone`, {
      name: newName
    });
    return response.data;
  }
}

export const dagOrchestratorService = new DagOrchestratorService();