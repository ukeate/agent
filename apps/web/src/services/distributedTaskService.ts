import apiClient from './apiClient';

export type TaskPriority = 'critical' | 'high' | 'medium' | 'low' | 'background';

export interface TaskSubmitRequest {
  task_type: string;
  task_data: Record<string, any>;
  requirements?: Record<string, any>;
  priority?: TaskPriority;
  decomposition_strategy?: string;
  assignment_strategy?: string;
}

export interface TaskResponse {
  task_id: string;
  status: string;
  message: string;
}

export interface SystemStats {
  node_id: string;
  raft_state: string;
  leader_id?: string;
  active_tasks: number;
  completed_tasks: number;
  queued_tasks: number;
  stats: Record<string, any>;
  state_summary: Record<string, any>;
}

export interface ConflictInfo {
  conflict_id: string;
  conflict_type: string;
  description: string;
  involved_tasks: string[];
  involved_agents: string[];
  timestamp: string;
  resolved: boolean;
  resolution_strategy?: string;
  resolution_result?: Record<string, any>;
}

class DistributedTaskService {
  private baseUrl = '/distributed-task';

  async initializeEngine(nodeId: string, clusterNodes: string[]): Promise<{ status: string; node_id: string }> {
    const response = await apiClient.post(`${this.baseUrl}/initialize`, clusterNodes, { params: { node_id: nodeId } });
    return response.data;
  }

  async submitTask(request: TaskSubmitRequest): Promise<TaskResponse> {
    const response = await apiClient.post(`${this.baseUrl}/submit`, request);
    return response.data;
  }

  async getTaskStatus(taskId: string): Promise<Record<string, any>> {
    const response = await apiClient.get(`${this.baseUrl}/status/${taskId}`);
    return response.data;
  }

  async cancelTask(taskId: string): Promise<Record<string, any>> {
    const response = await apiClient.post(`${this.baseUrl}/cancel/${taskId}`);
    return response.data;
  }

  async getSystemStats(): Promise<SystemStats> {
    const response = await apiClient.get(`${this.baseUrl}/stats`);
    return response.data;
  }

  async detectConflicts(): Promise<ConflictInfo[]> {
    const response = await apiClient.get(`${this.baseUrl}/conflicts`);
    return Array.isArray(response.data) ? response.data : [];
  }

  async resolveConflict(conflictId: string, strategy: string = 'priority_based'): Promise<Record<string, any>> {
    const response = await apiClient.post(`${this.baseUrl}/conflicts/resolve/${conflictId}`, null, { params: { strategy } });
    return response.data;
  }

  async createCheckpoint(name: string): Promise<Record<string, any>> {
    const response = await apiClient.post(`${this.baseUrl}/checkpoint/create`, null, { params: { name } });
    return response.data;
  }

  async rollbackCheckpoint(name: string): Promise<Record<string, any>> {
    const response = await apiClient.post(`${this.baseUrl}/checkpoint/rollback`, null, { params: { name } });
    return response.data;
  }

  async shutdownEngine(): Promise<Record<string, any>> {
    const response = await apiClient.post(`${this.baseUrl}/shutdown`);
    return response.data;
  }
}

export const distributedTaskService = new DistributedTaskService();

