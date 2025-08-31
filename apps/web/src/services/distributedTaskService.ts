/**
 * 分布式任务协调API服务
 */

const API_BASE = '/api/v1/distributed-task';

export interface TaskSubmitRequest {
  task_type: string;
  task_data: Record<string, any>;
  requirements?: Record<string, any>;
  priority?: string;
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
  resolved?: boolean;
  resolution_strategy?: string;
}

export const distributedTaskService = {
  /**
   * 初始化协调引擎
   */
  async initializeEngine(nodeId: string, clusterNodes: string[]) {
    const response = await fetch(`${API_BASE}/initialize?node_id=${nodeId}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(clusterNodes)
    });
    
    if (!response.ok) {
      throw new Error(`Failed to initialize engine: ${response.statusText}`);
    }
    
    return response.json();
  },

  /**
   * 提交任务
   */
  async submitTask(request: TaskSubmitRequest): Promise<TaskResponse> {
    const response = await fetch(`${API_BASE}/submit`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(request)
    });
    
    if (!response.ok) {
      throw new Error(`Failed to submit task: ${response.statusText}`);
    }
    
    return response.json();
  },

  /**
   * 获取任务状态
   */
  async getTaskStatus(taskId: string) {
    const response = await fetch(`${API_BASE}/status/${taskId}`);
    
    if (!response.ok) {
      throw new Error(`Failed to get task status: ${response.statusText}`);
    }
    
    return response.json();
  },

  /**
   * 取消任务
   */
  async cancelTask(taskId: string) {
    const response = await fetch(`${API_BASE}/cancel/${taskId}`, {
      method: 'POST'
    });
    
    if (!response.ok) {
      throw new Error(`Failed to cancel task: ${response.statusText}`);
    }
    
    return response.json();
  },

  /**
   * 获取系统统计
   */
  async getSystemStats(): Promise<SystemStats> {
    const response = await fetch(`${API_BASE}/stats`);
    
    if (!response.ok) {
      throw new Error(`Failed to get system stats: ${response.statusText}`);
    }
    
    return response.json();
  },

  /**
   * 检测冲突
   */
  async detectConflicts(): Promise<ConflictInfo[]> {
    const response = await fetch(`${API_BASE}/conflicts`);
    
    if (!response.ok) {
      throw new Error(`Failed to detect conflicts: ${response.statusText}`);
    }
    
    return response.json();
  },

  /**
   * 解决冲突
   */
  async resolveConflict(conflictId: string, strategy: string = 'priority_based') {
    const response = await fetch(`${API_BASE}/conflicts/resolve/${conflictId}?strategy=${strategy}`, {
      method: 'POST'
    });
    
    if (!response.ok) {
      throw new Error(`Failed to resolve conflict: ${response.statusText}`);
    }
    
    return response.json();
  },

  /**
   * 创建检查点
   */
  async createCheckpoint(name: string) {
    const response = await fetch(`${API_BASE}/checkpoint/create?name=${name}`, {
      method: 'POST'
    });
    
    if (!response.ok) {
      throw new Error(`Failed to create checkpoint: ${response.statusText}`);
    }
    
    return response.json();
  },

  /**
   * 回滚到检查点
   */
  async rollbackCheckpoint(name: string) {
    const response = await fetch(`${API_BASE}/checkpoint/rollback?name=${name}`, {
      method: 'POST'
    });
    
    if (!response.ok) {
      throw new Error(`Failed to rollback checkpoint: ${response.statusText}`);
    }
    
    return response.json();
  },

  /**
   * 关闭引擎
   */
  async shutdownEngine() {
    const response = await fetch(`${API_BASE}/shutdown`, {
      method: 'POST'
    });
    
    if (!response.ok) {
      throw new Error(`Failed to shutdown engine: ${response.statusText}`);
    }
    
    return response.json();
  }
};