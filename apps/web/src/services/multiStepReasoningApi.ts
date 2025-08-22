import apiClient from './apiClient';

// 类型定义
export interface DecompositionRequest {
  problem_statement: string;
  context?: string;
  strategy?: string;
  max_depth?: number;
  target_complexity?: number;
  enable_branching?: boolean;
  time_limit_minutes?: number;
}

export interface ExecutionRequest {
  workflow_definition_id: string;
  execution_mode?: string;
  max_parallel_steps?: number;
  scheduling_strategy?: string;
  input_data?: Record<string, any>;
}

export interface ExecutionControlRequest {
  execution_id: string;
  action: 'pause' | 'resume' | 'cancel';
}

export interface TaskDAG {
  id: string;
  name: string;
  description: string;
  nodes: TaskNode[];
  edges: TaskEdge[];
  parallel_groups: string[][];
  critical_path: string[];
  is_acyclic: boolean;
  total_nodes: number;
  max_depth: number;
}

export interface TaskNode {
  id: string;
  name: string;
  description: string;
  task_type: string;
  dependencies: string[];
  complexity_score: number;
  estimated_duration_minutes: number;
  priority: number;
}

export interface TaskEdge {
  from: string;
  to: string;
}

export interface WorkflowDefinition {
  id: string;
  name: string;
  description: string;
  steps: WorkflowStep[];
  execution_mode: string;
  max_parallel_steps: number;
  metadata: Record<string, any>;
}

export interface WorkflowStep {
  id: string;
  name: string;
  step_type: string;
  description: string;
  dependencies: string[];
  config: Record<string, any>;
  timeout_seconds: number;
}

export interface ExecutionResponse {
  execution_id: string;
  status: string;
  workflow_definition_id: string;
  progress: number;
  current_step?: string;
  start_time: string;
  estimated_completion?: string;
}

export interface SystemMetrics {
  active_workers: number;
  queue_depth: number;
  average_wait_time: number;
  success_rate: number;
  throughput: number;
  resource_utilization: Record<string, number>;
}

export interface DecompositionResponse {
  task_dag: TaskDAG;
  workflow_definition: WorkflowDefinition;
  decomposition_metadata: {
    strategy_used: string;
    complexity_achieved: number;
    total_estimated_time: number;
    parallelization_factor: number;
    critical_path_length: number;
  };
}

// 多步推理工作流API服务
export class MultiStepReasoningApi {
  private baseURL = '/multi-step-reasoning';

  // 问题分解
  async decomposeProblem(request: DecompositionRequest): Promise<DecompositionResponse> {
    try {
      const response = await apiClient.post(`${this.baseURL}/decompose`, request);
      return response.data;
    } catch (error) {
      console.error('Problem decomposition failed:', error);
      throw error;
    }
  }

  // 启动执行
  async startExecution(request: ExecutionRequest): Promise<ExecutionResponse> {
    try {
      const response = await apiClient.post(`${this.baseURL}/execute`, request);
      return response.data;
    } catch (error) {
      console.error('Execution start failed:', error);
      throw error;
    }
  }

  // 获取执行状态
  async getExecutionStatus(executionId: string): Promise<ExecutionResponse> {
    try {
      const response = await apiClient.get(`${this.baseURL}/executions/${executionId}`);
      return response.data;
    } catch (error) {
      console.error('Get execution status failed:', error);
      throw error;
    }
  }

  // 控制执行
  async controlExecution(request: ExecutionControlRequest): Promise<{ message: string; status: string }> {
    try {
      const response = await apiClient.post(`${this.baseURL}/executions/control`, request);
      return response.data;
    } catch (error) {
      console.error('Execution control failed:', error);
      throw error;
    }
  }

  // 获取系统指标
  async getSystemMetrics(): Promise<SystemMetrics> {
    try {
      const response = await apiClient.get(`${this.baseURL}/system/metrics`);
      return response.data;
    } catch (error) {
      console.error('Get system metrics failed:', error);
      throw error;
    }
  }

  // 获取工作流列表
  async listWorkflows(): Promise<{ workflows: Array<{
    id: string;
    name: string;
    description: string;
    step_count: number;
    execution_mode: string;
    created_at: string;
  }> }> {
    try {
      const response = await apiClient.get(`${this.baseURL}/workflows`);
      return response.data;
    } catch (error) {
      console.error('List workflows failed:', error);
      throw error;
    }
  }

  // 获取执行列表
  async listExecutions(): Promise<{ executions: Array<{
    id: string;
    workflow_definition_id: string;
    status: string;
    progress: number;
    started_at: string;
    total_steps: number;
    completed_steps: number;
  }> }> {
    try {
      const response = await apiClient.get(`${this.baseURL}/executions`);
      return response.data;
    } catch (error) {
      console.error('List executions failed:', error);
      throw error;
    }
  }

  // 删除执行
  async deleteExecution(executionId: string): Promise<{ message: string }> {
    try {
      const response = await apiClient.delete(`${this.baseURL}/executions/${executionId}`);
      return response.data;
    } catch (error) {
      console.error('Delete execution failed:', error);
      throw error;
    }
  }

  // 获取执行结果
  async getExecutionResults(executionId: string): Promise<{
    execution_id: string;
    results: {
      summary: string;
      validation_score: number;
      aggregated_result: Record<string, any>;
      step_results: Record<string, any>;
    };
    performance_metrics: {
      total_duration: number;
      average_step_duration: number;
      parallelization_efficiency: number;
      resource_utilization: number;
    };
  }> {
    try {
      const response = await apiClient.get(`${this.baseURL}/executions/${executionId}/results`);
      return response.data;
    } catch (error) {
      console.error('Get execution results failed:', error);
      throw error;
    }
  }

  // 轮询执行状态
  async pollExecutionStatus(
    executionId: string, 
    onUpdate: (status: ExecutionResponse) => void,
    interval: number = 2000
  ): Promise<void> {
    const poll = async () => {
      try {
        const status = await this.getExecutionStatus(executionId);
        onUpdate(status);
        
        // 如果还在执行中，继续轮询
        if (status.status === 'running' || status.status === 'paused') {
          setTimeout(poll, interval);
        }
      } catch (error) {
        console.error('Polling failed:', error);
        // 在错误情况下也继续轮询，但增加间隔
        setTimeout(poll, interval * 2);
      }
    };
    
    poll();
  }

  // 监控系统指标
  async monitorSystemMetrics(
    onUpdate: (metrics: SystemMetrics) => void,
    interval: number = 5000
  ): Promise<() => void> {
    let isMonitoring = true;
    
    const monitor = async () => {
      if (!isMonitoring) return;
      
      try {
        const metrics = await this.getSystemMetrics();
        onUpdate(metrics);
      } catch (error) {
        console.error('System metrics monitoring failed:', error);
      }
      
      if (isMonitoring) {
        setTimeout(monitor, interval);
      }
    };
    
    monitor();
    
    // 返回停止监控的函数
    return () => {
      isMonitoring = false;
    };
  }
}

// 导出单例实例
export const multiStepReasoningApi = new MultiStepReasoningApi();