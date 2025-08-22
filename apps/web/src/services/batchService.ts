/**
 * 批处理服务
 * 
 * 提供批处理相关的API调用服务
 */

import { apiClient } from './apiClient';

// 批处理任务状态
export enum BatchStatus {
  PENDING = 'pending',
  RUNNING = 'running',
  COMPLETED = 'completed',
  FAILED = 'failed',
  CANCELLED = 'cancelled'
}

// 批处理任务
export interface BatchTask {
  id: string;
  type: string;
  data: any;
  priority: number;
  retry_count: number;
  max_retries: number;
  status: BatchStatus;
  created_at: string;
  started_at?: string;
  completed_at?: string;
  result?: any;
  error?: string;
}

// 批处理作业
export interface BatchJob {
  id: string;
  tasks: BatchTask[];
  status: BatchStatus;
  total_tasks: number;
  completed_tasks: number;
  failed_tasks: number;
  created_at: string;
  started_at?: string;
  completed_at?: string;
}

// 批处理指标
export interface BatchMetrics {
  active_jobs: number;
  pending_jobs: number;
  completed_jobs: number;
  failed_jobs: number;
  total_tasks: number;
  completed_tasks: number;
  failed_tasks: number;
  pending_tasks: number;
  tasks_per_second: number;
  avg_task_duration: number;
  success_rate: number;
  queue_depth: number;
  active_workers: number;
  max_workers: number;
}

// 创建批处理作业请求
export interface CreateBatchJobRequest {
  tasks: Array<{
    type: string;
    data: any;
    priority?: number;
  }>;
  batch_size?: number;
  max_retries?: number;
}

class BatchService {
  private baseUrl = '/batch';

  /**
   * 创建批处理作业
   */
  async createJob(request: CreateBatchJobRequest): Promise<{ job_id: string }> {
    const response = await apiClient.post(`${this.baseUrl}/jobs`, request);
    return response.data;
  }

  /**
   * 获取批处理作业列表
   */
  async getJobs(status?: BatchStatus): Promise<{ jobs: BatchJob[] }> {
    const params = status ? { status } : {};
    const response = await apiClient.get(`${this.baseUrl}/jobs`, { params });
    return response.data;
  }

  /**
   * 获取批处理作业详情
   */
  async getJobDetails(jobId: string): Promise<BatchJob> {
    const response = await apiClient.get(`${this.baseUrl}/jobs/${jobId}`);
    return response.data;
  }

  /**
   * 取消批处理作业
   */
  async cancelJob(jobId: string): Promise<{ success: boolean }> {
    const response = await apiClient.post(`${this.baseUrl}/jobs/${jobId}/cancel`);
    return response.data;
  }

  /**
   * 重试失败的任务
   */
  async retryFailedTasks(jobId: string): Promise<{ retried_count: number }> {
    const response = await apiClient.post(`${this.baseUrl}/jobs/${jobId}/retry`);
    return response.data;
  }

  /**
   * 获取批处理系统指标
   */
  async getMetrics(): Promise<BatchMetrics> {
    const response = await apiClient.get(`${this.baseUrl}/metrics`);
    return response.data;
  }

  /**
   * 获取任务详情
   */
  async getTaskDetails(taskId: string): Promise<BatchTask> {
    const response = await apiClient.get(`${this.baseUrl}/tasks/${taskId}`);
    return response.data;
  }

  /**
   * 批量提交任务
   */
  async submitBulkTasks(tasks: Array<{ type: string; data: any; priority?: number }>): Promise<{ 
    job_id: string;
    task_count: number;
  }> {
    const response = await apiClient.post(`${this.baseUrl}/bulk`, { tasks });
    return response.data;
  }

  /**
   * 获取任务执行历史
   */
  async getTaskHistory(taskId: string): Promise<{
    executions: Array<{
      attempt: number;
      started_at: string;
      completed_at?: string;
      status: BatchStatus;
      error?: string;
    }>;
  }> {
    const response = await apiClient.get(`${this.baseUrl}/tasks/${taskId}/history`);
    return response.data;
  }

  /**
   * 暂停批处理作业
   */
  async pauseJob(jobId: string): Promise<{ success: boolean }> {
    const response = await apiClient.post(`${this.baseUrl}/jobs/${jobId}/pause`);
    return response.data;
  }

  /**
   * 恢复批处理作业
   */
  async resumeJob(jobId: string): Promise<{ success: boolean }> {
    const response = await apiClient.post(`${this.baseUrl}/jobs/${jobId}/resume`);
    return response.data;
  }

  /**
   * 获取工作线程状态
   */
  async getWorkerStatus(): Promise<{
    workers: Array<{
      id: string;
      status: 'idle' | 'busy';
      current_task?: string;
      processed_count: number;
      error_count: number;
      uptime: number;
    }>;
  }> {
    const response = await apiClient.get(`${this.baseUrl}/workers`);
    return response.data;
  }

  /**
   * 设置批处理配置
   */
  async updateConfig(config: {
    max_workers?: number;
    batch_size?: number;
    max_retries?: number;
    timeout?: number;
  }): Promise<{ success: boolean }> {
    const response = await apiClient.put(`${this.baseUrl}/config`, config);
    return response.data;
  }

  /**
   * 获取批处理配置
   */
  async getConfig(): Promise<{
    max_workers: number;
    batch_size: number;
    max_retries: number;
    timeout: number;
  }> {
    const response = await apiClient.get(`${this.baseUrl}/config`);
    return response.data;
  }
}

export const batchService = new BatchService();