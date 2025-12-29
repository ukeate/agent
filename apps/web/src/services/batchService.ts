/**
 * 批处理操作服务
 * 提供批处理任务管理和统计功能
 */

import apiClient from './apiClient';

// 批处理状态枚举
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
  name: string;
  status: BatchStatus;
  created_at: string;
  started_at?: string;
  completed_at?: string;
  progress: number;
  error_message?: string;
}

// 批处理指标
export interface BatchMetrics {
  total_jobs: number;
  active_jobs: number;
  completed_jobs: number;
  failed_jobs: number;
  success_rate: number;
  average_processing_time: number;
  // 扩展字段以支持组件需求
  tasks_per_second: number;
  active_workers: number;
  max_workers: number;
  queue_depth: number;
  total_tasks: number;
}

// 批处理统计接口
export interface BatchStatsSummary {
  total_jobs: number;
  active_jobs: number;
  completed_jobs: number;
  failed_jobs: number;
  success_rate: number;
  total_items_processed: number;
  average_processing_time: string;
  daily_throughput: number;
  resource_utilization: {
    cpu: number;
    memory: number;
    storage: number;
    network: number;
  };
  performance_metrics: {
    requests_per_second: number;
    average_latency: number;
    p99_latency: number;
    error_rate: number;
  };
}

// 批处理任务接口
export interface BatchJob {
  id?: string;  // 为了兼容
  job_id: string;
  name?: string;
  status: BatchStatus | 'pending' | 'running' | 'completed' | 'failed' | 'cancelled';
  job_type?: string;
  created_at: string;
  started_at?: string;
  completed_at?: string;
  total_items: number;
  processed_items: number;
  failed_items?: number;
  // 为组件兼容性添加别名字段
  total_tasks: number;
  completed_tasks: number;
  failed_tasks: number;
  progress: number; // 注意：API 返回的是 progress 而非 progress_percentage
  error_message?: string;
}

// 批处理任务创建请求
export interface BatchJobCreate {
  name: string;
  task_type: string;
  parameters: Record<string, any>;
  data_source?: string;
  priority?: number;
}

class BatchService {
  private baseUrl = '/batch';

  /**
   * 获取批处理统计汇总
   */
  async getStatsSummary(): Promise<BatchStatsSummary> {
    const response = await apiClient.get(`${this.baseUrl}/stats/summary`);
    return response.data;
  }

  /**
   * 获取批处理任务列表
   */
  async getJobs(
    status?: string,
    limit: number = 50,
    offset: number = 0
  ): Promise<BatchJob[]> {
    const response = await apiClient.get(`${this.baseUrl}/jobs`, {
      params: { status, limit, offset }
    });
    // API 返回格式: {jobs: [...], total: 25, limit: 50, offset: 0}
    // 提取 jobs 数组
    return response.data.jobs || [];
  }

  /**
   * 创建批处理任务
   */
  async createJob(jobData: BatchJobCreate): Promise<BatchJob> {
    const response = await apiClient.post(`${this.baseUrl}/jobs`, jobData);
    return response.data;
  }

  /**
   * 获取批处理任务详情
   */
  async getJob(jobId: string): Promise<BatchJob> {
    const response = await apiClient.get(`${this.baseUrl}/jobs/${jobId}`);
    return response.data;
  }

  /**
   * 取消批处理任务
   */
  async cancelJob(jobId: string): Promise<{ message: string }> {
    const response = await apiClient.post(`${this.baseUrl}/jobs/${jobId}/cancel`);
    return response.data;
  }

  /**
   * 重新启动批处理任务
   */
  async restartJob(jobId: string): Promise<{ message: string }> {
    const response = await apiClient.post(`${this.baseUrl}/jobs/${jobId}/restart`);
    return response.data;
  }

  /**
   * 获取任务执行日志
   */
  async getJobLogs(
    jobId: string,
    limit: number = 100
  ): Promise<Array<{
    timestamp: string;
    level: string;
    message: string;
  }>> {
    const response = await apiClient.get(`${this.baseUrl}/jobs/${jobId}/logs`, {
      params: { limit }
    });
    return response.data;
  }

  /**
   * 获取系统资源使用情况
   */
  async getResourceUsage(): Promise<{
    cpu_usage: number;
    memory_usage: number;
    disk_usage: number;
    network_io: {
      rx: string;
      tx: string;
    };
    active_workers: number;
    queue_size: number;
  }> {
    const response = await apiClient.get(`${this.baseUrl}/resources`);
    return response.data;
  }

  /**
   * 获取性能指标历史
   */
  async getPerformanceHistory(
    timeRange: '1h' | '24h' | '7d' | '30d' = '24h'
  ): Promise<Array<{
    timestamp: string;
    jobs_completed: number;
    average_processing_time: number;
    throughput: number;
    error_rate: number;
  }>> {
    const response = await apiClient.get(`${this.baseUrl}/performance/history`, {
      params: { time_range: timeRange }
    });
    return response.data;
  }

  /**
   * 批量操作任务
   */
  async batchOperation(
    operation: 'cancel' | 'restart' | 'delete',
    jobIds: string[]
  ): Promise<{
    success_count: number;
    failed_count: number;
    errors: string[];
  }> {
    const response = await apiClient.post(`${this.baseUrl}/jobs/batch`, {
      operation,
      job_ids: jobIds
    });
    return response.data;
  }

  /**
   * 获取任务队列状态
   */
  async getQueueStatus(): Promise<{
    total_queued: number;
    high_priority: number;
    normal_priority: number;
    low_priority: number;
    estimated_wait_time: string;
    worker_availability: {
      total_workers: number;
      active_workers: number;
      idle_workers: number;
    };
  }> {
    const response = await apiClient.get(`${this.baseUrl}/queue/status`);
    return response.data;
  }

  /**
   * 清理已完成的任务
   */
  async cleanupCompletedJobs(
    olderThanDays: number = 7
  ): Promise<{
    deleted_count: number;
    message: string;
  }> {
    const response = await apiClient.post(`${this.baseUrl}/cleanup`, {
      older_than_days: olderThanDays
    });
    return response.data;
  }

  /**
   * 获取批处理指标
   */
  async getMetrics(): Promise<BatchMetrics> {
    const stats = await this.getStatsSummary();
    return {
      total_jobs: stats.total_jobs,
      active_jobs: stats.active_jobs,
      completed_jobs: stats.completed_jobs,
      failed_jobs: stats.failed_jobs,
      success_rate: stats.success_rate,
      average_processing_time: parseFloat(stats.average_processing_time),
      // 扩展字段的默认值
      tasks_per_second: stats.performance_metrics?.requests_per_second || 0,
      active_workers: 5, // 默认值
      max_workers: 10, // 默认值
      queue_depth: stats.active_jobs * 2, // 估算值
      total_tasks: stats.total_items_processed || 0
    };
  }

  /**
   * 获取任务详情
   */
  async getJobDetails(jobId: string): Promise<BatchJob> {
    const job = await this.getJob(jobId);
    // 确保包含别名字段
    return {
      ...job,
      id: job.job_id,
      total_tasks: job.total_items,
      completed_tasks: job.processed_items,
      failed_tasks: job.failed_items || 0
    };
  }

  /**
   * 重试失败的任务
   */
  async retryFailedTasks(jobId: string): Promise<{ message: string }> {
    const response = await apiClient.post(`${this.baseUrl}/jobs/${jobId}/retry`);
    return response.data;
  }
}

export const batchService = new BatchService();