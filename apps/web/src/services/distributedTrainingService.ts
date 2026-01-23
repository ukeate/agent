/**
 * 分布式训练服务
 *
 * 管理多GPU分布式训练任务、资源调度和监控
 */

import apiClient from './apiClient'

import { logger } from '../utils/logger'
// ==================== 接口定义 ====================

export interface DistributedConfig {
  strategy:
    | 'data_parallel'
    | 'model_parallel'
    | 'parameter_server'
    | 'allreduce'
    | 'async_updates'
  num_workers: number
  num_ps: number
  batch_size_per_worker: number
  gradient_accumulation_steps: number
  sync_frequency: number
  compression_enabled: boolean
  bandwidth_limit?: number
  fault_tolerance: boolean
  checkpoint_frequency: number
  use_xla: boolean
  mixed_precision: boolean
  gradient_clipping: number
}

export interface GPUNode {
  node_id: string
  gpu_id: number
  name: string
  status: 'idle' | 'training' | 'error' | 'offline'
  usage_percent: number
  memory_used_gb: number
  memory_total_gb: number
  temperature_celsius: number
  power_watts: number
  current_task?: string
}

export interface TrainingJob {
  job_id: string
  name: string
  status: 'pending' | 'running' | 'completed' | 'failed' | 'paused'
  strategy: string
  num_workers: number
  assigned_gpus: string[]
  start_time?: string
  end_time?: string
  progress_percent: number
  current_epoch: number
  total_epochs: number
  current_loss: number
  sync_efficiency: number
  estimated_time_remaining?: string
}

export interface ClusterStatus {
  total_nodes: number
  active_nodes: number
  total_gpus: number
  available_gpus: number
  total_memory_gb: number
  used_memory_gb: number
  cluster_efficiency: number
  network_bandwidth_gbps: number
  jobs_running: number
  jobs_pending: number
}

export interface DeepSpeedConfig {
  zero_stage: 0 | 1 | 2 | 3
  offload_optimizer: boolean
  offload_params: boolean
  gradient_compression: boolean
  communication_backend: 'nccl' | 'gloo' | 'mpi'
  fp16_enabled: boolean
  bf16_enabled: boolean
  gradient_accumulation_steps: number
  train_batch_size: number
  train_micro_batch_size_per_gpu: number
}

export interface TrainingMetrics {
  timestamp: string
  job_id: string
  loss: number
  accuracy?: number
  learning_rate: number
  throughput_samples_per_sec: number
  gpu_utilization_percent: number[]
  memory_utilization_gb: number[]
  network_bandwidth_mbps: number
  sync_time_ms: number
}

// ==================== 服务类 ====================

class DistributedTrainingService {
  private baseUrl = '/distributed-training'

  // ==================== 集群管理 ====================

  /**
   * 获取集群状态
   */
  async getClusterStatus(): Promise<ClusterStatus> {
    try {
      const response = await apiClient.get(`${this.baseUrl}/cluster/status`)
      return response.data
    } catch (error) {
      logger.error('获取集群状态失败:', error)
      throw error
    }
  }

  /**
   * 获取GPU节点列表
   */
  async getGPUNodes(): Promise<GPUNode[]> {
    try {
      const response = await apiClient.get(`${this.baseUrl}/nodes`)
      return response.data
    } catch (error) {
      logger.error('获取GPU节点失败:', error)
      throw error
    }
  }

  // ==================== 任务管理 ====================

  /**
   * 创建分布式训练任务
   */
  async createTrainingJob(
    name: string,
    config: DistributedConfig,
    model_config: any
  ): Promise<TrainingJob> {
    try {
      const response = await apiClient.post(`${this.baseUrl}/jobs`, {
        name,
        config,
        model_config,
      })
      return response.data
    } catch (error) {
      logger.error('创建训练任务失败:', error)
      throw error
    }
  }

  /**
   * 获取训练任务列表
   */
  async getTrainingJobs(status?: string): Promise<TrainingJob[]> {
    try {
      const params = status ? { status } : {}
      const response = await apiClient.get(`${this.baseUrl}/jobs`, { params })
      return response.data
    } catch (error) {
      logger.error('获取训练任务失败:', error)
      throw error
    }
  }

  /**
   * 获取单个训练任务详情
   */
  async getTrainingJob(jobId: string): Promise<TrainingJob> {
    const response = await apiClient.get(`${this.baseUrl}/jobs/${jobId}`)
    return response.data
  }

  /**
   * 控制训练任务
   */
  async controlTrainingJob(
    jobId: string,
    action: 'start' | 'pause' | 'resume' | 'stop'
  ): Promise<{ success: boolean; message: string }> {
    const response = await apiClient.post(
      `${this.baseUrl}/jobs/${jobId}/control`,
      {
        action,
      }
    )
    return response.data
  }

  // ==================== DeepSpeed配置 ====================

  /**
   * 获取DeepSpeed配置
   */
  async getDeepSpeedConfig(jobId: string): Promise<DeepSpeedConfig> {
    try {
      const response = await apiClient.get(
        `${this.baseUrl}/jobs/${jobId}/deepspeed`
      )
      return response.data
    } catch (error) {
      logger.error('获取DeepSpeed配置失败:', error)
      throw error
    }
  }

  /**
   * 更新DeepSpeed配置
   */
  async updateDeepSpeedConfig(
    jobId: string,
    config: DeepSpeedConfig
  ): Promise<{ success: boolean }> {
    const response = await apiClient.put(
      `${this.baseUrl}/jobs/${jobId}/deepspeed`,
      config
    )
    return response.data
  }

  // ==================== 性能监控 ====================

  /**
   * 获取训练指标
   */
  async getTrainingMetrics(
    jobId: string,
    timeRange?: { start: string; end: string }
  ): Promise<TrainingMetrics[]> {
    try {
      const params = timeRange || {}
      const response = await apiClient.get(
        `${this.baseUrl}/jobs/${jobId}/metrics`,
        { params }
      )
      return response.data
    } catch (error) {
      logger.error('获取训练指标失败:', error)
      throw error
    }
  }

  /**
   * 获取同步效率分析
   */
  async getSyncEfficiency(jobId: string): Promise<{
    average_efficiency: number
    bottleneck_node?: string
    recommendations: string[]
  }> {
    try {
      const response = await apiClient.get(
        `${this.baseUrl}/jobs/${jobId}/sync-analysis`
      )
      return response.data
    } catch (error) {
      logger.error('获取同步效率分析失败:', error)
      throw error
    }
  }

  // ==================== 资源调度 ====================

  /**
   * 请求GPU资源
   */
  async requestGPUResources(
    numGPUs: number,
    requirements?: {
      min_memory_gb?: number
      preferred_nodes?: string[]
    }
  ): Promise<{
    allocated: boolean
    assigned_gpus: string[]
    message: string
  }> {
    const response = await apiClient.post(`${this.baseUrl}/resources/request`, {
      num_gpus: numGPUs,
      requirements,
    })
    return response.data
  }

  /**
   * 释放GPU资源
   */
  async releaseGPUResources(gpuIds: string[]): Promise<{ success: boolean }> {
    const response = await apiClient.post(`${this.baseUrl}/resources/release`, {
      gpu_ids: gpuIds,
    })
    return response.data
  }

  // ==================== 故障恢复 ====================

  /**
   * 获取检查点列表
   */
  async getCheckpoints(jobId: string): Promise<{
    checkpoints: Array<{
      checkpoint_id: string
      epoch: number
      step: number
      loss: number
      created_at: string
      size_mb: number
    }>
  }> {
    try {
      const response = await apiClient.get(
        `${this.baseUrl}/jobs/${jobId}/checkpoints`
      )
      return response.data
    } catch (error) {
      logger.error('获取检查点失败:', error)
      throw error
    }
  }

  /**
   * 从检查点恢复训练
   */
  async resumeFromCheckpoint(
    jobId: string,
    checkpointId: string
  ): Promise<{ success: boolean; message: string }> {
    const response = await apiClient.post(
      `${this.baseUrl}/jobs/${jobId}/resume`,
      {
        checkpoint_id: checkpointId,
      }
    )
    return response.data
  }
}

// 导出服务实例
export const distributedTrainingService = new DistributedTrainingService()
