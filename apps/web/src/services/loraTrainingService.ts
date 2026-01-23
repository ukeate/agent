/**
 * LoRA训练服务
 *
 * 管理LoRA/QLoRA微调训练任务
 */

import apiClient from './apiClient'

import { logger } from '../utils/logger'
// ==================== 接口定义 ====================

export interface LoRAConfig {
  r: number // LoRA rank
  lora_alpha: number
  lora_dropout: number
  target_modules: string[]
  use_rslora?: boolean
  use_dora?: boolean
  init_lora_weights?: 'gaussian' | 'pissa' | boolean
  bias?: 'none' | 'all' | 'lora_only'
}

export interface QLoRAConfig extends LoRAConfig {
  bnb_4bit_compute_dtype?: string
  bnb_4bit_quant_type?: 'nf4' | 'fp4'
  bnb_4bit_use_double_quant?: boolean
  load_in_4bit?: boolean
  load_in_8bit?: boolean
}

export interface TrainingConfig {
  base_model: string
  model_type: 'llama' | 'mistral' | 'qwen' | 'gpt' | 'custom'
  num_train_epochs: number
  per_device_train_batch_size: number
  per_device_eval_batch_size: number
  gradient_accumulation_steps: number
  learning_rate: number
  warmup_ratio: number
  lr_scheduler_type: string
  gradient_checkpointing: boolean
  fp16?: boolean
  bf16?: boolean
  optim: string
  weight_decay: number
  max_grad_norm: number
  max_seq_length: number
  save_steps: number
  eval_steps: number
  logging_steps: number
  save_total_limit: number
  load_best_model_at_end: boolean
  metric_for_best_model: string
  greater_is_better: boolean
}

export interface TrainingJob {
  job_id: string
  name: string
  status:
    | 'created'
    | 'pending'
    | 'running'
    | 'completed'
    | 'failed'
    | 'cancelled'
  base_model: string
  lora_config: LoRAConfig
  training_config: TrainingConfig
  created_at: string
  started_at?: string
  completed_at?: string
  error_message?: string
}

export interface TrainingProgress {
  job_id: string
  current_epoch: number
  total_epochs: number
  current_step: number
  total_steps: number
  progress_percent: number
  train_loss: number
  eval_loss?: number
  learning_rate: number
  grad_norm?: number
  gpu_utilization?: number
  memory_usage_gb?: number
  samples_per_second?: number
  training_time_seconds?: number
  estimated_time_remaining?: string
}

export interface TrainingMetric {
  step: number
  epoch: number
  train_loss?: number
  eval_loss?: number
  learning_rate: number
  grad_norm?: number
  perplexity?: number
  bleu_score?: number
  rouge_score?: number
  timestamp: string
}

export interface LoRALayer {
  layer_name: string
  module_type: string
  rank: number
  alpha: number
  dropout: number
  trainable: boolean
  param_count: number
  weight_norm?: number
  gradient_norm?: number
}

export interface ModelInfo {
  base_model: string
  model_size: string
  total_params: number
  trainable_params: number
  trainable_percentage: number
  lora_layers: LoRALayer[]
  memory_footprint_mb: number
  disk_size_mb: number
}

export interface CheckpointInfo {
  checkpoint_id: string
  job_id: string
  step: number
  epoch: number
  train_loss: number
  eval_loss?: number
  best_metric?: number
  created_at: string
  file_path: string
  file_size_mb: number
  is_best: boolean
}

// ==================== 服务类 ====================

class LoRATrainingService {
  private baseUrl = '/lora-training'

  // ==================== 任务管理 ====================

  /**
   * 创建LoRA训练任务
   */
  async createTrainingJob(params: {
    name: string
    base_model: string
    dataset_id: string
    lora_config: LoRAConfig
    training_config: TrainingConfig
    use_qlora?: boolean
  }): Promise<TrainingJob> {
    try {
      const response = await apiClient.post(`${this.baseUrl}/jobs`, params)
      return response.data
    } catch (error) {
      logger.error('创建LoRA训练任务失败:', error)
      throw error
    }
  }

  /**
   * 获取训练任务列表
   */
  async getTrainingJobs(filters?: {
    status?: string
    base_model?: string
    created_after?: string
    created_before?: string
  }): Promise<TrainingJob[]> {
    const response = await apiClient.get(`${this.baseUrl}/jobs`, {
      params: filters,
    })
    return response.data
  }

  /**
   * 获取单个训练任务详情
   */
  async getTrainingJob(jobId: string): Promise<TrainingJob> {
    const response = await apiClient.get(`${this.baseUrl}/jobs/${jobId}`)
    return response.data
  }

  /**
   * 启动训练任务
   */
  async startTraining(
    jobId: string
  ): Promise<{ success: boolean; message: string }> {
    const response = await apiClient.post(`${this.baseUrl}/jobs/${jobId}/start`)
    return response.data
  }

  /**
   * 停止训练任务
   */
  async stopTraining(
    jobId: string
  ): Promise<{ success: boolean; message: string }> {
    const response = await apiClient.post(`${this.baseUrl}/jobs/${jobId}/stop`)
    return response.data
  }

  /**
   * 暂停训练任务
   */
  async pauseTraining(
    jobId: string
  ): Promise<{ success: boolean; message: string }> {
    const response = await apiClient.post(`${this.baseUrl}/jobs/${jobId}/pause`)
    return response.data
  }

  /**
   * 恢复训练任务
   */
  async resumeTraining(
    jobId: string
  ): Promise<{ success: boolean; message: string }> {
    const response = await apiClient.post(
      `${this.baseUrl}/jobs/${jobId}/resume`
    )
    return response.data
  }

  // ==================== 训练监控 ====================

  /**
   * 获取训练进度
   */
  async getTrainingProgress(jobId: string): Promise<TrainingProgress> {
    const response = await apiClient.get(
      `${this.baseUrl}/jobs/${jobId}/progress`
    )
    return response.data
  }

  /**
   * 获取训练指标历史
   */
  async getTrainingMetrics(
    jobId: string,
    options?: {
      start_step?: number
      end_step?: number
      metric_names?: string[]
    }
  ): Promise<TrainingMetric[]> {
    const response = await apiClient.get(
      `${this.baseUrl}/jobs/${jobId}/metrics`,
      {
        params: options,
      }
    )
    return response.data
  }

  /**
   * 获取实时日志
   */
  async getTrainingLogs(
    jobId: string,
    options?: {
      tail?: number
      follow?: boolean
    }
  ): Promise<string[]> {
    const response = await apiClient.get(`${this.baseUrl}/jobs/${jobId}/logs`, {
      params: options,
    })
    return response.data
  }

  // ==================== 模型信息 ====================

  /**
   * 获取模型信息
   */
  async getModelInfo(jobId: string): Promise<ModelInfo> {
    const response = await apiClient.get(
      `${this.baseUrl}/jobs/${jobId}/model-info`
    )
    return response.data
  }

  /**
   * 获取LoRA层详情
   */
  async getLoRALayers(jobId: string): Promise<LoRALayer[]> {
    const response = await apiClient.get(
      `${this.baseUrl}/jobs/${jobId}/lora-layers`
    )
    return response.data
  }

  // ==================== 检查点管理 ====================

  /**
   * 获取检查点列表
   */
  async getCheckpoints(jobId: string): Promise<CheckpointInfo[]> {
    const response = await apiClient.get(
      `${this.baseUrl}/jobs/${jobId}/checkpoints`
    )
    return response.data
  }

  /**
   * 保存检查点
   */
  async saveCheckpoint(
    jobId: string,
    checkpointName?: string
  ): Promise<CheckpointInfo> {
    const response = await apiClient.post(
      `${this.baseUrl}/jobs/${jobId}/checkpoints`,
      {
        name: checkpointName,
      }
    )
    return response.data
  }

  /**
   * 加载检查点
   */
  async loadCheckpoint(
    jobId: string,
    checkpointId: string
  ): Promise<{ success: boolean; message: string }> {
    const response = await apiClient.post(
      `${this.baseUrl}/jobs/${jobId}/checkpoints/${checkpointId}/load`
    )
    return response.data
  }

  // ==================== 配置管理 ====================

  /**
   * 更新LoRA配置
   */
  async updateLoRAConfig(
    jobId: string,
    config: Partial<LoRAConfig>
  ): Promise<{ success: boolean; message: string }> {
    const response = await apiClient.patch(
      `${this.baseUrl}/jobs/${jobId}/lora-config`,
      config
    )
    return response.data
  }

  /**
   * 更新训练配置
   */
  async updateTrainingConfig(
    jobId: string,
    config: Partial<TrainingConfig>
  ): Promise<{ success: boolean; message: string }> {
    const response = await apiClient.patch(
      `${this.baseUrl}/jobs/${jobId}/training-config`,
      config
    )
    return response.data
  }

  // ==================== 模型导出 ====================

  /**
   * 导出LoRA适配器
   */
  async exportLoRAAdapter(
    jobId: string,
    format: 'pytorch' | 'safetensors' | 'onnx'
  ): Promise<{
    success: boolean
    download_url: string
    file_size_mb: number
  }> {
    const response = await apiClient.post(
      `${this.baseUrl}/jobs/${jobId}/export`,
      { format }
    )
    return response.data
  }

  /**
   * 合并LoRA到基础模型
   */
  async mergeLoRAWeights(
    jobId: string,
    options?: {
      merge_ratio?: number
      output_format?: 'pytorch' | 'safetensors'
    }
  ): Promise<{
    success: boolean
    merged_model_path: string
    model_size_gb: number
  }> {
    const response = await apiClient.post(
      `${this.baseUrl}/jobs/${jobId}/merge`,
      options
    )
    return response.data
  }

  // ==================== 推理测试 ====================

  /**
   * 测试推理
   */
  async testInference(
    jobId: string,
    input_text: string,
    generation_config?: {
      max_new_tokens?: number
      temperature?: number
      top_p?: number
      top_k?: number
      do_sample?: boolean
    }
  ): Promise<{
    input: string
    output: string
    generation_time_ms: number
    tokens_generated: number
  }> {
    const response = await apiClient.post(
      `${this.baseUrl}/jobs/${jobId}/inference`,
      {
        input_text,
        generation_config,
      }
    )
    return response.data
  }
}

// 导出服务实例
export const loraTrainingService = new LoRATrainingService()
