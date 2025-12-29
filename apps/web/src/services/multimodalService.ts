/**
 * 多模态处理服务
 * 
 * 提供多模态内容处理相关的API调用服务
 */

import apiClient from './apiClient';

// 内容类型
export enum ContentType {
  IMAGE = 'image',
  DOCUMENT = 'document',
  VIDEO = 'video',
  AUDIO = 'audio'
}

// 模型优先级
export enum ModelPriority {
  SPEED = 'speed',
  BALANCED = 'balanced',
  QUALITY = 'quality'
}

// 模型复杂度
export enum ModelComplexity {
  LOW = 'low',
  MEDIUM = 'medium',
  HIGH = 'high'
}

// 处理状态
export enum ProcessingStatus {
  PENDING = 'pending',
  PROCESSING = 'processing',
  COMPLETED = 'completed',
  FAILED = 'failed'
}

// 文件上传响应
export interface FileUploadResponse {
  content_id: string;
  content_type: string;
  file_size: number;
  mime_type: string;
  metadata?: Record<string, any>;
}

export interface TokenUsage {
  prompt_tokens?: number;
  completion_tokens?: number;
  total_tokens?: number;
}

// 处理请求
export interface ProcessingRequest {
  content_id: string;
  content_type: string;
  priority?: string;
  complexity?: string;
  max_tokens?: number;
  temperature?: number;
  enable_cache?: boolean;
  extract_text?: boolean;
  extract_objects?: boolean;
  extract_sentiment?: boolean;
}

// 处理响应
export interface ProcessingResponse {
  content_id: string;
  status: string;
  extracted_data: any;
  structured_data?: any;
  confidence_score: number;
  processing_time: number;
  model_used: string;
  tokens_used: number | TokenUsage;
  error_message?: string;
}

// 批量处理请求
export interface BatchProcessingRequest {
  content_ids: string[];
  priority?: string;
  complexity?: string;
  max_tokens?: number;
}

// 批量处理响应
export interface BatchProcessingResponse {
  batch_id: string;
  content_ids: string[];
  status: string;
  total_items: number;
  completed_items: number;
}

// 处理状态响应
export interface ProcessingStatusResponse {
  content_id: string;
  status: string;
  confidence_score?: number;
  processing_time?: number;
  model_used?: string;
  tokens_used?: number | TokenUsage;
  error_message?: string;
}

// 队列状态
export interface QueueStatus {
  total_jobs: number;
  pending_jobs: number;
  processing_jobs: number;
  completed_jobs: number;
  failed_jobs: number;
  average_wait_time: number;
  average_processing_time: number;
}

export interface MultimodalModelConfig {
  name: string;
  cost_per_1k_tokens: {
    input: number;
    output: number;
  };
  max_tokens?: number;
  max_image_size?: number;
  capabilities?: string[];
  best_for?: string[];
  supports_vision?: boolean;
  supports_file_upload?: boolean;
}

// 图像分析结果
export interface ImageAnalysisResult {
  extracted_data: {
    description?: string;
    objects?: string[];
    text?: string;
    keyPoints?: string[];
  };
  structured_data?: any;
  model_used: string;
  tokens_used: number | TokenUsage;
  cost: number;
  processing_time: number;
}

class MultimodalService {
  private baseUrl = '/multimodal';
  private modelPricing: Record<string, { input: number; output: number }> = {};
  private pricingLoaded = false;
  private pricingLoading: Promise<void> | null = null;

  /**
   * 上传文件
   */
  async uploadFile(file: File): Promise<FileUploadResponse> {
    const formData = new FormData();
    formData.append('file', file);

    const response = await apiClient.post(`${this.baseUrl}/upload`, formData, {
      headers: {
        'Content-Type': 'multipart/form-data'
      }
    });
    return response.data;
  }

  /**
   * 处理内容
   */
  async processContent(request: ProcessingRequest): Promise<ProcessingResponse> {
    const response = await apiClient.post(`${this.baseUrl}/process`, request);
    return response.data;
  }

  /**
   * 批量处理
   */
  async processBatch(request: BatchProcessingRequest): Promise<BatchProcessingResponse> {
    const response = await apiClient.post(`${this.baseUrl}/process/batch`, request);
    return response.data;
  }

  /**
   * 获取处理状态
   */
  async getProcessingStatus(contentId: string): Promise<ProcessingStatusResponse> {
    const response = await apiClient.get(`${this.baseUrl}/status/${contentId}`);
    return response.data;
  }

  /**
   * 获取队列状态
   */
  async getQueueStatus(): Promise<QueueStatus> {
    const response = await apiClient.get(`${this.baseUrl}/queue/status`);
    return response.data;
  }

  async getModelConfigs(): Promise<MultimodalModelConfig[]> {
    const response = await apiClient.get(`${this.baseUrl}/models`);
    return response.data?.models || [];
  }

  async ensureModelPricing(): Promise<void> {
    if (this.pricingLoaded) return;
    if (this.pricingLoading) return this.pricingLoading;
    this.pricingLoading = this.getModelConfigs()
      .then((models) => {
        const pricing: Record<string, { input: number; output: number }> = {};
        models.forEach((model) => {
          if (model.cost_per_1k_tokens) {
            pricing[model.name] = {
              input: Number(model.cost_per_1k_tokens.input) || 0,
              output: Number(model.cost_per_1k_tokens.output) || 0,
            };
          }
        });
        this.modelPricing = pricing;
        this.pricingLoaded = true;
      })
      .finally(() => {
        this.pricingLoading = null;
      });
    return this.pricingLoading;
  }

  getPricingMap(): Record<string, { input: number; output: number }> {
    return this.modelPricing;
  }

  /**
   * 分析图像（快速分析，不保存）
   */
  async analyzeImage(
    file: File,
    prompt: string = '分析这张图像',
    extractText: boolean = true,
    extractObjects: boolean = true,
    priority: ModelPriority = ModelPriority.BALANCED
  ): Promise<ImageAnalysisResult> {
    const formData = new FormData();
    formData.append('file', file);

    const params = {
      prompt,
      extract_text: extractText,
      extract_objects: extractObjects,
      priority
    };

    const response = await apiClient.post(`${this.baseUrl}/analyze/image`, formData, {
      params,
      headers: {
        'Content-Type': 'multipart/form-data'
      }
    });
    return response.data;
  }

  /**
   * 删除文件
   */
  async deleteFile(contentId: string): Promise<{ message: string; content_id: string }> {
    const response = await apiClient.delete(`${this.baseUrl}/file/${contentId}`);
    return response.data;
  }

  /**
   * 计算Token成本
   */
  calculateCost(model: string, inputTokens: number, outputTokens: number): number {
    const modelPricing = this.modelPricing[model];
    if (!modelPricing) return 0;
    return (inputTokens * modelPricing.input + outputTokens * modelPricing.output) / 1000;
  }

  /**
   * 获取支持的文件类型
   */
  getSupportedFileTypes(): string[] {
    return [
      '.jpg', '.jpeg', '.png', '.webp', '.gif',  // 图像
      '.pdf', '.txt', '.docx', '.md', '.csv', '.xlsx',  // 文档
      '.mp4', '.avi', '.mov', '.mkv', '.webm',  // 视频
      '.mp3', '.wav', '.flac', '.ogg', '.m4a'  // 音频
    ];
  }

  /**
   * 验证文件
   */
  validateFile(file: File): { valid: boolean; error?: string } {
    // 检查文件大小（20MB限制）
    if (file.size > 20 * 1024 * 1024) {
      return { valid: false, error: '文件大小超过20MB限制' };
    }

    // 检查文件类型
    const fileName = file.name.toLowerCase();
    const supportedTypes = this.getSupportedFileTypes();
    const hasValidExtension = supportedTypes.some(ext => fileName.endsWith(ext));
    
    if (!hasValidExtension) {
      return { valid: false, error: '不支持的文件类型' };
    }

    return { valid: true };
  }
}

export const multimodalService = new MultimodalService();
