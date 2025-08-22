import apiClient from './apiClient';

export interface UploadFileResponse {
  content_id: string;
  content_type: string;
  file_size: number;
  mime_type?: string;
  metadata?: Record<string, any>;
}

export interface ProcessingRequest {
  contentId: string;
  contentType: string;
  priority?: string;
  complexity?: string;
  maxTokens?: number;
  temperature?: number;
  enableCache?: boolean;
  extractText?: boolean;
  extractObjects?: boolean;
  extractSentiment?: boolean;
}

export interface ProcessingResponse {
  content_id: string;
  status: string;
  extracted_data: Record<string, any>;
  structured_data?: Record<string, any>;
  confidence_score: number;
  processing_time: number;
  model_used?: string;
  tokens_used?: {
    prompt_tokens: number;
    completion_tokens: number;
    total_tokens: number;
  };
  cost?: number;
  error_message?: string;
}

export interface BatchProcessingRequest {
  contentIds: string[];
  priority?: string;
  complexity?: string;
  maxTokens?: number;
}

export interface BatchProcessingResponse {
  batch_id: string;
  content_ids: string[];
  status: string;
  total_items: number;
  completed_items: number;
}

export interface ProcessingStatus {
  content_id: string;
  status: string;
  confidence_score?: number;
  processing_time?: number;
  model_used?: string;
  tokens_used?: Record<string, number>;
  error_message?: string;
}

export interface QueueStatus {
  is_running: boolean;
  active_tasks: number;
  queued_tasks: number;
  completed_tasks: number;
  failed_tasks: number;
}

class MultimodalService {
  private baseURL = '/api/v1/multimodal';

  /**
   * 上传文件
   */
  async uploadFile(file: File): Promise<UploadFileResponse> {
    const formData = new FormData();
    formData.append('file', file);

    const response = await apiClient.post(`${this.baseURL}/upload`, formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return response.data;
  }

  /**
   * 处理内容
   */
  async processContent(request: ProcessingRequest): Promise<ProcessingResponse> {
    // 转换为API格式
    const apiRequest = {
      content_id: request.contentId,
      content_type: request.contentType,
      priority: request.priority,
      complexity: request.complexity,
      max_tokens: request.maxTokens,
      temperature: request.temperature,
      enable_cache: request.enableCache,
      extract_text: request.extractText,
      extract_objects: request.extractObjects,
      extract_sentiment: request.extractSentiment,
    };

    const response = await apiClient.post(`${this.baseURL}/process`, apiRequest);
    return response.data;
  }

  /**
   * 批量处理
   */
  async processBatch(request: BatchProcessingRequest): Promise<BatchProcessingResponse> {
    const apiRequest = {
      content_ids: request.contentIds,
      priority: request.priority,
      complexity: request.complexity,
      max_tokens: request.maxTokens,
    };

    const response = await apiClient.post(`${this.baseURL}/process/batch`, apiRequest);
    return response.data;
  }

  /**
   * 获取处理状态
   */
  async getProcessingStatus(contentId: string): Promise<ProcessingStatus> {
    const response = await apiClient.get(`${this.baseURL}/status/${contentId}`);
    return response.data;
  }

  /**
   * 获取队列状态
   */
  async getQueueStatus(): Promise<QueueStatus> {
    const response = await apiClient.get(`${this.baseURL}/queue/status`);
    return response.data;
  }

  /**
   * 直接分析图像（不保存文件）
   */
  async analyzeImageDirect(
    file: File,
    prompt: string,
    options?: {
      extractText?: boolean;
      extractObjects?: boolean;
      extractSentiment?: boolean;
      priority?: string;
    }
  ): Promise<any> {
    const formData = new FormData();
    formData.append('file', file);
    
    // 添加查询参数
    const params = new URLSearchParams();
    params.append('prompt', prompt);
    if (options?.extractText !== undefined) {
      params.append('extract_text', String(options.extractText));
    }
    if (options?.extractObjects !== undefined) {
      params.append('extract_objects', String(options.extractObjects));
    }
    if (options?.priority) {
      params.append('priority', options.priority);
    }

    const response = await apiClient.post(
      `${this.baseURL}/analyze/image?${params.toString()}`,
      formData,
      {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      }
    );
    return response.data;
  }

  /**
   * 删除文件
   */
  async deleteFile(contentId: string): Promise<void> {
    await apiClient.delete(`${this.baseURL}/file/${contentId}`);
  }

  /**
   * 获取支持的文件格式
   */
  getSupportedFormats(): Record<string, string[]> {
    return {
      image: ['jpg', 'jpeg', 'png', 'webp', 'gif'],
      document: ['pdf', 'txt', 'docx', 'md', 'csv'],
      video: ['mp4', 'avi', 'mov', 'mkv', 'webm'],
      audio: ['mp3', 'wav', 'flac', 'ogg', 'm4a'],
    };
  }

  /**
   * 获取模型配置
   */
  getModelConfigs(): Record<string, any> {
    return {
      'gpt-4o': {
        name: 'GPT-4o',
        maxTokens: 4096,
        costPerKTokens: { input: 5, output: 15 },
        capabilities: ['text', 'image', 'pdf'],
      },
      'gpt-4o-mini': {
        name: 'GPT-4o Mini',
        maxTokens: 16384,
        costPerKTokens: { input: 0.15, output: 0.6 },
        capabilities: ['text', 'image', 'pdf'],
      },
      'gpt-5': {
        name: 'GPT-5',
        maxTokens: 8192,
        costPerKTokens: { input: 12.5, output: 25 },
        capabilities: ['text', 'image', 'pdf', 'video'],
      },
      'gpt-5-nano': {
        name: 'GPT-5 Nano',
        maxTokens: 128000,
        costPerKTokens: { input: 0.05, output: 0.4 },
        capabilities: ['text', 'image'],
      },
    };
  }

  /**
   * 计算预估成本
   */
  calculateEstimatedCost(
    model: string,
    inputTokens: number,
    outputTokens: number
  ): number {
    const config = this.getModelConfigs()[model];
    if (!config) return 0;

    const inputCost = (inputTokens / 1000) * config.costPerKTokens.input;
    const outputCost = (outputTokens / 1000) * config.costPerKTokens.output;
    return inputCost + outputCost;
  }
}

export const multimodalService = new MultimodalService();