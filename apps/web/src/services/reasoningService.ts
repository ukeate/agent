import apiClient from './apiClient';
import { apiFetch, buildApiUrl } from '../utils/apiBase';
import { consumeSseJson } from '../utils/sse';

import { logger } from '../utils/logger'
interface ReasoningRequest {
  problem: string;
  strategy: 'ZERO_SHOT' | 'FEW_SHOT' | 'AUTO_COT';
  context?: string;
  max_steps: number;
  stream: boolean;
  enable_branching: boolean;
  examples?: Array<{
    problem: string;
    reasoning: string;
    answer: string;
  }>;
}

interface ReasoningChain {
  id: string;
  strategy: string;
  problem: string;
  context?: string;
  conclusion?: string;
  confidence_score?: number;
  created_at: string;
  completed_at?: string;
  steps: Array<{
    id: string;
    step_number: number;
    step_type: string;
    content: string;
    reasoning: string;
    confidence: number;
    duration_ms?: number;
  }>;
  branches?: Array<{
    id: string;
    parent_step_id: string;
    reason: string;
    created_at: string;
  }>;
}

interface ReasoningStreamChunk {
  chain_id: string;
  step_number: number;
  step_type: string;
  content: string;
  reasoning: string;
  confidence: number;
  is_final: boolean;
}

interface ReasoningValidation {
  step_id: string;
  is_valid: boolean;
  consistency_score: number;
  issues: string[];
  suggestions: string[];
}

interface ReasoningStats {
  total_chains: number;
  completed_chains: number;
  avg_confidence: number;
  avg_steps: number;
  strategy_distribution: Record<string, number>;
  quality_metrics: {
    high_quality: number;
    medium_quality: number;
    low_quality: number;
  };
}

export class ReasoningService {
  private baseUrl = '/reasoning';

  /**
   * 执行推理
   */
  async executeReasoning(request: ReasoningRequest): Promise<ReasoningChain> {
    try {
      const response = await apiClient.post(`${this.baseUrl}/chain`, request);
      return response.data;
    } catch (error) {
      logger.error('执行推理失败:', error);
      throw new Error('推理执行失败，请检查网络连接或稍后重试');
    }
  }

  /**
   * 流式推理
   */
  async streamReasoning(
    request: ReasoningRequest, 
    onChunk: (chunk: ReasoningStreamChunk) => void
  ): Promise<void> {
    try {
      const token = localStorage.getItem('access_token');
      const headers: HeadersInit = {
        'Content-Type': 'application/json',
        'Accept': 'text/event-stream',
      };
      if (token) headers['Authorization'] = `Bearer ${token}`;

      const response = await apiFetch(buildApiUrl(`${this.baseUrl}/stream`), {
        method: 'POST',
        headers,
        body: JSON.stringify(request)
      });

      await consumeSseJson<ReasoningStreamChunk>(
        response,
        (chunk) => {
          onChunk(chunk);
        },
        {
          onParseError: (error, raw) => {
            logger.warn('解析SSE数据失败:', error, 'data:', raw);
          },
        }
      );
    } catch (error) {
      logger.error('流式推理失败:', error);
      throw new Error('流式推理失败，请检查网络连接或稍后重试');
    }
  }

  /**
   * 获取推理链
   */
  async getReasoningChain(chainId: string): Promise<ReasoningChain> {
    try {
      const response = await apiClient.get(`${this.baseUrl}/chain/${chainId}`);
      return response.data;
    } catch (error) {
      logger.error('获取推理链失败:', error);
      throw new Error('获取推理链失败，请检查网络连接或稍后重试');
    }
  }

  /**
   * 获取推理历史
   */
  async getReasoningHistory(limit = 20, offset = 0): Promise<ReasoningChain[]> {
    try {
      const response = await apiClient.get(`${this.baseUrl}/history`, {
        params: { limit, offset }
      });
      return response.data;
    } catch (error) {
      logger.error('获取推理历史失败:', error);
      throw new Error('获取推理历史失败，请检查网络连接或稍后重试');
    }
  }

  /**
   * 验证推理链
   */
  async validateChain(chainId: string, stepNumber?: number): Promise<ReasoningValidation> {
    try {
      const response = await apiClient.post(`${this.baseUrl}/chain/${chainId}/validate`, {
        step_number: stepNumber
      });
      return response.data;
    } catch (error) {
      logger.error('验证推理链失败:', error);
      throw new Error('验证推理链失败，请检查网络连接或稍后重试');
    }
  }

  /**
   * 创建推理分支
   */
  async createBranch(
    chainId: string, 
    parentStepNumber: number, 
    reason: string
  ): Promise<string> {
    try {
      const response = await apiClient.post(`${this.baseUrl}/chain/${chainId}/branch`, {
        parent_step_number: parentStepNumber,
        reason
      });
      return response.data.branch_id;
    } catch (error) {
      logger.error('创建分支失败:', error);
      throw new Error('创建分支失败，请检查网络连接或稍后重试');
    }
  }

  /**
   * 恢复推理链
   */
  async recoverChain(chainId: string, strategy?: string): Promise<boolean> {
    try {
      const response = await apiClient.post(`${this.baseUrl}/chain/${chainId}/recover`, {
        strategy
      });
      return response.data.success;
    } catch (error) {
      logger.error('恢复推理链失败:', error);
      throw new Error('恢复推理链失败，请检查网络连接或稍后重试');
    }
  }

  /**
   * 获取推理统计
   */
  async getReasoningStats(): Promise<ReasoningStats> {
    try {
      const response = await apiClient.get(`${this.baseUrl}/stats`);
      return response.data;
    } catch (error) {
      logger.error('获取推理统计失败:', error);
      throw new Error('获取推理统计失败，请检查网络连接或稍后重试');
    }
  }

  /**
   * 删除推理链
   */
  async deleteReasoningChain(chainId: string): Promise<void> {
    try {
      await apiClient.delete(`${this.baseUrl}/chain/${chainId}`);
    } catch (error) {
      logger.error('删除推理链失败:', error);
      throw new Error('删除推理链失败，请检查网络连接或稍后重试');
    }
  }

  /**
   * 执行推理 - /api/v1/reasoning/execute
   */
  async execute(request: ReasoningRequest): Promise<ReasoningChain> {
    try {
      const response = await apiClient.post(`${this.baseUrl}/execute`, request);
      return response.data;
    } catch (error) {
      logger.error('执行推理失败:', error);
      throw new Error('执行推理失败，请检查网络连接或稍后重试');
    }
  }

  /**
   * 获取推理策略列表 - /api/v1/reasoning/strategies
   */
  async getStrategies(): Promise<Array<{
    id: string;
    name: string;
    description: string;
    enabled: boolean;
  }>> {
    try {
      const response = await apiClient.get(`${this.baseUrl}/strategies`);
      return response.data;
    } catch (error) {
      logger.error('获取推理策略失败:', error);
      throw new Error('获取推理策略失败，请检查网络连接或稍后重试');
    }
  }

  /**
   * 获取推理详情 - /api/v1/reasoning/{reasoning_id}
   */
  async getReasoningDetails(reasoningId: string): Promise<ReasoningChain> {
    try {
      const response = await apiClient.get(`${this.baseUrl}/${reasoningId}`);
      return response.data;
    } catch (error) {
      logger.error('获取推理详情失败:', error);
      throw new Error('获取推理详情失败，请检查网络连接或稍后重试');
    }
  }

  /**
   * 获取推理统计信息 - /api/v1/reasoning/statistics
   */
  async getStatistics(): Promise<ReasoningStats> {
    try {
      const response = await apiClient.get(`${this.baseUrl}/statistics`);
      return response.data;
    } catch (error) {
      logger.error('获取推理统计失败:', error);
      throw new Error('获取推理统计失败，请检查网络连接或稍后重试');
    }
  }

  /**
   * 验证推理 - /api/v1/reasoning/validate
   */
  async validate(reasoningChain: ReasoningChain): Promise<ReasoningValidation> {
    try {
      const response = await apiClient.post(`${this.baseUrl}/validate`, reasoningChain);
      return response.data;
    } catch (error) {
      logger.error('验证推理失败:', error);
      throw new Error('验证推理失败，请检查网络连接或稍后重试');
    }
  }

}

export const reasoningService = new ReasoningService();
