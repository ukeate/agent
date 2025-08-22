import { apiClient } from './apiClient';

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
  private baseUrl = '/api/v1/reasoning';

  /**
   * 执行推理
   */
  async executeReasoning(request: ReasoningRequest): Promise<ReasoningChain> {
    try {
      const response = await apiClient.post(`${this.baseUrl}/chain`, request);
      return response.data;
    } catch (error) {
      console.error('执行推理失败:', error);
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
      const response = await fetch(`${this.baseUrl}/stream`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${localStorage.getItem('access_token')}`
        },
        body: JSON.stringify(request)
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      if (!response.body) {
        throw new Error('响应体为空');
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

      try {
        while (true) {
          const { done, value } = await reader.read();
          
          if (done) break;

          buffer += decoder.decode(value, { stream: true });
          
          // 处理SSE消息
          const lines = buffer.split('\n');
          buffer = lines.pop() || ''; // 保留最后一个不完整的行

          for (const line of lines) {
            if (line.startsWith('data: ')) {
              const data = line.slice(6).trim();
              
              if (data === '[DONE]') {
                return;
              }

              try {
                const chunk = JSON.parse(data) as ReasoningStreamChunk;
                onChunk(chunk);
              } catch (parseError) {
                console.warn('解析SSE数据失败:', parseError, 'data:', data);
              }
            }
          }
        }
      } finally {
        reader.releaseLock();
      }
    } catch (error) {
      console.error('流式推理失败:', error);
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
      console.error('获取推理链失败:', error);
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
      console.error('获取推理历史失败:', error);
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
      console.error('验证推理链失败:', error);
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
      console.error('创建分支失败:', error);
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
      console.error('恢复推理链失败:', error);
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
      console.error('获取推理统计失败:', error);
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
      console.error('删除推理链失败:', error);
      throw new Error('删除推理链失败，请检查网络连接或稍后重试');
    }
  }
}

export const reasoningService = new ReasoningService();