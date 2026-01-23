/**
 * 情感建模API客户端
 */

import { ApiClient, type ApiClientConfig } from './base';
import type { ApiResponse } from '../types/api';

export interface EmotionState {
  id: string;
  emotion: string;
  intensity: number;
  valence: number;
  arousal: number;
  dominance: number;
  timestamp: string;
  confidence: number;
  triggers?: string[];
  context?: Record<string, unknown>;
  source?: string;
  session_id?: string;
}

export interface EmotionStateInput {
  emotion: string;
  intensity?: number;
  valence?: number;
  arousal?: number;
  dominance?: number;
  confidence?: number;
  timestamp?: string;
  triggers?: string[];
  context?: Record<string, unknown>;
  source?: string;
  session_id?: string;
}

export interface PersonalityProfile {
  user_id: string;
  emotional_traits: Record<string, number>;
  baseline_emotions: Record<string, number>;
  emotion_volatility: number;
  recovery_rate: number;
  dominant_emotions: string[];
  sample_count: number;
  confidence_score: number;
  created_at: string;
  updated_at: string;
}

export interface EmotionPrediction {
  user_id: string;
  time_horizon_hours: number;
  predictions: Array<{
    emotion: string;
    probability: number;
    intensity_range: [number, number];
  }>;
  confidence: number;
  factors: Record<string, number>;
  timestamp: string;
}

export interface EmotionAnalytics {
  user_id: string;
  period_days: number;
  temporal_patterns: {
    best_hours: Array<[number, number]>;
    worst_hours: Array<[number, number]>;
    weekly_patterns: Record<string, number>;
    monthly_patterns: Record<string, number>;
  };
  emotion_distribution: Record<string, number>;
  volatility: {
    overall_volatility: number;
    valence_volatility: number;
    arousal_volatility: number;
    dominance_volatility: number;
  };
  clusters: Array<{
    name: string;
    emotions: string[];
    frequency: number;
  }>;
  patterns: Array<{
    pattern_type: string;
    description: string;
    frequency: number;
    confidence: number;
  }>;
  recovery_analysis: {
    average_recovery_time: number;
    recovery_rate: number;
    triggers: Record<string, number>;
  };
}

export interface EmotionStatistics {
  user_id: string;
  period_start: string;
  period_end: string;
  total_records: number;
  unique_emotions: number;
  average_intensity: number;
  average_valence: number;
  average_arousal: number;
  average_dominance: number;
  most_frequent_emotion: string;
  emotion_counts: Record<string, number>;
  daily_averages: Record<string, number>;
}

export class EmotionModelingApiClient extends ApiClient {
  private readonly BASE_PATH = '/api/v1/emotion';

  constructor(config: ApiClientConfig) {
    super(config);
  }

  // 情感状态记录
  async recordEmotionState(data: EmotionStateInput): Promise<ApiResponse<EmotionState>> {
    return this.post<EmotionState>(`${this.BASE_PATH}/state`, data);
  }

  // 获取最新情感状态
  async getLatestEmotionState(): Promise<ApiResponse<EmotionState>> {
    return this.get<EmotionState>(`${this.BASE_PATH}/state/latest`);
  }

  // 获取情感历史
  async getEmotionHistory(params: {
    limit?: number;
    start_date?: string;
    end_date?: string;
    emotions?: string[];
  } = {}): Promise<ApiResponse<EmotionState[]>> {
    const searchParams = new URLSearchParams();
    
    if (params.limit) {
      searchParams.set('limit', params.limit.toString());
    }
    if (params.start_date) {
      searchParams.set('start_date', params.start_date);
    }
    if (params.end_date) {
      searchParams.set('end_date', params.end_date);
    }
    if (params.emotions) {
      params.emotions.forEach(emotion => searchParams.append('emotions', emotion));
    }

    const queryString = searchParams.toString();
    const url = queryString ? `${this.BASE_PATH}/state/history?${queryString}` : `${this.BASE_PATH}/state/history`;
    
    return this.get<EmotionState[]>(url);
  }

  // 情感预测
  async predictEmotions(time_horizon_hours: number = 1): Promise<ApiResponse<EmotionPrediction>> {
    return this.post<EmotionPrediction>(`${this.BASE_PATH}/predict`, {
      time_horizon_hours
    });
  }

  // 情感分析报告
  async getEmotionAnalytics(days_back: number = 30): Promise<ApiResponse<EmotionAnalytics>> {
    return this.post<EmotionAnalytics>(`${this.BASE_PATH}/analytics`, {
      days_back
    });
  }

  // 获取个性画像
  async getPersonalityProfile(): Promise<ApiResponse<PersonalityProfile>> {
    return this.get<PersonalityProfile>(`${this.BASE_PATH}/profile`);
  }

  // 检测情感模式
  async detectPatterns(): Promise<ApiResponse<unknown>> {
    return this.get(`${this.BASE_PATH}/patterns`);
  }

  // 获取情感聚类分析
  async getEmotionClusters(): Promise<ApiResponse<unknown>> {
    return this.get(`${this.BASE_PATH}/clusters`);
  }

  // 获取情感转换分析
  async getTransitionAnalysis(): Promise<ApiResponse<unknown>> {
    return this.get(`${this.BASE_PATH}/transitions`);
  }

  // 获取情感统计数据
  async getEmotionStatistics(days: number = 30): Promise<ApiResponse<EmotionStatistics>> {
    return this.get<EmotionStatistics>(`${this.BASE_PATH}/statistics?days=${days}`);
  }

  // 导出数据
  async exportData(): Promise<ApiResponse<unknown>> {
    return this.get(`${this.BASE_PATH}/export`);
  }

  // 删除用户数据
  async deleteUserData(): Promise<ApiResponse<{message: string}>> {
    return this.delete(`${this.BASE_PATH}/data`);
  }

  // 获取系统状态
  async getSystemStatus(): Promise<ApiResponse<unknown>> {
    return this.get(`${this.BASE_PATH}/status`);
  }

  // WebSocket连接（用于实时更新）
  connectRealtime(userId: string, callbacks: {
    onOpen?: () => void;
    onMessage?: (data: unknown) => void;
    onClose?: () => void;
    onError?: (error: Event) => void;
  }): WebSocket {
    const wsUrl = `${this.config.baseURL.replace('http', 'ws')}${this.BASE_PATH}/realtime/${userId}`;
    const ws = new WebSocket(wsUrl);

    if (callbacks.onOpen) {
      ws.onopen = callbacks.onOpen;
    }
    ws.onmessage = (event): void => {
      try {
        const data = JSON.parse(event.data);
        if (callbacks.onMessage) {
          callbacks.onMessage(data);
        }
      } catch (error) {
        if (callbacks.onError) {
          callbacks.onError(new Event('messageerror'));
        }
      }
    };
    if (callbacks.onClose) {
      ws.onclose = callbacks.onClose;
    }
    if (callbacks.onError) {
      ws.onerror = callbacks.onError;
    }

    return ws;
  }
}
