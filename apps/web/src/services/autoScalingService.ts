import apiClient from './apiClient';

import { logger } from '../utils/logger'
// 扩量模式枚举
export enum ScalingMode {
  AGGRESSIVE = 'aggressive',
  CONSERVATIVE = 'conservative',
  BALANCED = 'balanced',
  ADAPTIVE = 'adaptive'
}

// 扩量方向枚举
export enum ScalingDirection {
  UP = 'scale_up',
  DOWN = 'scale_down',
  NONE = 'none'
}

// 触发器类型枚举
export enum ScalingTrigger {
  METRIC_THRESHOLD = 'metric_threshold',
  STATISTICAL_SIGNIFICANCE = 'statistical_significance',
  SAMPLE_SIZE = 'sample_size',
  CONFIDENCE_INTERVAL = 'confidence_interval',
  TIME_BASED = 'time_based'
}

// 扩量规则接口
export interface ScalingRule {
  id: string;
  name: string;
  experiment_id: string;
  mode: ScalingMode;
  variant?: string;
  description?: string;
  scale_increment: number;
  scale_decrement: number;
  min_percentage: number;
  max_percentage: number;
  cooldown_minutes: number;
  enabled: boolean;
  scale_up_conditions_count?: number;
  scale_down_conditions_count?: number;
  created_at?: string;
}

// 扩量条件接口
export interface ScalingCondition {
  trigger: ScalingTrigger;
  metric_name?: string;
  operator: string;
  threshold: number;
  confidence_level?: number;
  min_sample_size?: number;
}

// 扩量决策接口
export interface ScalingDecision {
  timestamp: string;
  direction: ScalingDirection;
  from: number;
  to: number;
  reason: string;
  confidence: number;
}

// 扩量历史接口
export interface ScalingHistory {
  experiment_id: string;
  current_percentage: number;
  last_scaled_at?: string;
  total_scale_ups: number;
  total_scale_downs: number;
  recent_decisions: ScalingDecision[];
}

export interface ScalingSimulation {
  day: number;
  current_percentage: number;
  new_percentage: number;
  action: string;
  confidence: number;
  metrics: {
    conversion_rate?: number;
    sample_size?: number;
    p_value?: number;
  };
}

// 扩量建议接口
export interface ScalingRecommendation {
  action: string;
  confidence: number;
  reason: string;
  suggested_percentage?: number;
  risk_level: 'low' | 'medium' | 'high';
  expected_impact?: string;
}

// 扩量模式信息接口
export interface ScalingModeInfo {
  value: ScalingMode;
  name: string;
  description: string;
  scale_increment: number | string;
  cooldown_minutes: number | string;
}

// 触发器信息接口
export interface TriggerInfo {
  value: ScalingTrigger;
  name: string;
  description: string;
  required_params: string[];
}

// 创建扩量规则请求接口
export interface CreateScalingRuleRequest {
  experiment_id: string;
  name: string;
  mode?: ScalingMode;
  variant?: string;
  description?: string;
  scale_increment?: number;
  scale_decrement?: number;
  min_percentage?: number;
  max_percentage?: number;
  cooldown_minutes?: number;
  enabled?: boolean;
}

// 创建条件请求接口
export interface CreateConditionRequest {
  trigger: ScalingTrigger;
  metric_name?: string;
  operator: string;
  threshold: number;
  confidence_level?: number;
  min_sample_size?: number;
}

class AutoScalingService {
  private baseUrl = '/auto-scaling';

  // 创建扩量规则
  async createScalingRule(request: CreateScalingRuleRequest): Promise<{ success: boolean; rule: ScalingRule }> {
    try {
      const response = await apiClient.post(`${this.baseUrl}/rules`, request);
      return response.data;
    } catch (error) {
      logger.error('创建扩量规则失败:', error);
      throw error;
    }
  }

  // 添加扩量条件
  async addCondition(
    ruleId: string,
    conditionType: 'scale_up' | 'scale_down',
    condition: CreateConditionRequest
  ): Promise<{ success: boolean; message: string }> {
    try {
      const response = await apiClient.post(
        `${this.baseUrl}/rules/${ruleId}/conditions?condition_type=${conditionType}`,
        condition
      );
      return response.data;
    } catch (error) {
      logger.error('添加扩量条件失败:', error);
      throw error;
    }
  }

  // 启动自动扩量
  async startAutoScaling(ruleId: string): Promise<{ success: boolean; message: string; rule_id: string }> {
    try {
      const response = await apiClient.post(`${this.baseUrl}/start`, { rule_id: ruleId });
      return response.data;
    } catch (error) {
      logger.error('启动自动扩量失败:', error);
      throw error;
    }
  }

  // 停止自动扩量
  async stopAutoScaling(ruleId: string): Promise<{ success: boolean; message: string }> {
    try {
      const response = await apiClient.post(`${this.baseUrl}/stop/${ruleId}`);
      return response.data;
    } catch (error) {
      logger.error('停止自动扩量失败:', error);
      throw error;
    }
  }

  // 获取扩量规则列表
  async listRules(experimentId?: string): Promise<{ success: boolean; rules: ScalingRule[]; total: number }> {
    try {
      const params = experimentId ? { experiment_id: experimentId } : {};
      const response = await apiClient.get(`${this.baseUrl}/rules`, { params });
      return response.data;
    } catch (error) {
      logger.error('获取扩量规则列表失败:', error);
      throw error;
    }
  }

  // 获取扩量历史
  async getScalingHistory(experimentId: string): Promise<{ success: boolean; history: ScalingHistory | null; message?: string }> {
    try {
      const response = await apiClient.get(`${this.baseUrl}/history/${experimentId}`);
      return response.data;
    } catch (error) {
      logger.error('获取扩量历史失败:', error);
      throw error;
    }
  }

  // 获取扩量建议
  async getRecommendations(experimentId: string): Promise<{ success: boolean; recommendations: ScalingRecommendation[] }> {
    try {
      const response = await apiClient.get(`${this.baseUrl}/recommendations/${experimentId}`);
      return response.data;
    } catch (error) {
      logger.error('获取扩量建议失败:', error);
      throw error;
    }
  }

  async simulateScaling(experimentId: string, days: number = 7): Promise<{
    success: boolean;
    simulations: ScalingSimulation[];
    summary: {
      start_percentage: number;
      end_percentage: number;
      total_days: number;
      scale_ups: number;
    };
  }> {
    try {
      const response = await apiClient.post(`${this.baseUrl}/simulate`, {
        experiment_id: experimentId,
        days
      });
      return response.data;
    } catch (error) {
      logger.error('扩量模拟失败:', error);
      throw error;
    }
  }

  // 创建安全扩量模板
  async createSafeTemplate(experimentId: string): Promise<{ success: boolean; rule: Partial<ScalingRule> }> {
    try {
      const response = await apiClient.post(`${this.baseUrl}/templates/safe?experiment_id=${experimentId}`);
      return response.data;
    } catch (error) {
      logger.error('创建安全扩量模板失败:', error);
      throw error;
    }
  }

  // 创建激进扩量模板
  async createAggressiveTemplate(experimentId: string): Promise<{ success: boolean; rule: Partial<ScalingRule> }> {
    try {
      const response = await apiClient.post(`${this.baseUrl}/templates/aggressive?experiment_id=${experimentId}`);
      return response.data;
    } catch (error) {
      logger.error('创建激进扩量模板失败:', error);
      throw error;
    }
  }

  // 获取扩量模式列表
  async listScalingModes(): Promise<{ success: boolean; modes: ScalingModeInfo[] }> {
    try {
      const response = await apiClient.get(`${this.baseUrl}/modes`);
      return response.data;
    } catch (error) {
      logger.error('获取扩量模式列表失败:', error);
      throw error;
    }
  }

  // 获取触发器列表
  async listTriggers(): Promise<{ success: boolean; triggers: TriggerInfo[] }> {
    try {
      const response = await apiClient.get(`${this.baseUrl}/triggers`);
      return response.data;
    } catch (error) {
      logger.error('获取触发器列表失败:', error);
      throw error;
    }
  }

  // 获取扩量服务状态
  async getScalingStatus(): Promise<{
    success: boolean;
    status: {
      active_rules: number;
      total_rules: number;
      active_experiments: Array<{
        experiment_id: string;
        rule_name: string;
        mode: ScalingMode;
      }>;
    };
  }> {
    try {
      const response = await apiClient.get(`${this.baseUrl}/status`);
      return response.data;
    } catch (error) {
      logger.error('获取扩量服务状态失败:', error);
      throw error;
    }
  }

  // 健康检查
  async healthCheck(): Promise<{
    success: boolean;
    service: string;
    status: string;
    active_monitors: number;
    total_rules: number;
  }> {
    try {
      const response = await apiClient.get(`${this.baseUrl}/health`);
      return response.data;
    } catch (error) {
      logger.error('健康检查失败:', error);
      throw error;
    }
  }
}

// 导出服务实例
export const autoScalingService = new AutoScalingService();
export default autoScalingService;