import apiClient from './apiClient';

import { logger } from '../utils/logger'
// 风险等级枚举
export enum RiskLevel {
  MINIMAL = 'minimal',
  LOW = 'low',
  MEDIUM = 'medium',
  HIGH = 'high',
  CRITICAL = 'critical'
}

// 风险类别枚举
export enum RiskCategory {
  PERFORMANCE = 'performance',
  BUSINESS = 'business',
  TECHNICAL = 'technical',
  USER_EXPERIENCE = 'user_experience',
  DATA_QUALITY = 'data_quality'
}

// 回滚策略枚举
export enum RollbackStrategy {
  IMMEDIATE = 'immediate',
  GRADUAL = 'gradual',
  PARTIAL = 'partial',
  MANUAL = 'manual'
}

// 风险因素接口
export interface RiskFactor {
  category: RiskCategory;
  name: string;
  description: string;
  risk_score: number;
  severity: string;
  likelihood: string;
  impact: string;
  mitigation: string;
}

// 风险评估接口
export interface RiskAssessment {
  experiment_id: string;
  assessment_time: string;
  overall_risk_level: RiskLevel;
  overall_risk_score: number;
  requires_rollback: boolean;
  rollback_strategy?: RollbackStrategy;
  confidence: number;
  risk_factors: RiskFactor[];
  recommendations: string[];
}

// 回滚计划接口
export interface RollbackPlan {
  experiment_id: string;
  trigger_reason: string;
  strategy: RollbackStrategy;
  steps: string[];
  estimated_duration_minutes: number;
  auto_execute: boolean;
  approval_required: boolean;
  created_at: string;
}

// 回滚执行接口
export interface RollbackExecution {
  plan_id: string;
  experiment_id: string;
  started_at: string;
  completed_at?: string;
  status: string;
  steps_completed: number;
  total_steps: number;
  errors: string[];
  metrics_before?: Record<string, any>;
  metrics_after?: Record<string, any>;
}

// 风险阈值接口
export interface RiskThreshold {
  category: string;
  metric: string;
  value: number;
}

// 风险等级信息接口
export interface RiskLevelInfo {
  value: RiskLevel;
  name: string;
  score_range: string;
  color: string;
  action: string;
}

// 风险类别信息接口
export interface RiskCategoryInfo {
  value: RiskCategory;
  name: string;
  description: string;
  metrics: string[];
}

// 回滚策略信息接口
export interface RollbackStrategyInfo {
  value: RollbackStrategy;
  name: string;
  description: string;
  duration: string;
  use_case: string;
}

// 评估风险请求接口
export interface AssessRiskRequest {
  experiment_id: string;
  include_predictions?: boolean;
}

// 创建回滚计划请求接口
export interface CreateRollbackPlanRequest {
  experiment_id: string;
  strategy?: RollbackStrategy;
  auto_execute?: boolean;
}

// 执行回滚请求接口
export interface ExecuteRollbackRequest {
  plan_id: string;
  force?: boolean;
}

// 监控风险请求接口
export interface MonitorRiskRequest {
  experiment_id: string;
  check_interval_minutes?: number;
}

class RiskAssessmentService {
  private baseUrl = '/risk-assessment';

  // 评估风险
  async assessRisk(request: AssessRiskRequest): Promise<{ success: boolean; assessment: RiskAssessment }> {
    try {
      const response = await apiClient.post(`${this.baseUrl}/assess`, request);
      return response.data;
    } catch (error) {
      logger.error('评估风险失败:', error);
      throw error;
    }
  }

  // 获取风险历史
  async getRiskHistory(experimentId: string, limit: number = 10): Promise<{
    success: boolean;
    experiment_id: string;
    assessments: Array<{
      assessment_time: string;
      risk_level: RiskLevel;
      risk_score: number;
      requires_rollback: boolean;
      num_risk_factors: number;
    }>;
    total_assessments: number;
  }> {
    try {
      const response = await apiClient.get(`${this.baseUrl}/history/${experimentId}`, {
        params: { limit }
      });
      return response.data;
    } catch (error) {
      logger.error('获取风险历史失败:', error);
      throw error;
    }
  }

  // 创建回滚计划
  async createRollbackPlan(request: CreateRollbackPlanRequest): Promise<{
    success: boolean;
    plan_id: string;
    plan: RollbackPlan;
  }> {
    try {
      const response = await apiClient.post(`${this.baseUrl}/rollback-plan`, request);
      return response.data;
    } catch (error) {
      logger.error('创建回滚计划失败:', error);
      throw error;
    }
  }

  // 执行回滚
  async executeRollback(request: ExecuteRollbackRequest): Promise<{
    success: boolean;
    execution: RollbackExecution;
  }> {
    try {
      const response = await apiClient.post(`${this.baseUrl}/rollback/execute`, request);
      return response.data;
    } catch (error) {
      logger.error('执行回滚失败:', error);
      throw error;
    }
  }

  // 获取回滚状态
  async getRollbackStatus(execId: string): Promise<{
    success: boolean;
    status: {
      exec_id: string;
      experiment_id: string;
      status: string;
      started_at: string;
      completed_at?: string;
      progress: string;
      errors: string[];
      metrics_comparison: {
        before: Record<string, any>;
        after: Record<string, any>;
      };
    };
  }> {
    try {
      const response = await apiClient.get(`${this.baseUrl}/rollback/status/${execId}`);
      return response.data;
    } catch (error) {
      logger.error('获取回滚状态失败:', error);
      throw error;
    }
  }

  // 启动风险监控
  async startMonitoring(request: MonitorRiskRequest): Promise<{
    success: boolean;
    message: string;
    experiment_id: string;
    check_interval_minutes: number;
  }> {
    try {
      const response = await apiClient.post(`${this.baseUrl}/monitor`, request);
      return response.data;
    } catch (error) {
      logger.error('启动风险监控失败:', error);
      throw error;
    }
  }

  // 获取风险阈值
  async getRiskThresholds(): Promise<{
    success: boolean;
    thresholds: Record<string, Record<string, number>>;
  }> {
    try {
      const response = await apiClient.get(`${this.baseUrl}/thresholds`);
      return response.data;
    } catch (error) {
      logger.error('获取风险阈值失败:', error);
      throw error;
    }
  }

  // 更新风险阈值
  async updateRiskThreshold(category: string, metric: string, value: number): Promise<{
    success: boolean;
    message: string;
    category: string;
    metric: string;
    new_value: number;
  }> {
    try {
      const response = await apiClient.put(`${this.baseUrl}/thresholds`, null, {
        params: { category, metric, value }
      });
      return response.data;
    } catch (error) {
      logger.error('更新风险阈值失败:', error);
      throw error;
    }
  }

  // 获取风险等级列表
  async listRiskLevels(): Promise<{ success: boolean; levels: RiskLevelInfo[] }> {
    try {
      const response = await apiClient.get(`${this.baseUrl}/risk-levels`);
      return response.data;
    } catch (error) {
      logger.error('获取风险等级列表失败:', error);
      throw error;
    }
  }

  // 获取风险类别列表
  async listRiskCategories(): Promise<{ success: boolean; categories: RiskCategoryInfo[] }> {
    try {
      const response = await apiClient.get(`${this.baseUrl}/categories`);
      return response.data;
    } catch (error) {
      logger.error('获取风险类别列表失败:', error);
      throw error;
    }
  }

  // 获取回滚策略列表
  async listRollbackStrategies(): Promise<{ success: boolean; strategies: RollbackStrategyInfo[] }> {
    try {
      const response = await apiClient.get(`${this.baseUrl}/strategies`);
      return response.data;
    } catch (error) {
      logger.error('获取回滚策略列表失败:', error);
      throw error;
    }
  }

  // 健康检查
  async healthCheck(): Promise<{
    success: boolean;
    service: string;
    status: string;
    total_assessments: number;
    active_rollback_plans: number;
    rollback_executions: number;
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
export const riskAssessmentService = new RiskAssessmentService();
export default riskAssessmentService;