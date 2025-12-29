import apiClient from './apiClient';

// ==================== 类型定义 ====================

export interface CreateStageRequest {
  name: string;
  environment: 'development' | 'testing' | 'staging' | 'production';
  traffic_percentage: number;
  duration_hours: number;
  success_criteria: Record<string, any>;
  rollback_criteria: Record<string, any>;
  approval_required: boolean;
  approvers: string[];
}

export interface CreateStrategyRequest {
  experiment_id: string;
  name: string;
  description?: string;
  release_type: 'canary' | 'blue_green' | 'rolling' | 'feature_flag' | 'gradual' | 'shadow';
  stages: CreateStageRequest[];
  approval_level: 'none' | 'single' | 'multiple' | 'tiered';
  auto_promote: boolean;
  auto_rollback: boolean;
  monitoring_config: Record<string, any>;
  notification_config: Record<string, any>;
}

export interface CreateFromTemplateRequest {
  experiment_id: string;
  template_name: string;
  customizations?: Record<string, any>;
}

export interface ApproveStageRequest {
  exec_id: string;
  stage_index: number;
  approver: string;
  approved: boolean;
  comments?: string;
}

export interface ReleaseStrategy {
  id: string;
  name: string;
  description: string;
  experiment_id: string;
  release_type: string;
  stages: ReleaseStage[];
  approval_level: string;
  auto_promote: boolean;
  auto_rollback: boolean;
  monitoring_config: Record<string, any>;
  notification_config: Record<string, any>;
  created_at: string;
  updated_at: string;
}

export interface ReleaseStage {
  name: string;
  environment: string;
  traffic_percentage: number;
  duration_hours: number;
  success_criteria: Record<string, any>;
  rollback_criteria: Record<string, any>;
  approval_required: boolean;
  approvers: string[];
}

export interface StrategyExecution {
  strategy_id: string;
  experiment_id: string;
  status: string;
  current_stage: number;
  started_at: string;
}

export interface ReleaseTemplate {
  name: string;
  display_name: string;
  description: string;
  release_type: string;
  num_stages: number;
  approval_level: string;
  auto_promote: boolean;
  auto_rollback: boolean;
}

export interface ReleaseType {
  value: string;
  name: string;
  description: string;
  use_case: string;
}

export interface ApprovalLevel {
  value: string;
  name: string;
  description: string;
}

export interface Environment {
  value: string;
  name: string;
  description: string;
}

// ==================== Service Class ====================

class ReleaseStrategyService {
  private baseUrl = '/release-strategy';

  // ==================== 策略管理 ====================

  async createStrategy(request: CreateStrategyRequest): Promise<{
    success: boolean;
    strategy: {
      id: string;
      name: string;
      experiment_id: string;
      release_type: string;
      num_stages: number;
      approval_level: string;
      auto_promote: boolean;
      auto_rollback: boolean;
    };
    validation_errors: string[];
  }> {
    const response = await apiClient.post(`${this.baseUrl}/strategies`, request);
    return response.data;
  }

  async createFromTemplate(request: CreateFromTemplateRequest): Promise<{
    success: boolean;
    strategy: {
      id: string;
      name: string;
      experiment_id: string;
      release_type: string;
      template_used: string;
      num_stages: number;
    };
  }> {
    const response = await apiClient.post(`${this.baseUrl}/strategies/from-template`, request);
    return response.data;
  }

  async listStrategies(params: {
    experiment_id?: string;
    release_type?: string;
  } = {}): Promise<{
    success: boolean;
    strategies: Array<{
      id: string;
      name: string;
      experiment_id: string;
      release_type: string;
      num_stages: number;
      approval_level: string;
      created_at: string;
    }>;
    total: number;
  }> {
    const response = await apiClient.get(`${this.baseUrl}/strategies`, { params });
    return response.data;
  }

  async getStrategy(strategyId: string): Promise<{
    success: boolean;
    strategy: ReleaseStrategy;
  }> {
    const response = await apiClient.get(`${this.baseUrl}/strategies/${strategyId}`);
    return response.data;
  }

  // ==================== 执行管理 ====================

  async executeStrategy(strategyId: string): Promise<{
    success: boolean;
    exec_id: string;
    execution: StrategyExecution;
  }> {
    const response = await apiClient.post(`${this.baseUrl}/execute/${strategyId}`);
    return response.data;
  }

  async approveStage(request: ApproveStageRequest): Promise<{
    success: boolean;
    stage_approved: boolean;
    message: string;
  }> {
    const response = await apiClient.post(`${this.baseUrl}/approve`, request);
    return response.data;
  }

  async getExecutionStatus(execId: string): Promise<{
    success: boolean;
    status: any;
  }> {
    const response = await apiClient.get(`${this.baseUrl}/executions/${execId}`);
    return response.data;
  }

  // ==================== 模板和配置 ====================

  async listTemplates(): Promise<{
    success: boolean;
    templates: ReleaseTemplate[];
  }> {
    const response = await apiClient.get(`${this.baseUrl}/templates`);
    return response.data;
  }

  async listReleaseTypes(): Promise<{
    success: boolean;
    release_types: ReleaseType[];
  }> {
    const response = await apiClient.get(`${this.baseUrl}/release-types`);
    return response.data;
  }

  async listApprovalLevels(): Promise<{
    success: boolean;
    approval_levels: ApprovalLevel[];
  }> {
    const response = await apiClient.get(`${this.baseUrl}/approval-levels`);
    return response.data;
  }

  async listEnvironments(): Promise<{
    success: boolean;
    environments: Environment[];
  }> {
    const response = await apiClient.get(`${this.baseUrl}/environments`);
    return response.data;
  }

  // ==================== 系统状态 ====================

  async getHealthCheck(): Promise<{
    success: boolean;
    service: string;
    status: string;
    total_strategies: number;
    active_executions: number;
  }> {
    const response = await apiClient.get(`${this.baseUrl}/health`);
    return response.data;
  }

  // ==================== 工具方法 ====================

  createSampleStrategy(): CreateStrategyRequest {
    return {
      experiment_id: `exp_${Date.now()}`,
      name: '金丝雀发布策略',
      description: '用于新功能的金丝雀发布验证',
      release_type: 'canary',
      stages: [
        {
          name: '测试环境验证',
          environment: 'testing',
          traffic_percentage: 100,
          duration_hours: 2,
          success_criteria: {
            error_rate: { max: 0.01 },
            response_time: { p95: 500 }
          },
          rollback_criteria: {
            error_rate: { max: 0.05 }
          },
          approval_required: false,
          approvers: []
        },
        {
          name: '预发环境验证',
          environment: 'staging',
          traffic_percentage: 100,
          duration_hours: 4,
          success_criteria: {
            error_rate: { max: 0.005 },
            response_time: { p95: 300 }
          },
          rollback_criteria: {
            error_rate: { max: 0.02 }
          },
          approval_required: true,
          approvers: ['qa-lead', 'product-manager']
        },
        {
          name: '生产环境金丝雀',
          environment: 'production',
          traffic_percentage: 5,
          duration_hours: 8,
          success_criteria: {
            error_rate: { max: 0.003 },
            conversion_rate: { min_change: -0.05 }
          },
          rollback_criteria: {
            error_rate: { max: 0.01 }
          },
          approval_required: true,
          approvers: ['sre-lead', 'product-owner']
        },
        {
          name: '生产环境全量',
          environment: 'production',
          traffic_percentage: 100,
          duration_hours: 24,
          success_criteria: {
            error_rate: { max: 0.002 },
            performance_regression: { max: 0.1 }
          },
          rollback_criteria: {
            error_rate: { max: 0.008 }
          },
          approval_required: true,
          approvers: ['engineering-director']
        }
      ],
      approval_level: 'single',
      auto_promote: false,
      auto_rollback: true,
      monitoring_config: {
        metrics: ['error_rate', 'response_time', 'conversion_rate'],
        alerts: {
          error_rate_threshold: 0.01,
          response_time_threshold: 500
        }
      },
      notification_config: {
        channels: ['slack', 'email'],
        recipients: ['dev-team', 'qa-team', 'product-team']
      }
    };
  }

  createSampleTemplateRequest(): CreateFromTemplateRequest {
    return {
      experiment_id: `exp_${Date.now()}`,
      template_name: 'standard-canary',
      customizations: {
        stages: {
          production_canary: {
            traffic_percentage: 10,
            duration_hours: 12
          }
        },
        monitoring_config: {
          additional_metrics: ['business_kpi']
        }
      }
    };
  }

  createSampleApprovalRequest(): ApproveStageRequest {
    return {
      exec_id: 'exec_sample',
      stage_index: 1,
      approver: 'qa-lead',
      approved: true,
      comments: '测试通过，可以继续下一阶段'
    };
  }
}

// ==================== 导出 ====================

export const releaseStrategyService = new ReleaseStrategyService();
export default releaseStrategyService;