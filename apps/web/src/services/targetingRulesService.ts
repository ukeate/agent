import apiClient from './apiClient';

import { logger } from '../utils/logger'
export interface RuleCondition {
  field: string;
  operator: string;
  value: any;
  case_sensitive?: boolean;
}

export interface CompositeCondition {
  logical_operator: string;
  conditions: (RuleCondition | CompositeCondition)[];
}

export interface TargetingRule {
  rule_id: string;
  name: string;
  description: string;
  rule_type: string;
  condition: RuleCondition | CompositeCondition;
  priority: number;
  is_active: boolean;
  experiment_ids: string[];
  variant_ids: string[];
  metadata: Record<string, any>;
  created_at?: string;
  updated_at?: string;
}

export interface CreateRuleRequest {
  rule_id: string;
  name: string;
  description: string;
  rule_type: string;
  condition: Record<string, any>;
  priority?: number;
  is_active?: boolean;
  experiment_ids?: string[];
  variant_ids?: string[];
  metadata?: Record<string, any>;
}

export interface UpdateRuleRequest {
  name?: string;
  description?: string;
  condition?: Record<string, any>;
  priority?: number;
  is_active?: boolean;
  experiment_ids?: string[];
  variant_ids?: string[];
  metadata?: Record<string, any>;
}

export interface EvaluateUserRequest {
  user_id: string;
  user_context: Record<string, any>;
  experiment_id?: string;
}

export interface BatchEvaluateRequest {
  user_contexts: Record<string, any>[];
  experiment_id?: string;
}

export interface EvaluationResult {
  rule_id: string;
  rule_type: string;
  matched: boolean;
  evaluation_reason: string;
  forced_variant_id?: string;
  experiment_ids: string[];
  evaluation_time: number;
  metadata: Record<string, any>;
}

export interface UserEvaluationResponse {
  user_id: string;
  experiment_id?: string;
  total_rules_evaluated: number;
  matched_rules: number;
  results: EvaluationResult[];
}

export interface BatchEvaluationResponse {
  experiment_id?: string;
  total_users: number;
  matched_users: number;
  match_rate_percentage: number;
  detailed_results: Array<{
    user_id: string;
    total_rules_evaluated: number;
    matched_rules_count: number;
    has_forced_variant: boolean;
    matched_rule_types: string[];
  }>;
}

export interface EligibilityResult {
  is_eligible: boolean;
  eligibility_reason: string;
  matched_rules: string[];
  forced_variant_id?: string;
  metadata: Record<string, any>;
}

export interface RuleStatistics {
  total_rules: number;
  active_rules: number;
  inactive_rules: number;
  rule_types: Record<string, number>;
  average_priority: number;
  recent_evaluations: number;
}

export interface RuleOperator {
  operator: string;
  name: string;
  description: string;
}

export interface LogicalOperator {
  operator: string;
  name: string;
  description: string;
}

export interface OperatorsResponse {
  rule_operators: RuleOperator[];
  logical_operators: LogicalOperator[];
  rule_types: string[];
}

export interface RuleTemplate {
  name: string;
  description: string;
  rule_type: string;
  condition: Record<string, any>;
}

class TargetingRulesService {
  private baseUrl = '/targeting';

  // 创建定向规则
  async createRule(request: CreateRuleRequest): Promise<{
    message: string;
    rule_id: string;
    rule_type: string;
    is_active: boolean;
  }> {
    const response = await apiClient.post(`${this.baseUrl}/rules`, request);
    return response.data;
  }

  // 获取规则列表
  async getRules(
    rule_type?: string,
    experiment_id?: string,
    active_only: boolean = true
  ): Promise<{
    total_rules: number;
    rules: TargetingRule[];
  }> {
    const params = new URLSearchParams();
    if (rule_type) params.append('rule_type', rule_type);
    if (experiment_id) params.append('experiment_id', experiment_id);
    params.append('active_only', active_only.toString());

    const response = await apiClient.get(`${this.baseUrl}/rules?${params}`);
    return response.data;
  }

  // 获取特定规则
  async getRule(rule_id: string): Promise<TargetingRule> {
    const response = await apiClient.get(`${this.baseUrl}/rules/${rule_id}`);
    return response.data;
  }

  // 更新规则
  async updateRule(rule_id: string, request: UpdateRuleRequest): Promise<{
    message: string;
    rule_id: string;
    updated_at: string;
  }> {
    const response = await apiClient.put(`${this.baseUrl}/rules/${rule_id}`, request);
    return response.data;
  }

  // 删除规则
  async deleteRule(rule_id: string): Promise<{
    message: string;
    rule_id: string;
  }> {
    const response = await apiClient.delete(`${this.baseUrl}/rules/${rule_id}`);
    return response.data;
  }

  // 评估用户定向
  async evaluateUser(request: EvaluateUserRequest): Promise<UserEvaluationResponse> {
    const response = await apiClient.post(`${this.baseUrl}/evaluate`, request);
    return response.data;
  }

  // 批量评估用户定向
  async batchEvaluateUsers(request: BatchEvaluateRequest): Promise<BatchEvaluationResponse> {
    const response = await apiClient.post(`${this.baseUrl}/evaluate/batch`, request);
    return response.data;
  }

  // 检查用户资格
  async checkUserEligibility(
    user_id: string,
    experiment_id: string,
    user_context: Record<string, any> = {}
  ): Promise<EligibilityResult & { user_id: string; experiment_id: string }> {
    const params = new URLSearchParams();
    params.append('user_id', user_id);
    params.append('experiment_id', experiment_id);

    const response = await apiClient.post(
      `${this.baseUrl}/check-eligibility?${params}`,
      user_context
    );
    return response.data;
  }

  // 获取统计信息
  async getStatistics(): Promise<RuleStatistics> {
    const response = await apiClient.get(`${this.baseUrl}/statistics`);
    return response.data;
  }

  // 清除规则
  async clearRules(rule_type?: string): Promise<{
    message: string;
    cleared_count: number;
    rule_type: string;
  }> {
    const params = new URLSearchParams();
    params.append('confirm', 'true');
    if (rule_type) params.append('rule_type', rule_type);

    const response = await apiClient.delete(`${this.baseUrl}/rules?${params}`);
    return response.data;
  }

  // 获取规则模板
  async getRuleTemplates(): Promise<{
    message: string;
    templates: Record<string, RuleTemplate>;
  }> {
    const response = await apiClient.post(`${this.baseUrl}/rules/templates`);
    return response.data;
  }

  // 获取可用操作符
  async getOperators(): Promise<OperatorsResponse> {
    const response = await apiClient.get(`${this.baseUrl}/operators`);
    return response.data;
  }

  // 创建简单条件规则
  async createSimpleRule(
    rule_id: string,
    name: string,
    description: string,
    rule_type: 'blacklist' | 'whitelist' | 'targeting',
    field: string,
    operator: string,
    value: any,
    priority: number = 0,
    experiment_ids: string[] = []
  ): Promise<any> {
    const request: CreateRuleRequest = {
      rule_id,
      name,
      description,
      rule_type,
      condition: {
        field,
        operator,
        value,
        case_sensitive: false
      },
      priority,
      is_active: true,
      experiment_ids,
      variant_ids: [],
      metadata: {}
    };

    return this.createRule(request);
  }

  // 创建复合条件规则
  async createCompositeRule(
    rule_id: string,
    name: string,
    description: string,
    rule_type: 'blacklist' | 'whitelist' | 'targeting',
    logical_operator: 'and' | 'or',
    conditions: Array<{
      field: string;
      operator: string;
      value: any;
      case_sensitive?: boolean;
    }>,
    priority: number = 0,
    experiment_ids: string[] = []
  ): Promise<any> {
    const request: CreateRuleRequest = {
      rule_id,
      name,
      description,
      rule_type,
      condition: {
        logical_operator,
        conditions: conditions.map(cond => ({
          field: cond.field,
          operator: cond.operator,
          value: cond.value,
          case_sensitive: cond.case_sensitive || false
        }))
      },
      priority,
      is_active: true,
      experiment_ids,
      variant_ids: [],
      metadata: {}
    };

    return this.createRule(request);
  }

  // 测试用户场景
  async testUserScenario(
    user_id: string,
    user_context: Record<string, any>
  ): Promise<UserEvaluationResponse> {
    return this.evaluateUser({
      user_id,
      user_context
    });
  }

  // 创建预设规则
  async createPresetRules(): Promise<{
    created_rules: string[];
    results: any[];
  }> {
    const presetRules = [
      {
        rule_id: 'blacklist_restricted_countries',
        name: '受限国家黑名单',
        description: '禁止特定国家用户参与实验',
        rule_type: 'blacklist',
        field: 'country',
        operator: 'in',
        value: ['CN', 'RU', 'KP'],
        priority: 10
      },
      {
        rule_id: 'whitelist_premium_users',
        name: '高级用户白名单',
        description: '仅允许高级用户参与特定实验',
        rule_type: 'whitelist',
        field: 'user_tier',
        operator: 'eq',
        value: 'premium',
        priority: 5
      },
      {
        rule_id: 'targeting_mobile_users',
        name: '移动端用户定向',
        description: '针对移动设备用户的定向规则',
        rule_type: 'targeting',
        field: 'device_type',
        operator: 'eq',
        value: 'mobile',
        priority: 0
      }
    ];

    const results = [];
    const created_rules = [];

    for (const rule of presetRules) {
      try {
        const result = await this.createSimpleRule(
          rule.rule_id,
          rule.name,
          rule.description,
          rule.rule_type as any,
          rule.field,
          rule.operator,
          rule.value,
          rule.priority
        );
        results.push(result);
        created_rules.push(rule.rule_id);
      } catch (error) {
        logger.error(`创建规则失败 ${rule.rule_id}:`, error);
      }
    }

    return { created_rules, results };
  }
}

export default new TargetingRulesService();
