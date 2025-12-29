/**
 * 自动化工作流统计服务
 * 提供工作流自动化监控和统计功能
 */

import apiClient from './apiClient';

// 工作流统计接口
export interface WorkflowStats {
  total_workflows: number;
  active_workflows: number;
  automated_tasks: number;
  success_rate: number;
  time_saved_hours: number;
  cost_reduction_percent: number;
  workflow_categories: {
    data_processing: number;
    model_training: number;
    deployment: number;
    monitoring: number;
  };
  efficiency_metrics: {
    automation_rate: number;
    manual_intervention_rate: number;
    average_execution_time: string;
  };
}

// 工作流执行记录接口
export interface WorkflowExecution {
  execution_id: string;
  workflow_id: string;
  workflow_name: string;
  status: 'running' | 'completed' | 'failed' | 'cancelled';
  started_at: string;
  completed_at?: string;
  duration_seconds?: number;
  steps_total: number;
  steps_completed: number;
  steps_failed: number;
  success_rate: number;
  error_message?: string;
  resource_usage: {
    cpu_time: number;
    memory_peak_mb: number;
    io_operations: number;
  };
}

// 自动化规则接口
export interface AutomationRule {
  rule_id: string;
  name: string;
  description: string;
  trigger_type: 'schedule' | 'event' | 'condition';
  trigger_config: Record<string, any>;
  workflow_id: string;
  enabled: boolean;
  created_at: string;
  last_triggered?: string;
  trigger_count: number;
  success_count: number;
  failure_count: number;
}

// 工作流模板接口
export interface WorkflowTemplate {
  template_id: string;
  name: string;
  category: string;
  description: string;
  steps: Array<{
    step_id: string;
    name: string;
    type: string;
    config: Record<string, any>;
    dependencies: string[];
  }>;
  estimated_duration: string;
  complexity: 'simple' | 'medium' | 'complex';
  tags: string[];
}

class AutomationService {
  private baseUrl = '/automation';

  /**
   * 获取工作流自动化统计
   */
  async getWorkflowStats(): Promise<WorkflowStats> {
    const response = await apiClient.get(`${this.baseUrl}/workflow-stats`);
    return response.data;
  }

  /**
   * 获取工作流执行历史
   */
  async getExecutionHistory(
    workflowId?: string,
    status?: string,
    limit: number = 50,
    offset: number = 0
  ): Promise<WorkflowExecution[]> {
    const response = await apiClient.get(`${this.baseUrl}/executions`, {
      params: { workflow_id: workflowId, status, limit, offset }
    });
    return response.data;
  }

  /**
   * 获取执行详情
   */
  async getExecutionDetails(executionId: string): Promise<WorkflowExecution & {
    step_details: Array<{
      step_id: string;
      name: string;
      status: string;
      started_at: string;
      completed_at?: string;
      duration_seconds?: number;
      logs: string[];
      output?: any;
      error?: string;
    }>;
  }> {
    const response = await apiClient.get(`${this.baseUrl}/executions/${executionId}`);
    return response.data;
  }

  /**
   * 获取自动化规则列表
   */
  async getAutomationRules(enabled_only: boolean = false): Promise<AutomationRule[]> {
    const response = await apiClient.get(`${this.baseUrl}/rules`, {
      params: { enabled_only }
    });
    return response.data;
  }

  /**
   * 创建自动化规则
   */
  async createAutomationRule(rule: Omit<AutomationRule, 'rule_id' | 'created_at' | 'last_triggered' | 'trigger_count' | 'success_count' | 'failure_count'>): Promise<AutomationRule> {
    const response = await apiClient.post(`${this.baseUrl}/rules`, rule);
    return response.data;
  }

  /**
   * 更新自动化规则
   */
  async updateAutomationRule(ruleId: string, updates: Partial<AutomationRule>): Promise<AutomationRule> {
    const response = await apiClient.put(`${this.baseUrl}/rules/${ruleId}`, updates);
    return response.data;
  }

  /**
   * 删除自动化规则
   */
  async deleteAutomationRule(ruleId: string): Promise<{ message: string }> {
    const response = await apiClient.delete(`${this.baseUrl}/rules/${ruleId}`);
    return response.data;
  }

  /**
   * 启用/禁用自动化规则
   */
  async toggleAutomationRule(ruleId: string, enabled: boolean): Promise<{ message: string }> {
    const response = await apiClient.patch(`${this.baseUrl}/rules/${ruleId}/toggle`, {
      enabled
    });
    return response.data;
  }

  /**
   * 手动触发自动化规则
   */
  async triggerRule(ruleId: string, parameters?: Record<string, any>): Promise<{
    execution_id: string;
    message: string;
  }> {
    const response = await apiClient.post(`${this.baseUrl}/rules/${ruleId}/trigger`, {
      parameters
    });
    return response.data;
  }

  /**
   * 获取工作流模板
   */
  async getWorkflowTemplates(category?: string): Promise<WorkflowTemplate[]> {
    const response = await apiClient.get(`${this.baseUrl}/templates`, {
      params: { category }
    });
    return response.data;
  }

  /**
   * 从模板创建工作流
   */
  async createWorkflowFromTemplate(
    templateId: string,
    name: string,
    customizations?: Record<string, any>
  ): Promise<{
    workflow_id: string;
    message: string;
  }> {
    const response = await apiClient.post(`${this.baseUrl}/templates/${templateId}/create`, {
      name,
      customizations
    });
    return response.data;
  }

  /**
   * 获取性能指标
   */
  async getPerformanceMetrics(timeRange: '24h' | '7d' | '30d' = '24h'): Promise<{
    total_executions: number;
    successful_executions: number;
    failed_executions: number;
    average_execution_time: number;
    median_execution_time: number;
    p95_execution_time: number;
    throughput_per_hour: number;
    error_rate: number;
    resource_efficiency: {
      cpu_utilization: number;
      memory_efficiency: number;
      cost_per_execution: number;
    };
    trend_data: Array<{
      timestamp: string;
      executions: number;
      success_rate: number;
      avg_duration: number;
    }>;
  }> {
    const response = await apiClient.get(`${this.baseUrl}/metrics/performance`, {
      params: { time_range: timeRange }
    });
    return response.data;
  }

  /**
   * 获取资源使用情况
   */
  async getResourceUsage(): Promise<{
    current_executions: number;
    queued_executions: number;
    available_workers: number;
    total_workers: number;
    cpu_usage: number;
    memory_usage: number;
    disk_usage: number;
    network_io: {
      ingress: number;
      egress: number;
    };
  }> {
    const response = await apiClient.get(`${this.baseUrl}/resources/usage`);
    return response.data;
  }

  /**
   * 获取节省成本报告
   */
  async getCostSavings(timeRange: '30d' | '90d' | '1y' = '30d'): Promise<{
    total_cost_saved: number;
    time_saved_hours: number;
    manual_effort_avoided: number;
    efficiency_improvement: number;
    breakdown: {
      data_processing: number;
      model_training: number;
      deployment: number;
      monitoring: number;
      other: number;
    };
    historical_data: Array<{
      period: string;
      cost_saved: number;
      time_saved: number;
    }>;
  }> {
    const response = await apiClient.get(`${this.baseUrl}/cost-savings`, {
      params: { time_range: timeRange }
    });
    return response.data;
  }

  /**
   * 获取失败分析
   */
  async getFailureAnalysis(): Promise<{
    total_failures: number;
    failure_rate: number;
    common_causes: Array<{
      cause: string;
      count: number;
      percentage: number;
    }>;
    failure_trends: Array<{
      timestamp: string;
      failure_count: number;
      total_executions: number;
    }>;
    affected_workflows: Array<{
      workflow_id: string;
      workflow_name: string;
      failure_count: number;
      last_failure: string;
    }>;
  }> {
    const response = await apiClient.get(`${this.baseUrl}/analysis/failures`);
    return response.data;
  }

  /**
   * 获取自动化建议
   */
  async getAutomationSuggestions(): Promise<Array<{
    suggestion_id: string;
    type: 'optimization' | 'new_automation' | 'error_reduction';
    title: string;
    description: string;
    potential_savings: {
      time_hours: number;
      cost_amount: number;
      efficiency_gain: number;
    };
    complexity: 'low' | 'medium' | 'high';
    implementation_steps: string[];
    priority: number;
  }>> {
    const response = await apiClient.get(`${this.baseUrl}/suggestions`);
    return response.data;
  }

  /**
   * 应用自动化建议
   */
  async applySuggestion(suggestionId: string): Promise<{
    message: string;
    workflow_id?: string;
    rule_id?: string;
  }> {
    const response = await apiClient.post(`${this.baseUrl}/suggestions/${suggestionId}/apply`);
    return response.data;
  }

  /**
   * 导出自动化报告
   */
  async exportReport(
    format: 'pdf' | 'excel',
    timeRange: '30d' | '90d' | '1y' = '30d'
  ): Promise<Blob> {
    const response = await apiClient.get(`${this.baseUrl}/reports/export`, {
      params: { format, time_range: timeRange },
      responseType: 'blob'
    });
    return response.data;
  }
}

export const automationService = new AutomationService();