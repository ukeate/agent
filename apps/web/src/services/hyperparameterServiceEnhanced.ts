import { buildApiUrl } from '../utils/apiBase'
import apiClient from './apiClient';
import { hyperparameterService } from './hyperparameterService';

import { logger } from '../utils/logger'
// 继承原有接口
export * from './hyperparameterService';

// 扩展接口
export interface ExperimentProgress {
  experiment_id: string;
  current_trial: number;
  total_trials: number;
  best_value?: number;
  best_params?: Record<string, any>;
  elapsed_time: number;
  estimated_remaining_time?: number;
  status: string;
}

export interface TaskInfo {
  name: string;
  description: string;
  parameters: Array<{
    name: string;
    type: string;
    range?: any;
    description?: string;
  }>;
  algorithm: string;
  pruning?: string;
  direction: 'minimize' | 'maximize';
  n_trials: number;
  early_stopping?: boolean;
  patience?: number;
}

export interface CustomTaskRequest {
  task_name: string;
  parameters: Array<{
    name: string;
    type: 'float' | 'int' | 'categorical' | 'boolean';
    low?: number;
    high?: number;
    choices?: string[];
    step?: number;
    log?: boolean;
  }>;
  algorithm: string;
  pruning?: string;
  direction: 'minimize' | 'maximize';
  n_trials: number;
  early_stopping?: boolean;
  patience?: number;
}

export interface OptimizationResult {
  task_name: string;
  best_value: number;
  best_params: Record<string, any>;
  total_trials: number;
  convergence_history: Array<{
    trial: number;
    value: number;
    parameters: Record<string, any>;
  }>;
  algorithm_used: string;
  elapsed_time: number;
}

export interface AlgorithmComparison {
  task_name: string;
  algorithms: Array<{
    name: string;
    best_value: number;
    best_params: Record<string, any>;
    convergence_rate: number;
    stability: number;
    efficiency_score: number;
    total_trials: number;
  }>;
  winner: string;
  comparison_metrics: {
    convergence_speed: Record<string, number>;
    final_performance: Record<string, number>;
    stability_score: Record<string, number>;
  };
}

export interface ResourceStats {
  cpu_usage: number;
  memory_usage: number;
  active_experiments: number;
  pending_trials: number;
  completed_trials: number;
  failed_trials: number;
  storage_used: number;
  storage_available: number;
  network_io: {
    bytes_sent: number;
    bytes_received: number;
  };
}

export interface AlgorithmInfo {
  algorithms: string[];
  descriptions: Record<string, string>;
  parameters?: Record<string, Record<string, any>>;
}

export interface PruningInfo {
  pruning_strategies: string[];
  descriptions: Record<string, string>;
}

export interface ParameterTypeInfo {
  parameter_types: string[];
  descriptions: Record<string, string>;
  examples: Record<string, any>;
}

export interface ExperimentVisualization {
  optimization_history: {
    data: Array<{ trial: number; value: number; is_best: boolean }>;
    chart_config: any;
  };
  parameter_importance: {
    data: Record<string, number>;
    chart_config: any;
  };
  parallel_coordinates: {
    data: Array<Record<string, any>>;
    chart_config: any;
  };
  convergence_plot: {
    data: Array<{ trial: number; best_value: number; current_value: number }>;
    chart_config: any;
  };
  hyperparameter_heatmap: {
    data: Array<Array<number>>;
    parameters: string[];
    chart_config: any;
  };
}

class HyperparameterServiceEnhanced {
  private baseUrl = '/hyperparameter-optimization';

  // 继承原有方法
  listExperiments = hyperparameterService.listExperiments.bind(hyperparameterService);
  getExperiment = hyperparameterService.getExperiment.bind(hyperparameterService);
  createExperiment = hyperparameterService.createExperiment.bind(hyperparameterService);
  deleteExperiment = hyperparameterService.deleteExperiment.bind(hyperparameterService);
  stopExperiment = hyperparameterService.stopExperiment.bind(hyperparameterService);
  listTrials = hyperparameterService.listTrials.bind(hyperparameterService);
  getExperimentStatistics = hyperparameterService.getExperimentStatistics.bind(hyperparameterService);

  // 启动实验
  async startExperiment(experimentId: string): Promise<{ status: string; experiment_id: string }> {
    const response = await apiClient.post(`${this.baseUrl}/experiments/${experimentId}/start`);
    return response.data;
  }

  // 获取实验可视化
  async getExperimentVisualizations(experimentId: string): Promise<ExperimentVisualization> {
    const response = await apiClient.get(`${this.baseUrl}/experiments/${experimentId}/visualizations`);
    return response.data;
  }

  // 获取实验实时进度
  async getExperimentProgress(experimentId: string): Promise<ExperimentProgress> {
    const response = await apiClient.get(`${this.baseUrl}/experiments/${experimentId}/progress`);
    return response.data;
  }

  // 获取预设任务列表
  async getPresetTasks(): Promise<string[]> {
    const response = await apiClient.get(`${this.baseUrl}/tasks`);
    return response.data;
  }

  // 获取任务信息
  async getTaskInfo(taskName: string): Promise<TaskInfo> {
    const response = await apiClient.get(`${this.baseUrl}/tasks/${taskName}`);
    return response.data;
  }

  // 创建自定义任务
  async createCustomTask(request: CustomTaskRequest): Promise<string> {
    const response = await apiClient.post(`${this.baseUrl}/tasks`, request);
    return response.data;
  }

  // 任务优化
  async optimizeForTask(
    taskName: string, 
    customConfig?: Record<string, any>
  ): Promise<{
    status: string;
    task_name: string;
    message: string;
  }> {
    const response = await apiClient.post(`${this.baseUrl}/optimize/${taskName}`, {
      custom_config: customConfig
    });
    return response.data;
  }

  // 算法比较
  async compareAlgorithms(
    taskName: string,
    algorithms: string[] = ['tpe', 'cmaes', 'random']
  ): Promise<{
    status: string;
    task_name: string;
    algorithms: string[];
    message: string;
  }> {
    const response = await apiClient.post(`${this.baseUrl}/compare-algorithms/${taskName}`, null, {
      params: { algorithms }
    });
    return response.data;
  }

  // 获取资源状态
  async getResourceStatus(): Promise<ResourceStats> {
    const response = await apiClient.get(`${this.baseUrl}/resource-status`);
    return response.data;
  }

  // 获取活跃实验
  async getActiveExperiments(): Promise<any[]> {
    const response = await apiClient.get(`${this.baseUrl}/active-experiments`);
    return response.data;
  }

  // 健康检查
  async healthCheck(): Promise<{
    status: string;
    timestamp: string;
    services: Record<string, string>;
  }> {
    const response = await apiClient.get(`${this.baseUrl}/health`);
    return response.data;
  }

  // 获取可用算法
  async getAvailableAlgorithms(): Promise<AlgorithmInfo> {
    const response = await apiClient.get(`${this.baseUrl}/algorithms`);
    return response.data;
  }

  // 获取剪枝策略
  async getPruningStrategies(): Promise<PruningInfo> {
    const response = await apiClient.get(`${this.baseUrl}/pruning-strategies`);
    return response.data;
  }

  // 获取参数类型
  async getParameterTypes(): Promise<ParameterTypeInfo> {
    const response = await apiClient.get(`${this.baseUrl}/parameter-types`);
    return response.data;
  }

  // 流式获取优化进度（使用Server-Sent Events）
  async streamOptimizationProgress(
    experimentId: string,
    onProgress: (progress: ExperimentProgress) => void,
    onError?: (error: Error) => void
  ): Promise<EventSource> {
    try {
      const eventSource = new EventSource(buildApiUrl(`${this.baseUrl}/experiments/${experimentId}/stream-progress`));
      
      eventSource.onmessage = (event) => {
        try {
          const progress = JSON.parse(event.data) as ExperimentProgress;
          onProgress(progress);
        } catch (error) {
          logger.error('解析进度数据失败:', error);
          if (onError) onError(new Error('数据格式错误'));
        }
      };

      eventSource.onerror = (event) => {
        logger.error('流式进度连接出错:', event);
        if (onError) onError(new Error('连接中断'));
      };

      return eventSource;
    } catch (error) {
      logger.error('创建流式进度连接失败:', error);
      if (onError) onError(error as Error);
      throw error;
    }
  }

  // 批量实验管理
  async batchStartExperiments(experimentIds: string[]): Promise<{
    success: string[];
    failed: Array<{ id: string; error: string }>;
  }> {
    try {
      const results = await Promise.allSettled(
        experimentIds.map(id => this.startExperiment(id))
      );

      const success: string[] = [];
      const failed: Array<{ id: string; error: string }> = [];

      results.forEach((result, index) => {
        if (result.status === 'fulfilled') {
          success.push(experimentIds[index]);
        } else {
          failed.push({
            id: experimentIds[index],
            error: result.reason?.message || '启动失败'
          });
        }
      });

      return { success, failed };
    } catch (error) {
      logger.error('批量启动实验失败:', error);
      throw new Error('批量启动实验失败，请检查网络连接或稍后重试');
    }
  }

  // 批量停止实验
  async batchStopExperiments(experimentIds: string[]): Promise<{
    success: string[];
    failed: Array<{ id: string; error: string }>;
  }> {
    try {
      const results = await Promise.allSettled(
        experimentIds.map(id => this.stopExperiment(id))
      );

      const success: string[] = [];
      const failed: Array<{ id: string; error: string }> = [];

      results.forEach((result, index) => {
        if (result.status === 'fulfilled') {
          success.push(experimentIds[index]);
        } else {
          failed.push({
            id: experimentIds[index],
            error: result.reason?.message || '停止失败'
          });
        }
      });

      return { success, failed };
    } catch (error) {
      logger.error('批量停止实验失败:', error);
      throw new Error('批量停止实验失败，请检查网络连接或稍后重试');
    }
  }

  // 实验模板管理
  async saveExperimentTemplate(
    templateName: string,
    experiment: any
  ): Promise<{ success: boolean; template_id: string }> {
    try {
      const response = await apiClient.post(`${this.baseUrl}/templates`, {
        name: templateName,
        config: experiment
      });
      return response.data;
    } catch (error) {
      logger.error('保存实验模板失败:', error);
      throw new Error('保存实验模板失败，请检查网络连接或稍后重试');
    }
  }

  async getExperimentTemplates(): Promise<Array<{
    id: string;
    name: string;
    description?: string;
    created_at: string;
    config: any;
  }>> {
    try {
      const response = await apiClient.get(`${this.baseUrl}/templates`);
      return response.data;
    } catch (error) {
      logger.error('获取实验模板失败:', error);
      throw new Error('获取实验模板失败，请检查网络连接或稍后重试');
    }
  }

  // 高级分析功能
  async performParameterSensitivityAnalysis(
    experimentId: string
  ): Promise<{
    sensitivity_scores: Record<string, number>;
    interaction_matrix: Array<Array<number>>;
    parameter_names: string[];
    recommendations: string[];
  }> {
    try {
      const response = await apiClient.post(`${this.baseUrl}/experiments/${experimentId}/sensitivity-analysis`);
      return response.data;
    } catch (error) {
      logger.error('参数敏感性分析失败:', error);
      throw new Error('参数敏感性分析失败，请检查网络连接或稍后重试');
    }
  }
}

export const hyperparameterServiceEnhanced = new HyperparameterServiceEnhanced();
