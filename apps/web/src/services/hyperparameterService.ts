import apiClient from './apiClient'

export interface Experiment {
  id: string
  name: string
  status: 'created' | 'running' | 'completed' | 'failed' | 'stopped'
  algorithm: string
  objective: 'minimize' | 'maximize'
  created_at: string
  updated_at?: string
  best_value?: number
  total_trials?: number
  successful_trials?: number
  failed_trials?: number
  config?: ExperimentConfig
  metadata?: Record<string, any>
}

export interface ExperimentConfig {
  study_name: string
  algorithm: 'random' | 'grid' | 'bayesian' | 'tpe' | 'cmaes'
  objective: 'minimize' | 'maximize'
  metric_name: string
  max_trials: number
  parallel_trials?: number
  parameters: ParameterConfig[]
  early_stopping?: EarlyStoppingConfig
}

export interface ParameterConfig {
  name: string
  type: 'float' | 'int' | 'categorical' | 'log_uniform'
  low?: number
  high?: number
  choices?: (string | number)[]
  step?: number
}

export interface EarlyStoppingConfig {
  enabled: boolean
  min_trials: number
  patience: number
  min_improvement: number
}

export interface Trial {
  id: string
  experiment_id: string
  trial_number: number
  parameters: Record<string, any>
  value?: number
  state: 'running' | 'complete' | 'pruned' | 'fail'
  start_time?: string
  end_time?: string
  duration?: number
  error_message?: string
  intermediate_values?: Array<{ step: number; value: number }>
}

export interface CreateExperimentRequest {
  name: string
  algorithm: string
  objective: 'minimize' | 'maximize'
  metric_name: string
  max_trials: number
  parameters: ParameterConfig[]
  early_stopping?: EarlyStoppingConfig
}

export interface TrialSuggestion {
  trial_id: string
  parameters: Record<string, any>
}

export interface TrialResult {
  trial_id: string
  value: number
  metadata?: Record<string, any>
}

export interface ExperimentStatistics {
  total_trials: number
  successful_trials: number
  failed_trials: number
  pruned_trials: number
  best_value: number
  best_parameters: Record<string, any>
  average_duration: number
  convergence_history: Array<{ trial: number; best_value: number }>
}

class HyperparameterService {
  private baseUrl = '/hyperparameter-optimization'

  // 实验管理
  async listExperiments(status?: string): Promise<Experiment[]> {
    const params = status ? { status } : {}
    const response = await apiClient.get(`${this.baseUrl}/experiments`, {
      params,
    })
    return response.data
  }

  async getExperiment(experimentId: string): Promise<Experiment> {
    const response = await apiClient.get(
      `${this.baseUrl}/experiments/${experimentId}`
    )
    return response.data
  }

  async createExperiment(
    request: CreateExperimentRequest
  ): Promise<Experiment> {
    const response = await apiClient.post(
      `${this.baseUrl}/experiments`,
      request
    )
    return response.data
  }

  async deleteExperiment(experimentId: string): Promise<void> {
    await apiClient.delete(`${this.baseUrl}/experiments/${experimentId}`)
  }

  async stopExperiment(experimentId: string): Promise<void> {
    await apiClient.post(`${this.baseUrl}/experiments/${experimentId}/stop`)
  }

  async resumeExperiment(experimentId: string): Promise<void> {
    await apiClient.post(`${this.baseUrl}/experiments/${experimentId}/resume`)
  }

  // Trial管理
  async listTrials(experimentId: string, state?: string): Promise<Trial[]> {
    const params = state ? { state } : {}
    const response = await apiClient.get(
      `${this.baseUrl}/experiments/${experimentId}/trials`,
      { params }
    )
    return response.data
  }

  async getTrial(experimentId: string, trialId: string): Promise<Trial> {
    const response = await apiClient.get(
      `${this.baseUrl}/experiments/${experimentId}/trials/${trialId}`
    )
    return response.data
  }

  async suggestTrial(experimentId: string): Promise<TrialSuggestion> {
    const response = await apiClient.post(
      `${this.baseUrl}/experiments/${experimentId}/suggest`
    )
    return response.data
  }

  async reportTrialResult(
    experimentId: string,
    result: TrialResult
  ): Promise<void> {
    await apiClient.post(
      `${this.baseUrl}/experiments/${experimentId}/report`,
      result
    )
  }

  async pruneTriaml(experimentId: string, trialId: string): Promise<void> {
    await apiClient.post(
      `${this.baseUrl}/experiments/${experimentId}/trials/${trialId}/prune`
    )
  }

  // 统计和分析
  async getExperimentStatistics(
    experimentId: string
  ): Promise<ExperimentStatistics> {
    const response = await apiClient.get(
      `${this.baseUrl}/experiments/${experimentId}/statistics`
    )
    return response.data
  }

  async getParameterImportance(
    experimentId: string
  ): Promise<Record<string, number>> {
    const response = await apiClient.get(
      `${this.baseUrl}/experiments/${experimentId}/importance`
    )
    return response.data
  }

  async getOptimizationHistory(experimentId: string): Promise<
    Array<{
      trial: number
      value: number
      parameters: Record<string, any>
    }>
  > {
    const response = await apiClient.get(
      `${this.baseUrl}/experiments/${experimentId}/history`
    )
    return response.data
  }

  async getParallelCoordinates(experimentId: string): Promise<any> {
    const response = await apiClient.get(
      `${this.baseUrl}/experiments/${experimentId}/parallel-coordinates`
    )
    return response.data
  }

  // 预设任务
  async listPresetTasks(): Promise<string[]> {
    const response = await apiClient.get(`${this.baseUrl}/tasks`)
    return response.data
  }

  async getPresetTaskConfig(taskName: string): Promise<ExperimentConfig> {
    const response = await apiClient.get(`${this.baseUrl}/tasks/${taskName}`)
    return response.data
  }

  // 导出和报告
  async exportExperiment(
    experimentId: string,
    format: 'json' | 'csv' | 'pdf' = 'json'
  ): Promise<Blob> {
    const response = await apiClient.get(
      `${this.baseUrl}/experiments/${experimentId}/export`,
      {
        params: { format },
        responseType: 'blob',
      }
    )
    return response.data
  }

  async generateReport(experimentId: string): Promise<{
    summary: string
    best_configuration: Record<string, any>
    recommendations: string[]
    visualizations: any[]
  }> {
    const response = await apiClient.post(
      `${this.baseUrl}/experiments/${experimentId}/report`
    )
    return response.data
  }

  // 批量操作
  async batchSuggest(
    experimentId: string,
    count: number
  ): Promise<TrialSuggestion[]> {
    const response = await apiClient.post(
      `${this.baseUrl}/experiments/${experimentId}/batch-suggest`,
      {
        count,
      }
    )
    return response.data
  }

  async batchReport(
    experimentId: string,
    results: TrialResult[]
  ): Promise<void> {
    await apiClient.post(
      `${this.baseUrl}/experiments/${experimentId}/batch-report`,
      {
        results,
      }
    )
  }
}

export const hyperparameterService = new HyperparameterService()
