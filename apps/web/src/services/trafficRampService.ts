import { apiClient } from './apiClient'

export enum RampStrategy {
  LINEAR = 'linear',
  EXPONENTIAL = 'exponential',
  LOGARITHMIC = 'logarithmic',
  STEP = 'step',
  CUSTOM = 'custom',
}

export enum RampStatus {
  CREATED = 'created',
  RUNNING = 'running',
  PAUSED = 'paused',
  COMPLETED = 'completed',
  FAILED = 'failed',
  ROLLBACK = 'rollback',
}

export enum RolloutPhase {
  CANARY = 'canary',
  PILOT = 'pilot',
  BETA = 'beta',
  GRADUAL = 'gradual',
  FULL = 'full',
}

export interface CreateRampPlanRequest {
  experiment_id: string
  variant: string
  strategy: RampStrategy
  start_percentage: number
  target_percentage: number
  duration_hours: number
  num_steps: number
  health_checks?: any
  rollback_conditions?: any
}

export interface RampPlan {
  plan_id: string
  experiment_id: string
  variant: string
  strategy: RampStrategy
  start_percentage: number
  target_percentage: number
  duration_hours: number
  num_steps: number
  created_at: string
  steps: RampStep[]
}

export interface RampStep {
  step: number
  target: number
  duration_minutes: number
}

export interface RampExecution {
  exec_id: string
  plan_id: string
  experiment_id: string
  status: RampStatus
  current_step: number
  current_percentage: number
  started_at?: string
  completed_at?: string
  rollback_reason?: string
}

export interface LoadBalancerRequest {
  capability: string
  strategy: string
  tags?: string[]
  requirements?: any
}

export interface GetRecommendedPlanRequest {
  experiment_id: string
  risk_level: string
}

export interface QuickRampRequest {
  experiment_id: string
  phase: RolloutPhase
  duration_hours?: number
}

export interface StrategyInfo {
  value: RampStrategy
  name: string
  description: string
  use_case: string
}

export interface PhaseInfo {
  value: RolloutPhase
  name: string
  range: string
  description: string
}

export class TrafficRampService {
  private baseUrl = '/traffic-ramp'

  async createRampPlan(request: CreateRampPlanRequest): Promise<any> {
    const response = await apiClient.post(`${this.baseUrl}/plans`, request)
    return response.data
  }

  async startRamp(planId: string): Promise<any> {
    const response = await apiClient.post(`${this.baseUrl}/start`, {
      plan_id: planId,
    })
    return response.data
  }

  async pauseRamp(execId: string): Promise<any> {
    const response = await apiClient.post(`${this.baseUrl}/pause`, {
      exec_id: execId,
    })
    return response.data
  }

  async resumeRamp(execId: string): Promise<any> {
    const response = await apiClient.post(`${this.baseUrl}/resume`, {
      exec_id: execId,
    })
    return response.data
  }

  async rollbackRamp(execId: string, reason: string): Promise<any> {
    const response = await apiClient.post(`${this.baseUrl}/rollback`, null, {
      params: { exec_id: execId, reason },
    })
    return response.data
  }

  async getRampStatus(execId: string): Promise<any> {
    const response = await apiClient.get(`${this.baseUrl}/status/${execId}`)
    return response.data
  }

  async listPlans(experimentId?: string): Promise<any> {
    const response = await apiClient.get(`${this.baseUrl}/plans`, {
      params: { experiment_id: experimentId },
    })
    return response.data
  }

  async listExecutions(
    experimentId?: string,
    status?: RampStatus
  ): Promise<any> {
    const response = await apiClient.get(`${this.baseUrl}/executions`, {
      params: { experiment_id: experimentId, status },
    })
    return response.data
  }

  async getRecommendedPlan(request: GetRecommendedPlanRequest): Promise<any> {
    const response = await apiClient.post(
      `${this.baseUrl}/recommended-plan`,
      request
    )
    return response.data
  }

  async quickRamp(request: QuickRampRequest): Promise<any> {
    const response = await apiClient.post(`${this.baseUrl}/quick-ramp`, request)
    return response.data
  }

  async listStrategies(): Promise<{
    success: boolean
    strategies: StrategyInfo[]
  }> {
    const response = await apiClient.get(`${this.baseUrl}/strategies`)
    return response.data
  }

  async listPhases(): Promise<{ success: boolean; phases: PhaseInfo[] }> {
    const response = await apiClient.get(`${this.baseUrl}/phases`)
    return response.data
  }

  async getCurrentPhase(execId: string): Promise<any> {
    const response = await apiClient.get(
      `${this.baseUrl}/current-phase/${execId}`
    )
    return response.data
  }

  async healthCheck(): Promise<any> {
    const response = await apiClient.get(`${this.baseUrl}/health`)
    return response.data
  }

  // 便捷方法
  async createLinearRamp(
    experimentId: string,
    targetPercentage: number,
    durationHours: number
  ): Promise<any> {
    return this.createRampPlan({
      experiment_id: experimentId,
      variant: 'treatment',
      strategy: RampStrategy.LINEAR,
      start_percentage: 0,
      target_percentage: targetPercentage,
      duration_hours: durationHours,
      num_steps: Math.max(5, Math.floor(durationHours)),
    })
  }

  async createCanaryRamp(experimentId: string): Promise<any> {
    return this.quickRamp({
      experiment_id: experimentId,
      phase: RolloutPhase.CANARY,
    })
  }

  async createFullRollout(experimentId: string): Promise<any> {
    return this.quickRamp({
      experiment_id: experimentId,
      phase: RolloutPhase.FULL,
    })
  }

  async createBetaRamp(
    experimentId: string,
    durationHours?: number
  ): Promise<any> {
    return this.quickRamp({
      experiment_id: experimentId,
      phase: RolloutPhase.BETA,
      duration_hours: durationHours,
    })
  }

  async getActiveRamps(): Promise<RampExecution[]> {
    const response = await this.listExecutions()
    if (response.success) {
      return response.executions.filter(
        (exec: RampExecution) =>
          exec.status === RampStatus.RUNNING ||
          exec.status === RampStatus.PAUSED
      )
    }
    return []
  }

  async getAllRampsForExperiment(
    experimentId: string
  ): Promise<RampExecution[]> {
    const response = await this.listExecutions(experimentId)
    return response.success ? response.executions : []
  }

  async emergencyRollback(execId: string): Promise<any> {
    return this.rollbackRamp(execId, 'Emergency rollback triggered by user')
  }

  async getStrategyRecommendation(
    riskLevel: 'low' | 'medium' | 'high'
  ): Promise<RampStrategy> {
    switch (riskLevel) {
      case 'low':
        return RampStrategy.LINEAR
      case 'medium':
        return RampStrategy.EXPONENTIAL
      case 'high':
        return RampStrategy.LOGARITHMIC
      default:
        return RampStrategy.LINEAR
    }
  }

  async calculateOptimalSteps(
    durationHours: number,
    strategy: RampStrategy
  ): Promise<number> {
    switch (strategy) {
      case RampStrategy.LINEAR:
        return Math.max(5, Math.min(20, Math.floor(durationHours / 2)))
      case RampStrategy.EXPONENTIAL:
        return Math.max(8, Math.min(15, Math.floor(durationHours / 1.5)))
      case RampStrategy.LOGARITHMIC:
        return Math.max(10, Math.min(25, Math.floor(durationHours)))
      case RampStrategy.STEP:
        return Math.max(3, Math.min(10, Math.floor(durationHours / 4)))
      default:
        return 10
    }
  }
}

export const trafficRampService = new TrafficRampService()
