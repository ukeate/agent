import apiClient from './apiClient'

export interface SystemStatus {
  module_name: string
  status: 'running' | 'stopped' | 'error' | 'warning'
  health_score: number
  uptime: number
  cpu_usage: number
  memory_usage: number
  last_updated: string
  error_count: number
  warning_count: number
  performance_metrics: {
    accuracy?: number
    latency?: number
    throughput?: number
    quality_score?: number
    satisfaction?: number
    response_time?: number
    decision_accuracy?: number
    context_score?: number
  }
}

export interface ServiceMetric {
  service_name: string
  requests_count: number
  success_rate: number
  average_latency: number
  error_rate: number
  peak_qps: number
}

export interface UserInteraction {
  id: string
  user_id: string
  timestamp: string
  interaction_type:
    | 'emotion_analysis'
    | 'empathy_response'
    | 'social_decision'
    | 'conflict_resolution'
  emotion_detected: string
  response_generated: string
  satisfaction_score: number
  context: Record<string, any>
}

export interface SystemAlert {
  id: string
  level: 'info' | 'warning' | 'error' | 'critical'
  module: string
  message: string
  timestamp: string
  resolved: boolean
}

export interface Configuration {
  module: string
  settings: Record<string, any>
  enabled: boolean
  last_modified: string
}

export interface EmotionAnalysisRequest {
  text?: string
  audio_url?: string
  context?: Record<string, any>
}

export interface EmotionAnalysisResponse {
  primary_emotion: string
  emotion_scores: Record<string, number>
  confidence: number
  cultural_context?: string
  social_cues?: string[]
}

export interface EmpathyResponseRequest {
  user_emotion: string
  context: string
  cultural_background?: string
  response_style?: 'supportive' | 'neutral' | 'encouraging'
}

export interface EmpathyResponseResponse {
  response_text: string
  response_type: string
  empathy_level: number
  cultural_adaptation: boolean
}

class SocialEmotionService {
  private baseUrl = '/social-emotion'

  // System Status APIs
  async getSystemStatus(): Promise<SystemStatus[]> {
    const response = await apiClient.get(`${this.baseUrl}/status`)
    return response.data
  }

  async getModuleStatus(moduleName: string): Promise<SystemStatus> {
    const response = await apiClient.get(`${this.baseUrl}/status/${moduleName}`)
    return response.data
  }

  // Service Metrics APIs
  async getServiceMetrics(): Promise<ServiceMetric[]> {
    const response = await apiClient.get(`${this.baseUrl}/metrics`)
    return response.data
  }

  // User Interactions APIs
  async getUserInteractions(limit: number = 100): Promise<UserInteraction[]> {
    const response = await apiClient.get(`${this.baseUrl}/interactions`, {
      params: { limit },
    })
    return response.data
  }

  async getUserInteractionStats(userId?: string): Promise<{
    total_interactions: number
    average_satisfaction: number
    emotion_distribution: Record<string, number>
    interaction_types: Record<string, number>
  }> {
    const params = userId ? { user_id: userId } : {}
    const response = await apiClient.get(`${this.baseUrl}/interactions/stats`, {
      params,
    })
    return response.data
  }

  // System Alerts APIs
  async getSystemAlerts(
    unresolved_only: boolean = false
  ): Promise<SystemAlert[]> {
    const response = await apiClient.get(`${this.baseUrl}/alerts`, {
      params: { unresolved_only },
    })
    return response.data
  }

  async resolveAlert(alertId: string): Promise<void> {
    await apiClient.post(`${this.baseUrl}/alerts/${alertId}/resolve`)
  }

  // Configuration APIs
  async getConfigurations(): Promise<Configuration[]> {
    const response = await apiClient.get(`${this.baseUrl}/configurations`)
    return response.data
  }

  async updateConfiguration(
    module: string,
    settings: Record<string, any>
  ): Promise<Configuration> {
    const response = await apiClient.put(
      `${this.baseUrl}/configurations/${module}`,
      settings
    )
    return response.data
  }

  // Emotion Analysis APIs
  async analyzeEmotion(
    request: EmotionAnalysisRequest
  ): Promise<EmotionAnalysisResponse> {
    const response = await apiClient.post(
      `${this.baseUrl}/analyze-emotion`,
      request
    )
    return response.data
  }

  // Empathy Response APIs
  async generateEmpathyResponse(
    request: EmpathyResponseRequest
  ): Promise<EmpathyResponseResponse> {
    const response = await apiClient.post(
      `${this.baseUrl}/generate-empathy`,
      request
    )
    return response.data
  }

  // Social Decision APIs
  async makeSocialDecision(context: {
    situation: string
    participants: string[]
    cultural_context?: string
    desired_outcome?: string
  }): Promise<{
    recommended_action: string
    confidence: number
    alternative_actions: string[]
    cultural_considerations: string[]
  }> {
    const response = await apiClient.post(
      `${this.baseUrl}/social-decision`,
      context
    )
    return response.data
  }

  // Privacy Protection APIs
  async checkPrivacy(data: Record<string, any>): Promise<{
    is_safe: boolean
    privacy_risks: string[]
    recommendations: string[]
  }> {
    const response = await apiClient.post(`${this.baseUrl}/privacy-check`, data)
    return response.data
  }

  // Module Control APIs
  async startModule(
    moduleName: string
  ): Promise<{ message: string; status: string }> {
    const response = await apiClient.post(
      `${this.baseUrl}/modules/${moduleName}/start`
    )
    return response.data
  }

  async stopModule(
    moduleName: string
  ): Promise<{ message: string; status: string }> {
    const response = await apiClient.post(
      `${this.baseUrl}/modules/${moduleName}/stop`
    )
    return response.data
  }

  async restartModule(
    moduleName: string
  ): Promise<{ message: string; status: string }> {
    const response = await apiClient.post(
      `${this.baseUrl}/modules/${moduleName}/restart`
    )
    return response.data
  }
}

export const socialEmotionService = new SocialEmotionService()
