import { buildWsUrl } from '../utils/apiBase'
import apiClient from './apiClient'

import { logger } from '../utils/logger'
// ==================== 类型定义 ====================

export interface EmotionData {
  emotions: Record<string, number>
  intensity: number
  confidence: number
  context?: string
}

export interface ParticipantData {
  participant_id: string
  name: string
  emotion_data: EmotionData
  cultural_indicators?: Record<string, any>
  relationship_history?: Array<Record<string, any>>
}

export interface SocialEnvironmentData {
  scenario: string
  participants_count: number
  formality_level: number // 0-1
  emotional_intensity: number // 0-1
  time_pressure: number // 0-1
  cultural_context?: string
}

export interface AnalysisRequest {
  session_id: string
  participants: ParticipantData[]
  social_environment: SocialEnvironmentData
  analysis_types?: string[]
  real_time?: boolean
}

export interface GroupEmotionResponse {
  session_id: string
  group_emotion_state: Record<string, any>
  individual_emotions: Record<string, Record<string, number>>
  emotional_contagion: Record<string, any>
  recommendations: string[]
  confidence_score: number
}

export interface RelationshipAnalysisResponse {
  session_id: string
  relationship_dynamics: Record<string, any>
  interaction_patterns: Record<string, any>
  relationship_health_scores: Record<string, number>
  recommendations: string[]
}

export interface SocialDecisionResponse {
  session_id: string
  decision_recommendations: Array<{
    decision_type: string
    options: Array<{
      option_id: string
      description: string
      predicted_outcomes: Record<string, number>
      confidence: number
      recommended: boolean
    }>
    context_factors: Record<string, any>
  }>
  social_intelligence_score: number
  cultural_considerations: string[]
}

export interface SocialContextAnalysisResponse {
  session_id: string
  context_analysis: {
    social_environment_type: string
    formality_assessment: {
      level: number
      indicators: string[]
      recommendations: string[]
    }
    power_dynamics: {
      hierarchy_detected: boolean
      dominant_participants: string[]
      influence_patterns: Record<string, any>
    }
    cultural_factors: {
      detected_cultures: string[]
      potential_conflicts: string[]
      adaptation_strategies: string[]
    }
  }
  adaptation_recommendations: string[]
  confidence_metrics: Record<string, number>
}

export interface CulturalAdaptationResponse {
  session_id: string
  cultural_analysis: {
    detected_cultures: Array<{
      culture_id: string
      confidence: number
      key_indicators: string[]
    }>
    cross_cultural_dynamics: {
      potential_conflicts: Array<{
        conflict_type: string
        severity: number
        affected_participants: string[]
        mitigation_strategies: string[]
      }>
      communication_styles: Record<string, string>
      value_alignments: Record<string, number>
    }
  }
  adaptation_strategies: Array<{
    strategy_type: string
    description: string
    implementation_steps: string[]
    expected_outcomes: Record<string, number>
  }>
  cultural_intelligence_score: number
}

export interface ComprehensiveAnalysisResponse {
  session_id: string
  analysis_summary: {
    overall_emotional_state: Record<string, any>
    relationship_health: number
    social_cohesion: number
    cultural_harmony: number
    communication_effectiveness: number
  }
  detailed_insights: {
    group_emotion: GroupEmotionResponse
    relationships: RelationshipAnalysisResponse
    social_context: SocialContextAnalysisResponse
    cultural_adaptation: CulturalAdaptationResponse
    decision_recommendations: SocialDecisionResponse
  }
  priority_recommendations: Array<{
    priority: 'high' | 'medium' | 'low'
    category: string
    recommendation: string
    expected_impact: number
  }>
  confidence_score: number
}

export interface SystemAnalytics {
  total_sessions: number
  total_analyses_performed: number
  avg_confidence_score: number
  most_common_scenarios: Array<{
    scenario: string
    frequency: number
  }>
  cultural_diversity_stats: Record<string, number>
  emotional_state_trends: Record<
    string,
    Array<{
      timestamp: string
      value: number
    }>
  >
  relationship_health_trends: Array<{
    date: string
    avg_health_score: number
  }>
  system_performance: {
    avg_response_time: number
    success_rate: number
    error_rate: number
  }
}

// ==================== Service Class ====================

class SocialEmotionalService {
  private baseUrl = '/social-emotional-understanding'

  // ==================== 群体情感分析 ====================

  async analyzeGroupEmotion(
    request: AnalysisRequest
  ): Promise<GroupEmotionResponse> {
    const response = await apiClient.post(
      `${this.baseUrl}/analyze/group-emotion`,
      request
    )
    return response.data
  }

  // ==================== 关系分析 ====================

  async analyzeRelationships(
    request: AnalysisRequest
  ): Promise<RelationshipAnalysisResponse> {
    const response = await apiClient.post(
      `${this.baseUrl}/analyze/relationships`,
      request
    )
    return response.data
  }

  // ==================== 社交上下文分析 ====================

  async analyzeSocialContext(
    request: AnalysisRequest
  ): Promise<SocialContextAnalysisResponse> {
    const response = await apiClient.post(
      `${this.baseUrl}/analyze/social-context`,
      request
    )
    return response.data
  }

  // ==================== 文化适配分析 ====================

  async analyzeCulturalAdaptation(
    request: AnalysisRequest
  ): Promise<CulturalAdaptationResponse> {
    const response = await apiClient.post(
      `${this.baseUrl}/analyze/cultural-adaptation`,
      request
    )
    return response.data
  }

  // ==================== 社交决策生成 ====================

  async generateSocialDecisions(
    request: AnalysisRequest
  ): Promise<SocialDecisionResponse> {
    const response = await apiClient.post(
      `${this.baseUrl}/decisions/generate`,
      request
    )
    return response.data
  }

  // ==================== 综合分析 ====================

  async comprehensiveAnalysis(
    request: AnalysisRequest
  ): Promise<ComprehensiveAnalysisResponse> {
    const response = await apiClient.post(
      `${this.baseUrl}/comprehensive-analysis`,
      request
    )
    return response.data
  }

  // ==================== 系统状态和分析 ====================

  async getHealthCheck(): Promise<{ status: string; timestamp: string }> {
    const response = await apiClient.get(`${this.baseUrl}/health`)
    return response.data
  }

  async getSystemAnalytics(): Promise<SystemAnalytics> {
    const response = await apiClient.get(`${this.baseUrl}/analytics`)
    return response.data
  }

  // ==================== WebSocket 实时分析 ====================

  createWebSocketConnection(
    sessionId: string,
    onMessage: (data: any) => void,
    onError?: (error: Event) => void,
    onClose?: (event: CloseEvent) => void
  ): WebSocket {
    const wsUrl = buildWsUrl(`${this.baseUrl}/realtime/${sessionId}`)
    const ws = new WebSocket(wsUrl)

    ws.onopen = () => {
      logger.log(`社会情感分析流已连接: ${sessionId}`)
    }

    ws.onmessage = event => {
      try {
        const data = JSON.parse(event.data)
        onMessage(data)
      } catch (error) {
        logger.error('解析WebSocket消息失败:', error)
      }
    }

    ws.onerror = error => {
      logger.error('WebSocket错误:', error)
      if (onError) onError(error)
      if (
        ws.readyState !== WebSocket.CLOSING &&
        ws.readyState !== WebSocket.CLOSED
      ) {
        ws.close()
      }
    }

    ws.onclose = event => {
      logger.log('WebSocket连接已关闭:', event.code, event.reason)
      if (onClose) onClose(event)
    }

    return ws
  }

  // ==================== 工具方法 ====================

  generateSessionId(): string {
    return `session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`
  }
}

// ==================== 导出 ====================

export const socialEmotionalService = new SocialEmotionalService()
export default socialEmotionalService
