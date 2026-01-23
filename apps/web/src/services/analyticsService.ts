import { buildWsUrl } from '../utils/apiBase'
import apiClient from './apiClient'

import { logger } from '../utils/logger'
// ==================== 类型定义 ====================

export interface BehaviorEvent {
  event_id: string
  user_id: string
  session_id?: string
  event_type: string
  timestamp: string
  properties?: Record<string, any>
  context?: Record<string, any>
}

export interface EventSubmissionRequest {
  events: BehaviorEvent[]
  batch_id?: string
}

export interface EventSubmissionResponse {
  status: 'accepted' | 'rejected'
  event_count: number
  batch_id: string
  message: string
}

export interface AnalysisRequest {
  user_id?: string
  session_id?: string
  start_time?: string
  end_time?: string
  event_types?: string[]
  analysis_types?: ('patterns' | 'anomalies' | 'insights')[]
}

export interface AnalysisResult {
  patterns?: PatternAnalysis
  anomalies?: AnomalyAnalysis
  insights?: InsightAnalysis
  summary: AnalysisSummary
  metadata: {
    analysis_id: string
    timestamp: string
    processing_time: number
  }
}

export interface PatternAnalysis {
  frequent_patterns: Pattern[]
  user_segments: UserSegment[]
  temporal_patterns: TemporalPattern[]
  correlation_matrix?: Record<string, Record<string, number>>
}

export interface Pattern {
  pattern_id: string
  description: string
  events: string[]
  frequency: number
  confidence: number
  users_affected: number
}

export interface UserSegment {
  segment_id: string
  name: string
  size: number
  characteristics: Record<string, any>
  typical_behavior: string[]
}

export interface TemporalPattern {
  pattern_type: 'daily' | 'weekly' | 'monthly' | 'seasonal'
  peak_times: string[]
  low_times: string[]
  trend: 'increasing' | 'stable' | 'decreasing'
}

export interface AnomalyAnalysis {
  detected_anomalies: Anomaly[]
  anomaly_score: number
  risk_level: 'low' | 'medium' | 'high' | 'critical'
  affected_users: string[]
  recommendations: string[]
}

export interface Anomaly {
  anomaly_id: string
  type: string
  severity: 'low' | 'medium' | 'high' | 'critical'
  description: string
  detected_at: string
  confidence: number
  affected_events: string[]
  possible_causes?: string[]
}

export interface InsightAnalysis {
  key_insights: Insight[]
  opportunities: Opportunity[]
  risks: Risk[]
  recommendations: Recommendation[]
}

export interface Insight {
  insight_id: string
  type: string
  title: string
  description: string
  impact: 'low' | 'medium' | 'high'
  confidence: number
  supporting_data?: any
}

export interface Opportunity {
  opportunity_id: string
  title: string
  description: string
  potential_impact: string
  effort_required: 'low' | 'medium' | 'high'
  priority: number
}

export interface Risk {
  risk_id: string
  title: string
  description: string
  probability: number
  impact: 'low' | 'medium' | 'high'
  mitigation_strategies: string[]
}

export interface Recommendation {
  recommendation_id: string
  title: string
  description: string
  priority: 'low' | 'medium' | 'high' | 'urgent'
  expected_impact: string
  implementation_steps: string[]
}

export interface AnalysisSummary {
  total_events_analyzed: number
  unique_users: number
  unique_sessions: number
  time_range: {
    start: string
    end: string
  }
  key_metrics: Record<string, number>
}

export interface UserProfile {
  user_id: string
  first_seen: string
  last_seen: string
  total_events: number
  total_sessions: number
  avg_session_duration: number
  behavior_profile: {
    activity_level: 'low' | 'medium' | 'high'
    engagement_score: number
    retention_risk: 'low' | 'medium' | 'high'
    typical_actions: string[]
    preferred_features: string[]
  }
  segments: string[]
  tags: string[]
}

export interface SessionInfo {
  session_id: string
  user_id: string
  start_time: string
  end_time?: string
  duration: number
  event_count: number
  page_views: number
  actions_performed: string[]
  conversion_events: string[]
  session_quality: 'good' | 'average' | 'poor'
}

export interface ReportRequest {
  report_type: 'comprehensive' | 'summary' | 'custom'
  format: 'json' | 'html' | 'pdf'
  filters?: Record<string, any>
  include_visualizations?: boolean
}

export interface Report {
  report_id: string
  type: string
  generated_at: string
  content: any
  visualizations?: any[]
  download_url?: string
}

export interface RealtimeStats {
  active_users: number
  active_sessions: number
  events_per_minute: number
  current_anomalies: number
  system_health: 'good' | 'degraded' | 'critical'
  trending_events: Array<{
    event_type: string
    count: number
    trend: 'up' | 'down' | 'stable'
  }>
}

// ==================== Service Class ====================

class AnalyticsService {
  private baseUrl = '/analytics'

  // ==================== 事件管理 ====================

  async submitEvents(
    request: EventSubmissionRequest
  ): Promise<EventSubmissionResponse> {
    const response = await apiClient.post(`${this.baseUrl}/events`, request)
    return response.data
  }

  async getEvents(params?: {
    user_id?: string
    session_id?: string
    event_type?: string
    start_time?: string
    end_time?: string
    limit?: number
    offset?: number
  }): Promise<{
    events: BehaviorEvent[]
    total: number
    has_more: boolean
  }> {
    const response = await apiClient.get(`${this.baseUrl}/events`, { params })
    return response.data
  }

  async getEventTypes(): Promise<string[]> {
    const response = await apiClient.get(`${this.baseUrl}/event-types`)
    return response.data.event_types || []
  }

  // ==================== 会话管理 ====================

  async getSessions(params?: {
    user_id?: string
    start_time?: string
    end_time?: string
    min_duration?: number
    limit?: number
    offset?: number
  }): Promise<{
    sessions: SessionInfo[]
    total: number
    has_more: boolean
  }> {
    const response = await apiClient.get(`${this.baseUrl}/sessions`, { params })
    return response.data
  }

  async getSessionDetails(sessionId: string): Promise<SessionInfo> {
    const response = await apiClient.get(
      `${this.baseUrl}/sessions/${sessionId}`
    )
    return response.data
  }

  // ==================== 分析功能 ====================

  async analyze(request: AnalysisRequest): Promise<AnalysisResult> {
    const response = await apiClient.post(`${this.baseUrl}/analyze`, request)
    return response.data
  }

  async detectAnomalies(params?: {
    time_window?: number
    sensitivity?: 'low' | 'medium' | 'high'
    user_id?: string
  }): Promise<AnomalyAnalysis> {
    const response = await apiClient.post(
      `${this.baseUrl}/anomalies/detect`,
      params || {}
    )
    return response.data
  }

  async getPatterns(params?: {
    min_support?: number
    min_confidence?: number
    max_length?: number
  }): Promise<PatternAnalysis> {
    const response = await apiClient.get(`${this.baseUrl}/patterns`, { params })
    return response.data
  }

  async getInsights(params?: {
    insight_types?: string[]
    min_confidence?: number
  }): Promise<InsightAnalysis> {
    const response = await apiClient.get(`${this.baseUrl}/insights`, { params })
    return response.data
  }

  // ==================== 用户分析 ====================

  async getUserProfile(userId: string): Promise<UserProfile> {
    const response = await apiClient.get(
      `${this.baseUrl}/users/${userId}/profile`
    )
    return response.data
  }

  async getUserSegments(userId: string): Promise<UserSegment[]> {
    const response = await apiClient.get(
      `${this.baseUrl}/users/${userId}/segments`
    )
    return response.data.segments || []
  }

  async getUserJourney(
    userId: string,
    params?: {
      start_time?: string
      end_time?: string
    }
  ): Promise<{
    journey: BehaviorEvent[]
    milestones: any[]
    conversion_funnel?: any
  }> {
    const response = await apiClient.get(
      `${this.baseUrl}/users/${userId}/journey`,
      { params }
    )
    return response.data
  }

  // ==================== 报告生成 ====================

  async generateReport(request: ReportRequest): Promise<Report> {
    const response = await apiClient.post(`${this.baseUrl}/reports`, request)
    return response.data
  }

  async getReport(reportId: string): Promise<Report> {
    const response = await apiClient.get(`${this.baseUrl}/reports/${reportId}`)
    return response.data
  }

  async listReports(params?: {
    report_type?: string
    limit?: number
    offset?: number
  }): Promise<{
    reports: Report[]
    total: number
  }> {
    const response = await apiClient.get(`${this.baseUrl}/reports`, { params })
    return response.data
  }

  async downloadReport(
    reportId: string,
    format: 'json' | 'html' | 'pdf'
  ): Promise<Blob> {
    const response = await apiClient.get(
      `${this.baseUrl}/reports/${reportId}/download`,
      {
        params: { format },
        responseType: 'blob',
      }
    )
    return response.data
  }

  // ==================== 实时监控 ====================

  async getRealtimeStats(): Promise<RealtimeStats> {
    const response = await apiClient.get(`${this.baseUrl}/realtime/stats`)
    return response.data
  }

  async subscribeToRealtimeUpdates(
    onUpdate: (data: any) => void,
    filters?: Record<string, any>
  ): Promise<() => void> {
    // WebSocket连接实现
    const wsUrl = buildWsUrl(`${this.baseUrl}/realtime/stream`)
    const ws = new WebSocket(wsUrl)

    ws.onopen = () => {
      if (filters) {
        ws.send(JSON.stringify({ type: 'subscribe', filters }))
      }
    }

    ws.onmessage = event => {
      try {
        const data = JSON.parse(event.data)
        onUpdate(data)
      } catch (error) {
        logger.error('解析实时更新失败:', error)
      }
    }

    ws.onerror = error => {
      logger.error('WebSocket错误:', error)
      if (
        ws.readyState !== WebSocket.CLOSING &&
        ws.readyState !== WebSocket.CLOSED
      ) {
        ws.close()
      }
    }

    // 返回取消订阅函数
    return () => {
      ws.close()
    }
  }

  // ==================== 统计分析摘要 ====================

  async getStatisticalSummary(): Promise<{
    datasets_analyzed: number
    statistical_tests_performed: number
    hypothesis_tests: {
      total: number
      rejected_null: number
      accepted_null: number
      significance_level: number
      power_analysis: {
        average_power: number
        tests_with_adequate_power: number
        underpowered_tests: number
      }
      effect_sizes: {
        small: number
        medium: number
        large: number
      }
    }
    regression_analysis: {
      models_fitted: number
      r_squared_distribution: {
        excellent: number
        good: number
        moderate: number
        poor: number
      }
      prediction_accuracy: number
    }
    correlation_analysis: {
      correlations_computed: number
      significant_correlations: number
      strong_correlations: number
      average_correlation_strength: number
    }
    outlier_detection: {
      outliers_detected: number
      outlier_rate: number
      methods_used: string[]
    }
    time_series_analysis: {
      series_analyzed: number
      forecasting_accuracy: number
      trend_patterns: {
        increasing: number
        decreasing: number
        stable: number
        seasonal: number
      }
    }
    multiple_testing_corrections: {
      corrections_applied: number
      bonferroni_corrections: number
      fdr_corrections: number
      family_wise_error_rate: number
    }
    recent_activity: {
      tests_last_24h: number
      experiments_analyzed_today: number
      active_sessions: number
    }
    performance_metrics: {
      average_computation_time_ms: number
      cache_hit_rate: number
      error_rate: number
    }
  }> {
    const response = await apiClient.get(`${this.baseUrl}/statistical-summary`)
    return response.data
  }

  // ==================== 配置和管理 ====================

  async getConfiguration(): Promise<{
    anomaly_detection_enabled: boolean
    pattern_mining_enabled: boolean
    realtime_processing_enabled: boolean
    data_retention_days: number
    processing_interval_seconds: number
  }> {
    const response = await apiClient.get(`${this.baseUrl}/config`)
    return response.data
  }

  async updateConfiguration(
    config: Partial<{
      anomaly_detection_enabled?: boolean
      pattern_mining_enabled?: boolean
      realtime_processing_enabled?: boolean
      data_retention_days?: number
      processing_interval_seconds?: number
    }>
  ): Promise<{ success: boolean; message: string }> {
    const response = await apiClient.put(`${this.baseUrl}/config`, config)
    return response.data
  }

  async getHealthStatus(): Promise<{
    status: 'healthy' | 'degraded' | 'unhealthy'
    components: {
      event_processing: boolean
      anomaly_detection: boolean
      pattern_mining: boolean
      data_storage: boolean
    }
    metrics: {
      events_processed_today: number
      avg_processing_time_ms: number
      error_rate: number
    }
  }> {
    const response = await apiClient.get(`${this.baseUrl}/health`)
    return response.data
  }
}

// ==================== 导出 ====================

export const analyticsService = new AnalyticsService()
export default analyticsService
