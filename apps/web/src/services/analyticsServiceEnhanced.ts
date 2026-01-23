import { buildApiUrl } from '../utils/apiBase'
import apiClient from './apiClient'

// 增强的行为分析服务，补充高级功能
class AnalyticsServiceEnhanced {
  private baseUrl = '/analytics'

  // ========== 实时数据流和监控 ==========
  async getRealtimeEventStream(): Promise<EventSource> {
    // 使用Server-Sent Events获取实时事件流
    const eventSource = new EventSource(
      buildApiUrl(`${this.baseUrl}/realtime/events`)
    )
    return eventSource
  }

  async getWebSocketStats(): Promise<{
    active_connections: number
    total_connections: number
    messages_sent: number
    messages_received: number
    average_response_time_ms: number
    uptime_seconds: number
    status: 'healthy' | 'degraded' | 'unhealthy'
  }> {
    const response = await apiClient.get(`${this.baseUrl}/ws/stats`)
    return response.data
  }

  async broadcastRealtimeMessage(
    messageType: string,
    data: Record<string, any>,
    userId?: string,
    sessionId?: string
  ): Promise<{
    status: 'success' | 'failed'
    message: string
    timestamp: string
  }> {
    const response = await apiClient.post(
      `${this.baseUrl}/realtime/broadcast`,
      {
        message_type: messageType,
        data,
        user_id: userId,
        session_id: sessionId,
      }
    )
    return response.data
  }

  // ========== 高级报告和导出功能 ==========
  async generateAdvancedReport(params: {
    report_type: 'comprehensive' | 'summary' | 'custom'
    format: 'json' | 'html' | 'pdf'
    filters?: Record<string, any>
    include_visualizations?: boolean
  }): Promise<{
    status: 'accepted' | 'processing' | 'completed'
    report_id: string
    message: string
    estimated_completion_time?: string
  }> {
    const response = await apiClient.post(
      `${this.baseUrl}/reports/generate`,
      params
    )
    return response.data
  }

  async getReportStatus(reportId: string): Promise<{
    report_id: string
    status: 'processing' | 'completed' | 'failed'
    progress?: number
    generated_at?: string
    download_url?: string
    error_message?: string
  }> {
    const response = await apiClient.get(`${this.baseUrl}/reports/${reportId}`)
    return response.data
  }

  async downloadReportFile(
    reportId: string,
    format: 'json' | 'html' | 'pdf' = 'json'
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

  async exportEventData(params: {
    format: 'csv' | 'json' | 'xlsx'
    user_id?: string
    start_time?: string
    end_time?: string
    limit?: number
  }): Promise<Blob> {
    const response = await apiClient.get(`${this.baseUrl}/export/events`, {
      params,
      responseType: 'blob',
    })
    return response.data
  }

  // ========== 高级分析功能 ==========
  async performDeepUserAnalysis(
    userId: string,
    options: {
      include_behavior_timeline: boolean
      include_conversion_funnel: boolean
      include_retention_analysis: boolean
      include_segmentation: boolean
      time_window_days?: number
    }
  ): Promise<{
    user_id: string
    analysis_timestamp: string
    behavior_timeline?: Array<{
      timestamp: string
      event_type: string
      event_details: Record<string, any>
      session_context: Record<string, any>
    }>
    conversion_funnel?: {
      funnel_name: string
      stages: Array<{
        stage_name: string
        users_entered: number
        conversion_rate: number
        avg_time_to_next_stage_hours: number
      }>
      overall_conversion_rate: number
    }
    retention_analysis?: {
      day_1_retention: number
      day_7_retention: number
      day_30_retention: number
      retention_cohort: string
      churn_risk_score: number
    }
    user_segments?: Array<{
      segment_name: string
      confidence_score: number
      segment_characteristics: string[]
      behavioral_traits: Record<string, any>
    }>
    recommendations: string[]
  }> {
    const response = await apiClient.post(
      `${this.baseUrl}/users/${userId}/deep-analysis`,
      options
    )
    return response.data
  }

  async performCohortAnalysis(params: {
    cohort_type: 'acquisition' | 'behavioral' | 'revenue'
    time_period: 'daily' | 'weekly' | 'monthly'
    start_date: string
    end_date: string
    segmentation_criteria?: Record<string, any>
  }): Promise<{
    cohort_type: string
    cohort_data: Array<{
      cohort_period: string
      cohort_size: number
      retention_rates: Record<string, number>
      average_values: Record<string, number>
    }>
    insights: Array<{
      insight_type: string
      description: string
      significance_level: number
      recommendations: string[]
    }>
    summary_statistics: {
      total_cohorts: number
      avg_retention_rate: number
      best_performing_cohort: string
      trend_direction: 'improving' | 'stable' | 'declining'
    }
  }> {
    const response = await apiClient.post(
      `${this.baseUrl}/cohorts/analyze`,
      params
    )
    return response.data
  }

  async performAttributionAnalysis(params: {
    conversion_event: string
    attribution_model:
      | 'first_touch'
      | 'last_touch'
      | 'linear'
      | 'time_decay'
      | 'position_based'
    lookback_window_days: number
    touchpoint_types?: string[]
  }): Promise<{
    attribution_model: string
    conversion_event: string
    attribution_results: Array<{
      touchpoint_type: string
      touchpoint_value: string
      attribution_credit: number
      conversion_count: number
      conversion_rate: number
      average_time_to_conversion_days: number
    }>
    model_comparison: Record<
      string,
      Array<{
        touchpoint: string
        credit_percentage: number
      }>
    >
    insights: Array<{
      finding: string
      impact_level: 'high' | 'medium' | 'low'
      actionable_recommendations: string[]
    }>
  }> {
    const response = await apiClient.post(
      `${this.baseUrl}/attribution/analyze`,
      params
    )
    return response.data
  }

  // ========== 预测性分析 ==========
  async performPredictiveAnalysis(params: {
    prediction_type:
      | 'user_churn'
      | 'ltv'
      | 'next_action'
      | 'conversion_probability'
    user_ids?: string[]
    features_to_include?: string[]
    prediction_horizon_days?: number
    model_type?: 'regression' | 'classification' | 'time_series'
  }): Promise<{
    prediction_type: string
    model_info: {
      model_type: string
      accuracy_score: number
      feature_importance: Record<string, number>
      training_data_size: number
      last_trained: string
    }
    predictions: Array<{
      user_id?: string
      prediction_value: number
      confidence_interval: [number, number]
      key_influencing_factors: string[]
      recommendation: string
    }>
    global_insights: {
      avg_prediction: number
      risk_distribution: Record<string, number>
      key_success_factors: string[]
      improvement_opportunities: string[]
    }
  }> {
    const response = await apiClient.post(
      `${this.baseUrl}/predictions/analyze`,
      params
    )
    return response.data
  }

  async getLifetimeValueAnalysis(params: {
    user_segments?: string[]
    calculation_method: 'historical' | 'predictive' | 'hybrid'
    time_horizon_months?: number
  }): Promise<{
    calculation_method: string
    ltv_analysis: {
      overall_avg_ltv: number
      ltv_distribution: Record<string, number>
      segment_ltv: Array<{
        segment_name: string
        avg_ltv: number
        median_ltv: number
        ltv_growth_rate: number
        customer_count: number
      }>
    }
    revenue_forecasting: {
      projected_revenue_next_12_months: number
      confidence_interval: [number, number]
      key_growth_drivers: string[]
      risk_factors: string[]
    }
    optimization_recommendations: Array<{
      recommendation_type: string
      expected_ltv_improvement: number
      implementation_effort: 'low' | 'medium' | 'high'
      priority_score: number
    }>
  }> {
    const response = await apiClient.post(`${this.baseUrl}/ltv/analyze`, params)
    return response.data
  }

  // ========== 高级异常检测 ==========
  async performAdvancedAnomalyDetection(params: {
    detection_algorithms: (
      | 'isolation_forest'
      | 'local_outlier_factor'
      | 'one_class_svm'
      | 'statistical'
    )[]
    sensitivity: 'low' | 'medium' | 'high' | 'adaptive'
    time_window_hours: number
    min_anomaly_score?: number
    include_user_behavior?: boolean
    include_system_metrics?: boolean
  }): Promise<{
    detection_summary: {
      total_anomalies_detected: number
      high_severity_count: number
      algorithms_used: string[]
      detection_accuracy_estimate: number
    }
    anomalies: Array<{
      anomaly_id: string
      detected_at: string
      severity: 'low' | 'medium' | 'high' | 'critical'
      anomaly_score: number
      detection_method: string
      affected_entities: {
        user_ids?: string[]
        session_ids?: string[]
        event_types?: string[]
      }
      description: string
      potential_causes: string[]
      recommended_actions: string[]
      false_positive_probability: number
    }>
    trend_analysis: {
      anomaly_frequency_trend: 'increasing' | 'stable' | 'decreasing'
      peak_anomaly_hours: string[]
      seasonal_patterns: Record<string, number>
    }
  }> {
    const response = await apiClient.post(
      `${this.baseUrl}/anomalies/advanced-detect`,
      params
    )
    return response.data
  }

  async getAnomalyInvestigationReport(anomalyId: string): Promise<{
    anomaly_id: string
    investigation_report: {
      root_cause_analysis: {
        primary_cause: string
        contributing_factors: string[]
        confidence_level: number
      }
      impact_assessment: {
        affected_user_count: number
        revenue_impact_estimate: number
        system_performance_impact: string
        duration_minutes: number
      }
      timeline: Array<{
        timestamp: string
        event_description: string
        severity_change?: string
      }>
      resolution_status: 'investigating' | 'resolved' | 'false_positive'
      lessons_learned?: string[]
      prevention_recommendations?: string[]
    }
  }> {
    const response = await apiClient.get(
      `${this.baseUrl}/anomalies/${anomalyId}/investigate`
    )
    return response.data
  }

  // ========== 系统配置和管理 ==========
  async getAdvancedConfiguration(): Promise<{
    processing_settings: {
      batch_processing_enabled: boolean
      realtime_processing_enabled: boolean
      processing_queue_size: number
      max_concurrent_jobs: number
    }
    analysis_settings: {
      anomaly_detection_sensitivity: string
      pattern_mining_min_support: number
      retention_analysis_enabled: boolean
      predictive_models_enabled: boolean
    }
    performance_settings: {
      cache_duration_hours: number
      max_query_result_size: number
      query_timeout_seconds: number
    }
    notification_settings: {
      anomaly_alerts_enabled: boolean
      report_completion_notifications: boolean
      webhook_urls: string[]
    }
  }> {
    const response = await apiClient.get(`${this.baseUrl}/config/advanced`)
    return response.data
  }

  async updateAdvancedConfiguration(config: Record<string, any>): Promise<{
    success: boolean
    updated_settings: string[]
    validation_errors?: string[]
    restart_required?: boolean
  }> {
    const response = await apiClient.put(
      `${this.baseUrl}/config/advanced`,
      config
    )
    return response.data
  }

  async getSystemHealthDashboard(): Promise<{
    overall_health: 'healthy' | 'degraded' | 'critical'
    component_status: {
      event_ingestion: 'operational' | 'degraded' | 'down'
      pattern_analysis: 'operational' | 'degraded' | 'down'
      anomaly_detection: 'operational' | 'degraded' | 'down'
      reporting_engine: 'operational' | 'degraded' | 'down'
      websocket_service: 'operational' | 'degraded' | 'down'
    }
    performance_metrics: {
      events_processed_last_hour: number
      avg_processing_latency_ms: number
      memory_usage_percentage: number
      cpu_usage_percentage: number
      disk_usage_percentage: number
    }
    active_alerts: Array<{
      alert_id: string
      severity: 'info' | 'warning' | 'error' | 'critical'
      message: string
      created_at: string
      component: string
    }>
    system_statistics: {
      uptime_hours: number
      total_events_processed: number
      total_reports_generated: number
      total_anomalies_detected: number
    }
  }> {
    const response = await apiClient.get(
      `${this.baseUrl}/system/health-dashboard`
    )
    return response.data
  }

  // ========== 用户体验分析 ==========
  async performUserExperienceAnalysis(params: {
    analysis_type:
      | 'journey_mapping'
      | 'friction_analysis'
      | 'satisfaction_scoring'
    time_period_days: number
    user_segments?: string[]
    include_heatmaps?: boolean
  }): Promise<{
    analysis_type: string
    ux_insights: {
      critical_friction_points: Array<{
        page_or_feature: string
        friction_score: number
        affected_users_percentage: number
        avg_abandonment_rate: number
        improvement_suggestions: string[]
      }>
      user_journey_flow: Array<{
        step: number
        page_or_action: string
        completion_rate: number
        avg_time_spent_seconds: number
        drop_off_rate: number
      }>
      satisfaction_metrics: {
        overall_satisfaction_score: number
        nps_score?: number
        feature_satisfaction_scores: Record<string, number>
      }
    }
    heatmap_data?: Array<{
      page_url: string
      interaction_heatmap: Record<string, number>
      scroll_depth_stats: Record<string, number>
    }>
    recommendations: Array<{
      priority: 'high' | 'medium' | 'low'
      area: string
      issue_description: string
      suggested_solution: string
      estimated_impact: string
    }>
  }> {
    const response = await apiClient.post(`${this.baseUrl}/ux/analyze`, params)
    return response.data
  }
}

export const analyticsServiceEnhanced = new AnalyticsServiceEnhanced()
