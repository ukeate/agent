import apiClient from './apiClient';

export interface EmotionalDecision {
  decision_id: string;
  user_id: string;
  chosen_strategy: string;
  confidence_score: number;
  reasoning: string[];
  timestamp: string;
  decision_type: string;
  response?: string;
  execution_plan?: string[];
}

export interface RiskAssessment {
  assessment_id: string;
  user_id: string;
  risk_level: 'low' | 'medium' | 'high' | 'critical';
  risk_score: number;
  prediction_confidence: number;
  recommended_actions: string[];
  triggers?: string[];
  mitigation_strategies?: string[];
}

export interface InterventionPlan {
  plan_id: string;
  user_id: string;
  strategy_type: string;
  interventions: string[];
  priority: 'low' | 'medium' | 'high' | 'urgent';
  expected_effectiveness: number;
  implementation_timeline: string;
}

export interface CrisisAssessment {
  assessment_id: string;
  user_id: string;
  is_crisis: boolean;
  crisis_level: 'none' | 'mild' | 'moderate' | 'severe' | 'critical';
  confidence: number;
  immediate_actions: string[];
  professional_help_needed: boolean;
  support_resources: string[];
}

export interface HealthDashboard {
  user_id: string;
  period: string;
  overall_health_score: number;
  emotion_stability: number;
  stress_level: number;
  social_engagement: number;
  trend: 'improving' | 'stable' | 'declining';
  key_insights: string[];
  recommendations: string[];
}

export interface EmotionState {
  dominant_emotion: string;
  emotion_scores: Record<string, number>;
  valence: number;
  arousal: number;
  confidence: number;
}

export interface DecisionRequest {
  user_id: string;
  session_id?: string;
  user_input: string;
  current_emotion_state: EmotionState | Record<string, any>;
  emotion_history?: Array<EmotionState | Record<string, any>>;
  personality_profile?: Record<string, any>;
  environmental_factors?: Record<string, any>;
  previous_decisions?: Array<Record<string, any>>;
}

export interface RiskRequest {
  user_id: string;
  emotion_history: Array<EmotionState | Record<string, any>>;
  personality_profile?: Record<string, any>;
  context?: Record<string, any>;
}

export interface CrisisRequest {
  user_id: string;
  user_input: string;
  emotion_state: EmotionState | Record<string, any>;
  context?: Record<string, any>;
  emotion_history?: Array<EmotionState | Record<string, any>>;
}

export interface InterventionRequest {
  user_id: string;
  risk_assessment: RiskAssessment | Record<string, any>;
  user_preferences?: Record<string, any>;
  past_effectiveness?: Record<string, number>;
}

export interface HealthRequest {
  user_id: string;
  time_period_days?: number;
}

export interface SystemStats {
  total_decisions: number;
  average_confidence: number;
  high_risk_users: number;
  active_interventions: number;
  crisis_detections_24h: number;
  successful_interventions: number;
  system_uptime: number;
  last_update: string;
}

class EmotionalIntelligenceService {
  private baseUrl = '/emotional-intelligence';

  // 决策功能
  async makeDecision(request: DecisionRequest): Promise<EmotionalDecision> {
    const response = await apiClient.post(`${this.baseUrl}/decide`, request);
    return response.data.decision || response.data;
  }

  async getDecisionHistory(userId: string, limit?: number): Promise<EmotionalDecision[]> {
    const params: Record<string, any> = {};
    if (limit) params.limit = limit;
    if (userId && userId !== 'all') params.user_id = userId;
    const response = await apiClient.get(`${this.baseUrl}/decisions/history`, { params });
    return response.data.decisions || [];
  }

  async getDecisionById(decisionId: string): Promise<EmotionalDecision> {
    const decisions = await this.getDecisionHistory('all', 200);
    const decision = decisions.find(item => item.decision_id === decisionId);
    if (!decision) {
      throw new Error('未找到指定的决策记录');
    }
    return decision;
  }

  // 风险评估
  async assessRisk(request: RiskRequest): Promise<RiskAssessment> {
    const response = await apiClient.post(`${this.baseUrl}/risk-assessment`, request);
    return response.data.risk_assessment || response.data;
  }

  async getRiskHistory(userId: string, days?: number): Promise<RiskAssessment[]> {
    const assessment = await this.assessRisk({
      user_id: userId,
      emotion_history: [],
      context: days ? { days } : undefined
    });
    return [assessment];
  }

  async getHighRiskUsers(threshold?: number): Promise<Array<{ user_id: string; risk_score: number }>> {
    const params = threshold ? { threshold } : {};
    const response = await apiClient.get(`${this.baseUrl}/high-risk-users`, { params });
    return response.data.users || [];
  }

  // 危机检测
  async detectCrisis(request: CrisisRequest): Promise<CrisisAssessment> {
    const response = await apiClient.post(`${this.baseUrl}/crisis-detection`, request);
    return response.data.crisis_assessment || response.data;
  }

  async getCrisisAlerts(active?: boolean): Promise<CrisisAssessment[]> {
    const params = active !== undefined ? { active } : {};
    const response = await apiClient.get(`${this.baseUrl}/crisis-alerts`, { params });
    return response.data.alerts || [];
  }

  async acknowledgeCrisis(assessmentId: string, action: string): Promise<void> {
    await apiClient.post(`${this.baseUrl}/crisis/${assessmentId}/acknowledge`, { action });
  }

  // 干预计划
  async createInterventionPlan(request: InterventionRequest): Promise<InterventionPlan> {
    const response = await apiClient.post(`${this.baseUrl}/intervention-plan`, request);
    return response.data.intervention_plan || response.data;
  }

  async getInterventionPlans(userId: string, active?: boolean): Promise<InterventionPlan[]> {
    const params = active !== undefined ? { active } : {};
    const response = await apiClient.get(`${this.baseUrl}/interventions/${userId}`, { params });
    return response.data.plans || [];
  }

  async updateInterventionEffectiveness(planId: string, effectiveness: number): Promise<void> {
    await apiClient.put(`${this.baseUrl}/intervention/${planId}/effectiveness`, { effectiveness });
  }

  // 健康监控
  async getHealthDashboard(request: HealthRequest): Promise<HealthDashboard> {
    const response = await apiClient.get(`${this.baseUrl}/health-dashboard/${request.user_id}`, {
      params: { time_period_days: request.time_period_days }
    });
    return response.data.dashboard || response.data;
  }

  async getHealthTrends(userId: string, metric: string, days?: number): Promise<{
    dates: string[];
    values: number[];
    trend: string;
  }> {
    const response = await apiClient.get(`${this.baseUrl}/health-dashboard/${userId}`, {
      params: { time_period_days: days || 30 }
    });
    const trendData = response.data.dashboard?.emotion_trends?.[metric] || [];
    return {
      dates: trendData.map((item: [string, number]) => item[0]),
      values: trendData.map((item: [string, number]) => item[1]),
      trend: response.data.dashboard?.risk_trend || 'stable'
    };
  }

  async getHealthAlerts(userId: string): Promise<Array<{
    alert_type: string;
    severity: string;
    message: string;
    timestamp: string;
  }>> {
    const response = await apiClient.get(`${this.baseUrl}/risk-trends/${userId}`);
    const trend = response.data.trends?.trend || 'stable';
    return [{
      alert_type: 'risk_trend',
      severity: trend,
      message: `当前风险趋势: ${trend}`,
      timestamp: new Date().toISOString()
    }];
  }

  // 系统统计
  async getSystemStats(): Promise<SystemStats> {
    const response = await apiClient.get(`${this.baseUrl}/stats`);
    return response.data;
  }

  async getDecisionMetrics(period?: string): Promise<{
    total_decisions: number;
    by_type: Record<string, number>;
    average_confidence: number;
    success_rate: number;
  }> {
    const response = await apiClient.get(`${this.baseUrl}/decisions/history`, {
      params: { limit: 500 }
    });
    const decisions = response.data.decisions || [];
    const byType: Record<string, number> = {};
    decisions.forEach((item: any) => {
      byType[item.decision_type] = (byType[item.decision_type] || 0) + 1;
    });
    const averageConfidence = decisions.length
      ? decisions.reduce((sum: number, item: any) => sum + item.confidence_score, 0) / decisions.length
      : 0;
    return {
      total_decisions: decisions.length,
      by_type: byType,
      average_confidence: averageConfidence,
      success_rate: decisions.length ? decisions.filter((item: any) => item.effectiveness_score).length / decisions.length : 0
    };
  }

  async getRiskMetrics(): Promise<{
    current_high_risk: number;
    risk_distribution: Record<string, number>;
    average_risk_score: number;
    trend: string;
  }> {
    const response = await apiClient.get(`${this.baseUrl}/high-risk-users`, {
      params: { threshold: 0.7 }
    });
    const users = response.data.users || [];
    return {
      current_high_risk: users.length,
      risk_distribution: users.reduce((acc: Record<string, number>, item: any) => {
        const level = item.risk_level || 'unknown';
        acc[level] = (acc[level] || 0) + 1;
        return acc;
      }, {}),
      average_risk_score: users.length
        ? users.reduce((sum: number, item: any) => sum + item.risk_score, 0) / users.length
        : 0,
      trend: 'stable'
    };
  }

  // 批量操作
  async batchAssessRisk(userIds: string[]): Promise<Record<string, RiskAssessment>> {
    const results: Record<string, RiskAssessment> = {};
    for (const userId of userIds) {
      results[userId] = await this.assessRisk({ user_id: userId, emotion_history: [] });
    }
    return results;
  }

  async batchHealthCheck(userIds: string[]): Promise<Record<string, HealthDashboard>> {
    const results: Record<string, HealthDashboard> = {};
    for (const userId of userIds) {
      const dashboard = await this.getHealthDashboard({ user_id: userId });
      results[userId] = dashboard;
    }
    return results;
  }

  // 配置管理
  async getConfiguration(): Promise<Record<string, any>> {
    const response = await apiClient.get(`${this.baseUrl}/config`);
    return response.data.config || response.data;
  }

  async updateConfiguration(config: Record<string, any>): Promise<void> {
    await apiClient.put(`${this.baseUrl}/config`, config);
  }

  // 新增：未使用的API方法 - 根据api-ui.md
  
  /**
   * 情感智能决策 - POST /emotional-intelligence/decide
   */
  async makeEmotionalDecision(request: DecisionRequest): Promise<EmotionalDecision> {
    const response = await apiClient.post(`/emotional-intelligence/decide`, request);
    return response.data.decision || response.data;
  }

  /**
   * 风险评估 - POST /emotional-intelligence/risk-assessment
   */
  async performRiskAssessment(request: RiskRequest): Promise<RiskAssessment> {
    const response = await apiClient.post(`/emotional-intelligence/risk-assessment`, request);
    return response.data.risk_assessment || response.data;
  }

  /**
   * 危机检测 - POST /emotional-intelligence/crisis-detection
   */
  async performCrisisDetection(request: CrisisRequest): Promise<CrisisAssessment> {
    const response = await apiClient.post(`/emotional-intelligence/crisis-detection`, request);
    return response.data.crisis_assessment || response.data;
  }

  /**
   * 生成干预计划 - POST /emotional-intelligence/intervention-plan
   */
  async generateInterventionPlan(request: InterventionRequest): Promise<InterventionPlan> {
    const response = await apiClient.post(`/emotional-intelligence/intervention-plan`, request);
    return response.data.intervention_plan || response.data;
  }

  /**
   * 获取情感模式 - GET /emotional-intelligence/emotional-patterns/{user_id}
   */
  async getEmotionalPatterns(userId: string): Promise<any> {
    const response = await apiClient.get(`/emotional-intelligence/emotional-patterns/${userId}`);
    return response.data;
  }

  /**
   * 自杀风险评估 - POST /emotional-intelligence/suicide-risk-assessment
   */
  async assessSuicideRisk(request: any): Promise<any> {
    const response = await apiClient.post(`/emotional-intelligence/suicide-risk-assessment`, request);
    return response.data;
  }

  /**
   * 获取风险趋势 - GET /emotional-intelligence/risk-trends/{user_id}
   */
  async getRiskTrends(userId: string): Promise<any> {
    const response = await apiClient.get(`/emotional-intelligence/risk-trends/${userId}`);
    return response.data;
  }

  /**
   * 获取危机预测 - GET /emotional-intelligence/crisis-prediction/{user_id}
   */
  async getCrisisPrediction(userId: string): Promise<any> {
    const response = await apiClient.get(`/emotional-intelligence/crisis-prediction/${userId}`);
    return response.data;
  }

  /**
   * 评估干预有效性 - POST /emotional-intelligence/intervention-effectiveness
   */
  async assessInterventionEffectiveness(request: any): Promise<any> {
    const response = await apiClient.post(`/emotional-intelligence/intervention-effectiveness`, request);
    return response.data;
  }

  /**
   * 获取系统状态 - GET /emotional-intelligence/system-status
   */
  async getEmotionalIntelligenceSystemStatus(): Promise<any> {
    const response = await apiClient.get(`/emotional-intelligence/system-status`);
    return response.data;
  }

  // 情感状态管理 - emotion API

  /**
   * 记录情感状态 - POST /emotion/state
   */
  async recordEmotionState(emotionState: any): Promise<any> {
    const response = await apiClient.post(`/emotion/state`, emotionState);
    return response.data;
  }

  /**
   * 获取最新情感状态 - GET /emotion/state/latest
   */
  async getLatestEmotionState(userId?: string): Promise<any> {
    const params = userId ? { user_id: userId } : {};
    const response = await apiClient.get(`/emotion/state/latest`, { params });
    return response.data;
  }

  /**
   * 获取情感状态历史 - GET /emotion/state/history
   */
  async getEmotionStateHistory(userId?: string, limit?: number): Promise<any> {
    const params: any = {};
    if (userId) params.user_id = userId;
    if (limit) params.limit = limit;
    const response = await apiClient.get(`/emotion/state/history`, { params });
    return response.data;
  }

  /**
   * 情感预测 - POST /emotion/predict
   */
  async predictEmotion(request: any): Promise<any> {
    const response = await apiClient.post(`/emotion/predict`, request);
    return response.data;
  }

  /**
   * 情感分析 - POST /emotion/analytics
   */
  async performEmotionAnalytics(request: any): Promise<any> {
    const response = await apiClient.post(`/emotion/analytics`, request);
    return response.data;
  }

  /**
   * 获取情感档案 - GET /emotion/profile
   */
  async getEmotionProfile(userId?: string): Promise<any> {
    const params = userId ? { user_id: userId } : {};
    const response = await apiClient.get(`/emotion/profile`, { params });
    return response.data;
  }

  /**
   * 获取情感聚类 - GET /emotion/clusters
   */
  async getEmotionClusters(): Promise<any> {
    const response = await apiClient.get(`/emotion/clusters`);
    return response.data;
  }

  /**
   * 获取情感转换 - GET /emotion/transitions
   */
  async getEmotionTransitions(userId?: string): Promise<any> {
    const params = userId ? { user_id: userId } : {};
    const response = await apiClient.get(`/emotion/transitions`, { params });
    return response.data;
  }

  /**
   * 导出情感数据 - GET /emotion/export
   */
  async exportEmotionData(format?: string): Promise<Blob> {
    const params = format ? { format } : {};
    const response = await apiClient.get(`/emotion/export`, { 
      params,
      responseType: 'blob'
    });
    return response.data;
  }

  /**
   * 删除情感数据 - DELETE /emotion/data
   */
  async deleteEmotionData(userId?: string): Promise<void> {
    const params = userId ? { user_id: userId } : {};
    await apiClient.delete(`/emotion/data`, { params });
  }

  /**
   * 获取情感统计 - GET /emotion/statistics
   */
  async getEmotionStatistics(): Promise<any> {
    const response = await apiClient.get(`/emotion/statistics`);
    return response.data;
  }

  // 社交情感理解系统 API

  /**
   * 综合分析 - POST /social-emotional-understanding/comprehensive-analysis
   */
  async performComprehensiveAnalysis(request: any): Promise<any> {
    const response = await apiClient.post(`/social-emotional-understanding/comprehensive-analysis`, request);
    return response.data;
  }

  /**
   * 获取社交情感分析数据 - GET /social-emotional-understanding/analytics
   */
  async getSocialEmotionalAnalytics(): Promise<any> {
    const response = await apiClient.get(`/social-emotional-understanding/analytics`);
    return response.data;
  }

  // 社交情感API

  /**
   * 初始化系统 - POST /social-emotion/initialize
   */
  async initializeSocialEmotionSystem(config: any): Promise<any> {
    const response = await apiClient.post(`/social-emotion/initialize`, config);
    return response.data;
  }

  /**
   * 创建会话 - POST /social-emotion/session/create
   */
  async createSocialEmotionSession(sessionData: any): Promise<any> {
    const response = await apiClient.post(`/social-emotion/session/create`, sessionData);
    return response.data;
  }

  /**
   * 删除会话 - DELETE /social-emotion/session/{session_id}
   */
  async deleteSocialEmotionSession(sessionId: string): Promise<void> {
    await apiClient.delete(`/social-emotion/session/${sessionId}`);
  }

  /**
   * 隐私同意 - POST /social-emotion/privacy/consent
   */
  async submitPrivacyConsent(consentData: any): Promise<any> {
    const response = await apiClient.post(`/social-emotion/privacy/consent`, consentData);
    return response.data;
  }

  /**
   * 删除隐私同意 - DELETE /social-emotion/privacy/consent/{user_id}
   */
  async deletePrivacyConsent(userId: string): Promise<void> {
    await apiClient.delete(`/social-emotion/privacy/consent/${userId}`);
  }

  /**
   * 更新隐私政策 - POST /social-emotion/privacy/policy
   */
  async updatePrivacyPolicy(policyData: any): Promise<any> {
    const response = await apiClient.post(`/social-emotion/privacy/policy`, policyData);
    return response.data;
  }

  /**
   * 获取社交情感仪表板 - GET /social-emotion/dashboard
   */
  async getSocialEmotionDashboard(): Promise<any> {
    const response = await apiClient.get(`/social-emotion/dashboard`);
    return response.data;
  }

  /**
   * 导出社交情感数据 - POST /social-emotion/export
   */
  async exportSocialEmotionData(exportRequest: any): Promise<any> {
    const response = await apiClient.post(`/social-emotion/export`, exportRequest);
    return response.data;
  }

  /**
   * 获取情感识别模型 - GET /emotion/models
   */
  async getEmotionModels(): Promise<any> {
    const response = await apiClient.get(`/emotion/models`);
    return response.data;
  }

  /**
   * 广播消息 - POST /ws/broadcast
   */
  async broadcastEmotionMessage(message: any): Promise<any> {
    const response = await apiClient.post('/ws/connections/broadcast', message);
    return response.data;
  }

  // 情感分析 - 调用未使用的API
  async analyzeEmotion(text: string, context?: Record<string, any>): Promise<{
    emotion: string;
    confidence: number;
    emotions: Record<string, number>;
    sentiment: {
      score: number;
      label: string;
    };
    keywords: string[];
  }> {
    const response = await apiClient.post('/emotion-intelligence/analyze', { text, context });
    return response.data;
  }
}

export const emotionalIntelligenceService = new EmotionalIntelligenceService();
