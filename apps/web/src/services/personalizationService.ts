import apiClient from './apiClient';

// ==================== 类型定义 ====================

export interface UserProfile {
  user_id: string;
  preferences: UserPreferences;
  behavior_patterns: BehaviorPattern[];
  recommendations: Recommendation[];
  segments: string[];
  created_at: string;
  updated_at: string;
}

export interface UserPreferences {
  theme?: 'light' | 'dark' | 'auto';
  language?: string;
  timezone?: string;
  notification_settings?: {
    email: boolean;
    push: boolean;
    sms: boolean;
    frequency: 'immediate' | 'hourly' | 'daily' | 'weekly';
  };
  content_preferences?: {
    categories: string[];
    topics: string[];
    formats: string[];
  };
  interaction_style?: {
    complexity_level: 'beginner' | 'intermediate' | 'advanced';
    response_length: 'brief' | 'moderate' | 'detailed';
    tone: 'formal' | 'casual' | 'professional';
  };
  privacy_settings?: {
    data_collection: boolean;
    personalization: boolean;
    analytics: boolean;
  };
}

export interface BehaviorPattern {
  pattern_id: string;
  pattern_type: string;
  description: string;
  frequency: number;
  confidence: number;
  last_observed: string;
  metadata?: Record<string, any>;
}

export interface Recommendation {
  recommendation_id: string;
  type: 'content' | 'feature' | 'action' | 'setting';
  title: string;
  description: string;
  reason: string;
  score: number;
  metadata?: {
    category?: string;
    tags?: string[];
    source?: string;
  };
  action?: {
    type: string;
    url?: string;
    params?: Record<string, any>;
  };
}

export interface PersonalizationModel {
  model_id: string;
  model_type: string;
  version: string;
  accuracy: number;
  last_trained: string;
  features_used: string[];
  performance_metrics: Record<string, number>;
}

export interface PersonalizationEvent {
  event_id: string;
  user_id: string;
  event_type: string;
  timestamp: string;
  context?: Record<string, any>;
  outcome?: string;
}

export interface ABTestVariant {
  variant_id: string;
  name: string;
  description: string;
  percentage: number;
  is_control: boolean;
  config: Record<string, any>;
}

export interface PersonalizationExperiment {
  experiment_id: string;
  name: string;
  description: string;
  status: 'draft' | 'running' | 'paused' | 'completed';
  variants: ABTestVariant[];
  metrics: string[];
  start_date?: string;
  end_date?: string;
  results?: {
    winner?: string;
    confidence?: number;
    metrics: Record<string, any>;
  };
}

// ==================== Service Class ====================

class PersonalizationService {
  private baseUrl = '/personalization';
  private lastFeatureSnapshot: { version?: string; timestamp?: string } | null = null;

  getClientId(): string {
    if (typeof document === 'undefined') return '';
    const match = document.cookie.match(/(?:^|; )client_id=([^;]+)/);
    if (match) return decodeURIComponent(match[1]);
    const clientId = crypto.randomUUID();
    document.cookie = `client_id=${encodeURIComponent(clientId)}; path=/; samesite=lax`;
    return clientId;
  }

  private normalizeNumber(value: any): number {
    return typeof value === 'number' && !Number.isNaN(value) ? value : 0;
  }

  private buildFeatureItems(features: any): Array<{
    id: string;
    name: string;
    value: number;
    type: string;
    updated_at: string;
  }> {
    const items: Array<{ id: string; name: string; value: number; type: string; updated_at: string }> = [];
    const timestamp = features?.timestamp || '';
    const pushGroup = (group: string, values: any) => {
      if (!values || typeof values !== 'object') return;
      Object.entries(values).forEach(([key, value]) => {
        items.push({
          id: `${group}:${key}`,
          name: `${group}.${key}`,
          value: this.normalizeNumber(value),
          type: group,
          updated_at: timestamp,
        });
      });
    };
    pushGroup('temporal', features?.temporal);
    pushGroup('behavioral', features?.behavioral);
    pushGroup('contextual', features?.contextual);
    pushGroup('aggregated', features?.aggregated);
    return items;
  }

  private buildPatternItems(features: any): Array<any> {
    const items = this.buildFeatureItems(features);
    return items.map((item) => ({
      id: item.id,
      name: item.name,
      category: item.type === 'behavioral' ? 'interaction' : item.type === 'contextual' ? 'context' : item.type === 'aggregated' ? 'item' : 'user',
      type: 'realtime',
      value: item.value,
      importance: Math.min(1, Math.max(0, item.value)),
      frequency: '实时',
      status: 'active',
    }));
  }

  // ==================== 用户配置文件管理 ====================

  async getUserProfile(userId?: string): Promise<UserProfile> {
    const uid = userId || this.getClientId();
    const url = `${this.baseUrl}/user/${uid}/profile`;
    const response = await apiClient.get(url);
    return response.data;
  }

  async updatePreferences(preferences: Partial<UserPreferences>): Promise<UserPreferences> {
    const response = await apiClient.put(`${this.baseUrl}/preferences`, preferences);
    return response.data;
  }

  async getPreferences(): Promise<UserPreferences> {
    const response = await apiClient.get(`${this.baseUrl}/preferences`);
    return response.data;
  }

  async resetPreferences(): Promise<{ success: boolean; message: string }> {
    const response = await apiClient.post(`${this.baseUrl}/preferences/reset`);
    return response.data;
  }

  // ==================== 推荐系统 ====================

  async getRecommendations(params?: {
    type?: string;
    limit?: number;
    min_score?: number;
    scenario?: string;
    user_id?: string;
    context?: Record<string, any>;
  }): Promise<Array<{ id: string; item: string; score: number; reason: string }>> {
    const userId = params?.user_id || this.getClientId();
    const response = await apiClient.post(`${this.baseUrl}/recommend`, {
      user_id: userId,
      n_recommendations: params?.limit ?? 10,
      scenario: params?.scenario || 'content',
      context: params?.context || {},
      use_cache: true,
    });
    const recommendations = response.data?.recommendations || [];
    return recommendations.map((item: any) => ({
      id: item.item_id,
      item: item.metadata?.title || item.item_id,
      score: this.normalizeNumber(item.score),
      reason: item.explanation || item.metadata?.reason || '',
    }));
  }

  async getSystemOverview(): Promise<{
    latency_p99: number;
    throughput: number;
    cache_hit_rate: number;
    active_users: number;
    total_recommendations: number;
    error_rate: number;
  }> {
    const [metricsResponse, cacheResponse] = await Promise.all([
      apiClient.get(`${this.baseUrl}/metrics`),
      apiClient.get(`${this.baseUrl}/cache/stats`),
    ]);
    const metrics = metricsResponse.data || {};
    const cacheStats = cacheResponse.data || {};
    const featureStats = cacheStats.feature_engine_stats || {};
    const cacheHitRate = this.normalizeNumber(metrics.cache_hit_rate);
    return {
      latency_p99: this.normalizeNumber(metrics.p99_latency_ms),
      throughput: this.normalizeNumber(metrics.total_requests),
      cache_hit_rate: cacheHitRate * 100,
      active_users: this.normalizeNumber(featureStats.total_users),
      total_recommendations: this.normalizeNumber(metrics.total_requests),
      error_rate: this.normalizeNumber(metrics.error_rate) * 100,
    };
  }

  async getFeatureStore(): Promise<Array<{ id: string; name: string; value: number; type: string; updated_at: string }>> {
    const uid = this.getClientId();
    const response = await apiClient.get(`${this.baseUrl}/features/realtime/${uid}`);
    return this.buildFeatureItems(response.data);
  }

  async getModels(): Promise<Array<{ name: string; version: string; status: 'online' | 'offline' | 'updating'; accuracy: number; latency: number }>> {
    const response = await apiClient.get(`${this.baseUrl}/models/status`);
    const status = response.data || {};
    return Object.entries(status).map(([modelId, info]: [string, any]) => {
      const distribution = info.status_distribution || {};
      const isReady = (distribution.ready || 0) + (distribution.serving || 0) > 0;
      const isLoading = (distribution.loading || 0) > 0;
      const state: 'online' | 'offline' | 'updating' = isReady ? 'online' : isLoading ? 'updating' : 'offline';
      const errorRate = this.normalizeNumber(info.error_rate);
      return {
        name: modelId,
        version: (info.versions || [])[0] || '',
        status: state,
        accuracy: Math.max(0, (1 - errorRate) * 100),
        latency: this.normalizeNumber(info.average_latency_ms),
      };
    });
  }

  async dismissRecommendation(recommendationId: string): Promise<{ success: boolean }> {
    const response = await apiClient.post(`${this.baseUrl}/recommendations/${recommendationId}/dismiss`);
    return response.data;
  }

  async acceptRecommendation(recommendationId: string): Promise<{ success: boolean }> {
    const response = await apiClient.post(`${this.baseUrl}/recommendations/${recommendationId}/accept`);
    return response.data;
  }

  async provideFeedback(recommendationId: string, feedback: {
    rating: number;
    comment?: string;
  }): Promise<{ success: boolean }> {
    const response = await apiClient.post(`${this.baseUrl}/recommendations/${recommendationId}/feedback`, feedback);
    return response.data;
  }

  // ==================== 行为分析 ====================

  async trackEvent(event: Omit<PersonalizationEvent, 'event_id' | 'timestamp'>): Promise<{ event_id: string }> {
    const response = await apiClient.post(`${this.baseUrl}/events`, event);
    return response.data;
  }

  async getBehaviorPatterns(): Promise<BehaviorPattern[]> {
    const uid = this.getClientId();
    const response = await apiClient.get(`${this.baseUrl}/features/realtime/${uid}`);
    this.lastFeatureSnapshot = {
      version: response.data?.version,
      timestamp: response.data?.timestamp,
    };
    return this.buildPatternItems(response.data) as BehaviorPattern[];
  }

  getFeatureSnapshotInfo(): { version?: string; timestamp?: string } | null {
    return this.lastFeatureSnapshot;
  }

  async getEventHistory(params?: {
    event_type?: string;
    start_date?: string;
    end_date?: string;
    limit?: number;
  }): Promise<PersonalizationEvent[]> {
    const response = await apiClient.get(`${this.baseUrl}/events`, { params });
    return response.data;
  }

  // ==================== 用户分群 ====================

  async getUserSegments(): Promise<Array<{
    segment_id: string;
    name: string;
    description: string;
    size: number;
    criteria: Record<string, any>;
  }>> {
    const response = await apiClient.get(`${this.baseUrl}/segments`);
    return response.data;
  }

  async getSegmentDetails(segmentId: string): Promise<{
    segment_id: string;
    name: string;
    description: string;
    users: number;
    characteristics: Record<string, any>;
    trends: Array<{ date: string; count: number }>;
  }> {
    const response = await apiClient.get(`${this.baseUrl}/segments/${segmentId}`);
    return response.data;
  }

  // ==================== A/B测试 ====================

  async getActiveExperiments(): Promise<PersonalizationExperiment[]> {
    const response = await apiClient.get(`${this.baseUrl}/experiments/active`);
    return response.data;
  }

  async getExperimentVariant(experimentId: string): Promise<ABTestVariant> {
    const response = await apiClient.get(`${this.baseUrl}/experiments/${experimentId}/variant`);
    return response.data;
  }

  async recordExperimentEvent(experimentId: string, event: {
    metric: string;
    value: number;
    metadata?: Record<string, any>;
  }): Promise<{ success: boolean }> {
    const response = await apiClient.post(`${this.baseUrl}/experiments/${experimentId}/event`, event);
    return response.data;
  }

  // ==================== 个性化模型 ====================

  async getModelInfo(): Promise<PersonalizationModel> {
    const response = await apiClient.get(`${this.baseUrl}/model`);
    return response.data;
  }

  async trainModel(params?: {
    features?: string[];
    algorithm?: string;
  }): Promise<{ job_id: string; status: string }> {
    const response = await apiClient.post(`${this.baseUrl}/model/train`, params || {});
    return response.data;
  }

  async getModelPrediction(context: Record<string, any>): Promise<{
    predictions: Array<{
      item: string;
      score: number;
      confidence: number;
    }>;
  }> {
    const response = await apiClient.post(`${this.baseUrl}/model/predict`, context);
    return response.data;
  }

  // ==================== 内容个性化 ====================

  async getPersonalizedContent(params?: {
    category?: string;
    limit?: number;
  }): Promise<Array<{
    content_id: string;
    title: string;
    description: string;
    type: string;
    relevance_score: number;
    metadata?: Record<string, any>;
  }>> {
    const response = await apiClient.get(`${this.baseUrl}/content`, { params });
    return response.data;
  }

  async getPersonalizedLayout(): Promise<{
    layout_id: string;
    components: Array<{
      component_id: string;
      type: string;
      position: number;
      config: Record<string, any>;
    }>;
  }> {
    const response = await apiClient.get(`${this.baseUrl}/layout`);
    return response.data;
  }

  // ==================== 导入/导出 ====================

  async exportPersonalizationData(format: 'json' | 'csv' = 'json'): Promise<Blob> {
    const response = await apiClient.get(`${this.baseUrl}/export`, {
      params: { format },
      responseType: 'blob'
    });
    return response.data;
  }

  async importPersonalizationData(file: File): Promise<{ success: boolean; imported: number }> {
    const formData = new FormData();
    formData.append('file', file);
    
    const response = await apiClient.post(`${this.baseUrl}/import`, formData, {
      headers: { 'Content-Type': 'multipart/form-data' }
    });
    return response.data;
  }

  // ==================== 隐私控制 ====================

  async deletePersonalizationData(): Promise<{ success: boolean; message: string }> {
    const response = await apiClient.delete(`${this.baseUrl}/data`);
    return response.data;
  }

  async getDataUsage(): Promise<{
    data_points: number;
    storage_bytes: number;
    last_updated: string;
    categories: Record<string, number>;
  }> {
    const response = await apiClient.get(`${this.baseUrl}/data/usage`);
    return response.data;
  }

  async optOut(): Promise<{ success: boolean }> {
    const response = await apiClient.post(`${this.baseUrl}/opt-out`);
    return response.data;
  }

  async optIn(): Promise<{ success: boolean }> {
    const response = await apiClient.post(`${this.baseUrl}/opt-in`);
    return response.data;
  }
}

// ==================== 导出 ====================

export const personalizationService = new PersonalizationService();
export default personalizationService;
