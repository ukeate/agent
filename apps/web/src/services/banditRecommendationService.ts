/**
 * 多臂老虎机推荐系统API服务
 */

import apiClient from './apiClient';

export interface RecommendationRequest {
  user_id: string;
  context?: Record<string, any>;
  num_recommendations?: number;
  exclude_items?: string[];
  include_explanations?: boolean;
  experiment_id?: string;
}

export interface Recommendation {
  item_id: string;
  score: number;
  confidence: number;
}

export interface RecommendationResponse {
  request_id: string;
  user_id: string;
  recommendations: Recommendation[];
  algorithm_used: string;
  confidence_score: number;
  cold_start_strategy?: string;
  explanations?: string[];
  timestamp?: string;
  processing_time_ms: number;
}

export interface FeedbackRequest {
  user_id: string;
  item_id: string;
  feedback_type: string;
  feedback_value?: number;
  context?: Record<string, any>;
}

export interface InitializeConfig {
  enable_cold_start?: boolean;
  enable_evaluation?: boolean;
}

export interface EngineStatistics {
  engine_stats: {
    total_requests: number;
    cache_hits: number;
    cold_start_requests: number;
    algorithm_usage: Record<string, number>;
    average_response_time_ms: number;
  };
  algorithm_stats: Record<string, {
    average_reward: number;
    total_reward: number;
    total_pulls: number;
    regret: number;
  }>;
  active_users: number;
  cache_size: number;
}

export interface AlgorithmInfo {
  name: string;
  display_name: string;
  description: string;
  supports_context: boolean;
  supports_binary_feedback: boolean;
}

export interface ExperimentConfig {
  experiment_name: string;
  algorithms: Record<string, {
    algorithm_type: string;
    config: Record<string, any>;
  }>;
  traffic_split: Record<string, number>;
  duration_hours: number;
  min_sample_size?: number;
}

export interface ExperimentResult {
  experiment_id: string;
  experiment_name: string;
  status: string;
  start_time: string;
  end_time: string;
  traffic_split: Record<string, number>;
  results: Record<string, {
    metrics: {
      click_through_rate: number;
      conversion_rate: number;
      average_reward: number;
      cumulative_reward: number;
      coverage: number;
      diversity: number;
    };
    sample_size: number;
    bandit_stats: Record<string, any>;
  }>;
  significance: {
    significant: boolean;
    p_value: number;
    t_statistic?: number;
    compared_variants?: string[];
  };
  recommendation: string;
}

class BanditRecommendationService {
  private readonly baseUrl = '/bandit';

  /**
   * 初始化推荐引擎
   */
  async initialize(
    nItems: number,
    config: InitializeConfig = {}
  ): Promise<{ status: string; message: string; config: any }> {
    const params = new URLSearchParams({
      n_items: nItems.toString(),
      enable_cold_start: (config.enable_cold_start ?? true).toString(),
      enable_evaluation: (config.enable_evaluation ?? true).toString(),
    });

    const response = await apiClient.post(`${this.baseUrl}/initialize?${params}`);
    return response.data;
  }

  /**
   * 获取推荐
   */
  async getRecommendations(request: RecommendationRequest): Promise<RecommendationResponse> {
    const response = await apiClient.post(`${this.baseUrl}/recommend`, request);
    return response.data;
  }

  /**
   * 提交用户反馈
   */
  async submitFeedback(feedback: FeedbackRequest): Promise<{ status: string; message: string }> {
    const response = await apiClient.post(`${this.baseUrl}/feedback`, feedback);
    return response.data;
  }

  /**
   * 获取引擎统计信息
   */
  async getStatistics(): Promise<{ status: string; statistics: EngineStatistics }> {
    const response = await apiClient.get(`${this.baseUrl}/statistics`);
    return response.data;
  }

  /**
   * 更新用户上下文
   */
  async updateUserContext(
    userId: string,
    context: Record<string, any>
  ): Promise<{ status: string; message: string }> {
    const response = await apiClient.put(`${this.baseUrl}/user/${userId}/context`, context);
    return response.data;
  }

  /**
   * 更新物品特征
   */
  async updateItemFeatures(
    itemId: string,
    features: Record<string, any>
  ): Promise<{ status: string; message: string }> {
    const response = await apiClient.put(`${this.baseUrl}/item/${itemId}/features`, features);
    return response.data;
  }

  /**
   * 获取健康状态
   */
  async getHealth(): Promise<{
    status: string;
    is_initialized: boolean;
    timestamp: string;
    engine_stats?: any;
  }> {
    const response = await apiClient.get(`${this.baseUrl}/health`);
    return response.data;
  }

  /**
   * 获取可用算法列表
   */
  async getAlgorithms(): Promise<{ status: string; algorithms: AlgorithmInfo[]; count: number }> {
    const response = await apiClient.get(`${this.baseUrl}/algorithms`);
    return response.data;
  }

  /**
   * 创建A/B测试实验
   */
  async createExperiment(experiment: ExperimentConfig): Promise<{
    status: string;
    experiment_id: string;
    experiment_name: string;
    start_time: string;
    end_time: string;
    variants: string[];
    message: string;
  }> {
    const response = await apiClient.post(`${this.baseUrl}/experiments`, experiment);
    return response.data;
  }

  /**
   * 获取A/B测试实验列表
   */
  async getExperiments(): Promise<{
    status: string;
    experiments: Array<{
      experiment_id: string;
      experiment_name: string;
      start_time: string;
      end_time: string;
      variants: string[];
      total_events: number;
    }>;
    count: number;
  }> {
    const response = await apiClient.get(`${this.baseUrl}/experiments`);
    return response.data;
  }

  /**
   * 获取A/B测试实验结果
   */
  async getExperimentResults(experimentId: string): Promise<{
    status: string;
    results: ExperimentResult;
  }> {
    const response = await apiClient.get(`${this.baseUrl}/experiments/${experimentId}/results`);
    return response.data;
  }

  /**
   * 结束A/B测试实验
   */
  async endExperiment(experimentId: string): Promise<{
    status: string;
    message: string;
    timestamp: string;
  }> {
    const response = await apiClient.post(`${this.baseUrl}/experiments/${experimentId}/end`);
    return response.data;
  }

  /**
   * 批量获取推荐（用于性能测试）
   */
  async batchRecommendations(
    requests: RecommendationRequest[]
  ): Promise<RecommendationResponse[]> {
    const promises = requests.map(request => this.getRecommendations(request));
    return Promise.all(promises);
  }

  /**
   * 多次请求推荐，收集真实返回
   */
  async simulateUserSession(
    userId: string,
    numInteractions: number,
    contextGenerator?: () => Record<string, any>
  ): Promise<{
    recommendations: RecommendationResponse[];
    feedbacks: Array<{ item_id: string; feedback_type: string; feedback_value: number }>;
    totalReward: number;
  }> {
    const recommendations: RecommendationResponse[] = [];
    const feedbacks: Array<{ item_id: string; feedback_type: string; feedback_value: number }> = [];
    let totalReward = 0;

    for (let i = 0; i < numInteractions; i++) {
      const context = contextGenerator ? contextGenerator() : undefined;
      const recResponse = await this.getRecommendations({
        user_id: userId,
        num_recommendations: 3,
        context,
        include_explanations: false
      });
      recommendations.push(recResponse);
    }

    return { recommendations, feedbacks, totalReward };
  }

  /**
   * 获取算法性能对比数据
   */
  async getAlgorithmComparison(): Promise<{
    algorithms: string[];
    metrics: {
      algorithm: string;
      average_reward: number;
      total_pulls: number;
      regret: number;
      usage_count: number;
    }[];
  }> {
    const statsResponse = await this.getStatistics();
    const stats = statsResponse.statistics;
    
    const algorithms = Object.keys(stats.algorithm_stats);
    const metrics = algorithms.map(algorithm => ({
      algorithm,
      average_reward: stats.algorithm_stats[algorithm].average_reward || 0,
      total_pulls: stats.algorithm_stats[algorithm].total_pulls || 0,
      regret: stats.algorithm_stats[algorithm].regret || 0,
      usage_count: stats.engine_stats.algorithm_usage[algorithm] || 0
    }));

    return { algorithms, metrics };
  }
}

// 导出单例实例
export const banditRecommendationService = new BanditRecommendationService();
export default banditRecommendationService;
