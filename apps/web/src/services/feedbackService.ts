/**
 * 反馈服务API客户端
 * 处理与反馈系统后端API的交互
 */

import apiClient from './apiClient'
import { FeedbackEvent, FeedbackType } from './feedbackTracker'

import { logger } from '../utils/logger'
// API响应类型
export interface ApiResponse<T = any> {
  success: boolean
  data?: T
  error?: string
  message?: string
}

// 反馈批次提交接口
export interface FeedbackBatch {
  batch_id: string
  user_id: string
  session_id: string
  events: FeedbackEvent[]
  timestamp: number
}

// 显式反馈提交接口
export interface ExplicitFeedback {
  feedback_id: string
  user_id: string
  session_id: string
  item_id?: string
  feedback_type: FeedbackType
  value: any
  context: Record<string, any>
  metadata?: Record<string, any>
  timestamp: number
}

// 用户反馈历史查询参数
export interface FeedbackHistoryQuery {
  user_id: string
  start_date?: string
  end_date?: string
  feedback_types?: FeedbackType[]
  item_id?: string
  limit?: number
  offset?: number
}

// 反馈统计数据
export interface FeedbackAnalytics {
  user_id: string
  total_feedbacks: number
  feedback_distribution: Record<FeedbackType, number>
  average_rating?: number
  engagement_score: number
  last_activity: string
  preference_vector: number[]
}

// 推荐项反馈分析
export interface ItemFeedbackAnalytics {
  item_id: string
  total_feedbacks: number
  average_rating?: number
  like_ratio: number
  engagement_metrics: {
    total_views: number
    total_clicks: number
    average_dwell_time: number
    bounce_rate: number
  }
  feedback_distribution: Record<FeedbackType, number>
}

// 反馈质量评分
export interface FeedbackQualityScore {
  feedback_id: string
  quality_score: number
  quality_factors: {
    consistency: number
    temporal_validity: number
    anomaly_score: number
    trust_score: number
  }
  is_valid: boolean
  reasons?: string[]
}

class FeedbackService {
  /**
   * 提交隐式反馈批次
   */
  async submitImplicitFeedback(batch: FeedbackBatch): Promise<ApiResponse> {
    try {
      const response = await apiClient.post<ApiResponse>(
        '/feedback/implicit',
        batch
      )
      return response.data
    } catch (error) {
      logger.error('[FeedbackService] 提交隐式反馈失败:', error)
      throw error
    }
  }

  /**
   * 提交显式反馈
   */
  async submitExplicitFeedback(
    feedback: ExplicitFeedback
  ): Promise<ApiResponse> {
    try {
      const response = await apiClient.post<ApiResponse>(
        '/feedback/explicit',
        feedback
      )
      return response.data
    } catch (error) {
      logger.error('[FeedbackService] 提交显式反馈失败:', error)
      throw error
    }
  }

  /**
   * 获取用户反馈历史
   */
  async getUserFeedbackHistory(
    query: FeedbackHistoryQuery
  ): Promise<ApiResponse<FeedbackEvent[]>> {
    try {
      const response = await apiClient.get<ApiResponse<FeedbackEvent[]>>(
        `/feedback/user/${query.user_id}`,
        { params: query }
      )
      return response.data
    } catch (error) {
      logger.error('[FeedbackService] 获取用户反馈历史失败:', error)
      throw error
    }
  }

  /**
   * 获取用户反馈分析
   */
  async getUserFeedbackAnalytics(
    userId: string
  ): Promise<ApiResponse<FeedbackAnalytics>> {
    try {
      const response = await apiClient.get<ApiResponse<FeedbackAnalytics>>(
        `/feedback/analytics/user/${userId}`
      )
      return response.data
    } catch (error) {
      logger.error('[FeedbackService] 获取用户反馈分析失败:', error)
      throw error
    }
  }

  /**
   * 获取推荐项反馈分析
   */
  async getItemFeedbackAnalytics(
    itemId: string
  ): Promise<ApiResponse<ItemFeedbackAnalytics>> {
    try {
      const response = await apiClient.get<ApiResponse<ItemFeedbackAnalytics>>(
        `/feedback/analytics/item/${itemId}`
      )
      return response.data
    } catch (error) {
      logger.error('[FeedbackService] 获取推荐项反馈分析失败:', error)
      throw error
    }
  }

  /**
   * 获取反馈质量评分
   */
  async getFeedbackQualityScore(
    feedbackIds: string[]
  ): Promise<ApiResponse<FeedbackQualityScore[]>> {
    try {
      const response = await apiClient.post<
        ApiResponse<FeedbackQualityScore[]>
      >('/feedback/quality/score', { feedback_ids: feedbackIds })
      return response.data
    } catch (error) {
      logger.error('[FeedbackService] 获取反馈质量评分失败:', error)
      throw error
    }
  }

  /**
   * 批量处理反馈
   */
  async processFeedbackBatch(batchId: string): Promise<ApiResponse> {
    try {
      const response = await apiClient.post<ApiResponse>(
        '/feedback/process/batch',
        { batch_id: batchId }
      )
      return response.data
    } catch (error) {
      logger.error('[FeedbackService] 批量处理反馈失败:', error)
      throw error
    }
  }

  /**
   * 计算奖励信号
   */
  async calculateRewardSignal(
    userId: string,
    itemId: string,
    timeWindow?: number
  ): Promise<ApiResponse<number>> {
    try {
      const response = await apiClient.post<ApiResponse<number>>(
        '/feedback/reward/calculate',
        {
          user_id: userId,
          item_id: itemId,
          time_window: timeWindow,
        }
      )
      return response.data
    } catch (error) {
      logger.error('[FeedbackService] 计算奖励信号失败:', error)
      throw error
    }
  }

  /**
   * 提交评分反馈
   */
  async submitRating(
    userId: string,
    itemId: string,
    rating: number,
    sessionId: string,
    metadata?: Record<string, any>
  ): Promise<ApiResponse> {
    const feedback: ExplicitFeedback = {
      feedback_id: `rating-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
      user_id: userId,
      session_id: sessionId,
      item_id: itemId,
      feedback_type: FeedbackType.RATING,
      value: rating,
      context: {
        url: window.location.href,
        page_title: document.title,
        timestamp: Date.now(),
        user_agent: navigator.userAgent,
      },
      metadata,
      timestamp: Date.now(),
    }

    return this.submitExplicitFeedback(feedback)
  }

  /**
   * 提交点赞/踩反馈
   */
  async submitLike(
    userId: string,
    itemId: string,
    isLike: boolean,
    sessionId: string,
    metadata?: Record<string, any>
  ): Promise<ApiResponse> {
    const feedback: ExplicitFeedback = {
      feedback_id: `like-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
      user_id: userId,
      session_id: sessionId,
      item_id: itemId,
      feedback_type: isLike ? FeedbackType.LIKE : FeedbackType.DISLIKE,
      value: isLike ? 1 : 0,
      context: {
        url: window.location.href,
        page_title: document.title,
        timestamp: Date.now(),
        user_agent: navigator.userAgent,
      },
      metadata,
      timestamp: Date.now(),
    }

    return this.submitExplicitFeedback(feedback)
  }

  /**
   * 提交收藏反馈
   */
  async submitBookmark(
    userId: string,
    itemId: string,
    isBookmarked: boolean,
    sessionId: string,
    metadata?: Record<string, any>
  ): Promise<ApiResponse> {
    const feedback: ExplicitFeedback = {
      feedback_id: `bookmark-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
      user_id: userId,
      session_id: sessionId,
      item_id: itemId,
      feedback_type: FeedbackType.BOOKMARK,
      value: isBookmarked ? 1 : 0,
      context: {
        url: window.location.href,
        page_title: document.title,
        timestamp: Date.now(),
        user_agent: navigator.userAgent,
      },
      metadata,
      timestamp: Date.now(),
    }

    return this.submitExplicitFeedback(feedback)
  }

  /**
   * 提交分享反馈
   */
  async submitShare(
    userId: string,
    itemId: string,
    shareType: string,
    sessionId: string,
    metadata?: Record<string, any>
  ): Promise<ApiResponse> {
    const feedback: ExplicitFeedback = {
      feedback_id: `share-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
      user_id: userId,
      session_id: sessionId,
      item_id: itemId,
      feedback_type: FeedbackType.SHARE,
      value: shareType,
      context: {
        url: window.location.href,
        page_title: document.title,
        timestamp: Date.now(),
        user_agent: navigator.userAgent,
      },
      metadata,
      timestamp: Date.now(),
    }

    return this.submitExplicitFeedback(feedback)
  }

  /**
   * 提交评论反馈
   */
  async submitComment(
    userId: string,
    itemId: string,
    comment: string,
    sessionId: string,
    metadata?: Record<string, any>
  ): Promise<ApiResponse> {
    const feedback: ExplicitFeedback = {
      feedback_id: `comment-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
      user_id: userId,
      session_id: sessionId,
      item_id: itemId,
      feedback_type: FeedbackType.COMMENT,
      value: comment,
      context: {
        url: window.location.href,
        page_title: document.title,
        timestamp: Date.now(),
        user_agent: navigator.userAgent,
      },
      metadata,
      timestamp: Date.now(),
    }

    return this.submitExplicitFeedback(feedback)
  }

  /**
   * 获取反馈统计概览
   */
  async getFeedbackOverview(
    startDate?: string,
    endDate?: string
  ): Promise<
    ApiResponse<{
      total_feedbacks: number
      feedback_types: Record<FeedbackType, number>
      unique_users: number
      average_quality_score: number
      top_items: Array<{ item_id: string; feedback_count: number }>
    }>
  > {
    try {
      const response = await apiClient.get<ApiResponse<any>>(
        '/feedback/overview',
        {
          params: {
            start_date: startDate,
            end_date: endDate,
          },
        }
      )
      return response.data
    } catch (error) {
      logger.error('[FeedbackService] 获取反馈统计概览失败:', error)
      throw error
    }
  }

  /**
   * 获取实时反馈指标
   */
  async getRealTimeFeedbackMetrics(): Promise<
    ApiResponse<{
      active_sessions: number
      events_per_minute: number
      buffer_status: Record<string, number>
      processing_latency: number
    }>
  > {
    try {
      const response = await apiClient.get<ApiResponse<any>>(
        '/feedback/metrics/realtime'
      )
      return response.data
    } catch (error) {
      logger.error('[FeedbackService] 获取实时反馈指标失败:', error)
      throw error
    }
  }
}

// 导出单例实例
export const feedbackService = new FeedbackService()
export default feedbackService
