/**
 * Agentic RAG服务
 * 提供智能检索系统的前端接口
 */

import { apiFetch, buildApiUrl } from '../utils/apiBase'
import { consumeSseJson } from '../utils/sse'
import apiClient from './apiClient'

import { logger } from '../utils/logger'
// 枚举类型定义
export enum QueryIntentType {
  FACTUAL = 'factual',
  ANALYTICAL = 'analytical',
  COMPARATIVE = 'comparative',
  EXPLORATORY = 'exploratory',
  INSTRUCTIONAL = 'instructional',
}

export enum ExpansionStrategyType {
  SEMANTIC = 'semantic',
  SYNONYM = 'synonym',
  CONTEXTUAL = 'contextual',
  MULTILINGUAL = 'multilingual',
  DECOMPOSITION = 'decomposition',
}

export enum RetrievalStrategyType {
  SEMANTIC = 'semantic',
  KEYWORD = 'keyword',
  STRUCTURED = 'structured',
  HYBRID = 'hybrid',
}

export enum StreamEventType {
  QUERY_ANALYSIS = 'query_analysis',
  QUERY_EXPANSION = 'query_expansion',
  RETRIEVAL_START = 'retrieval_start',
  RETRIEVAL_COMPLETE = 'retrieval_complete',
  VALIDATION_START = 'validation_start',
  VALIDATION_COMPLETE = 'validation_complete',
  CONTEXT_COMPOSITION = 'context_composition',
  EXPLANATION_GENERATED = 'explanation_generated',
  COMPLETE = 'complete',
  ERROR = 'error',
}

// 接口定义
export interface AgenticQueryRequest {
  query: string
  context_history?: string[]
  expansion_strategies?: ExpansionStrategyType[]
  retrieval_strategies?: RetrievalStrategyType[]
  max_results?: number
  include_explanation?: boolean
  session_id?: string
}

export interface QueryAnalysisInfo {
  intent_type: QueryIntentType
  confidence: number
  complexity_score: number
  entities: string[]
  keywords: string[]
  domain?: string
  language?: string
}

export interface ExpandedQueryInfo {
  original_query: string
  expanded_queries: string[]
  strategy: ExpansionStrategyType
  confidence: number
  sub_questions?: string[]
  language_variants?: Record<string, string>
  explanation?: string
}

export interface RetrievalResultInfo {
  agent_type: RetrievalStrategyType
  results: Array<{
    id: string
    content: string
    file_path?: string
    content_type?: string
    metadata?: Record<string, any>
    score: number
  }>
  score: number
  confidence: number
  processing_time: number
  explanation?: string
}

export interface ValidationResultInfo {
  quality_scores: Record<string, any>
  conflicts: string[]
  overall_quality: number
  overall_confidence: number
  recommendations: string[]
}

export interface ComposedContext {
  fragments: any[]
  relationships: any[]
  total_tokens: number
  diversity_score: number
  coherence_score: number
  information_density: number
}

export interface RetrievalExplanation {
  path_record: {
    path_id: string
    query: string
    decision_points: Array<{
      step: string
      decision: string
      rationale: string
      confidence: number
      alternatives: string[]
      timestamp: Date
    }>
    total_time: number
    success: boolean
    final_results_count: number
    created_at: Date
  }
  confidence_analysis: {
    overall_confidence: number
    confidence_level: string
    uncertainty_factors: string[]
    confidence_explanation: string
  }
  summary: string
  detailed_explanation: string
  improvement_suggestions: string[]
  visualization_data: {
    flow_diagram?: any
    metrics_chart?: any
    timeline?: any
  }
}

export interface AgenticQueryResponse {
  success: boolean
  query_id: string
  query_analysis?: QueryAnalysisInfo
  expanded_queries?: ExpandedQueryInfo[]
  retrieval_results?: RetrievalResultInfo[]
  validation_result?: ValidationResultInfo
  composed_context?: ComposedContext
  explanation?: RetrievalExplanation
  fallback_result?: any
  processing_time: number
  timestamp: Date
  session_id?: string
  error?: string
}

export interface StreamEvent {
  event_type: StreamEventType
  data: any
  progress: number
  message: string
  timestamp: Date
}

export interface AgenticRagStats {
  total_queries: number
  successful_queries: number
  failed_queries: number
  average_response_time: number
  average_quality_score: number
  strategy_usage: Record<string, number>
  failure_patterns: Record<string, number>
  performance_metrics: {
    success_rate: number
    avg_response_time: number
    avg_quality_score: number
  }
  updated_at: Date
}

export interface HealthCheckResponse {
  status: 'healthy' | 'unhealthy'
  components: Record<string, string>
  stats?: AgenticRagStats
  error?: string
}

class AgenticRagService {
  /**
   * 执行智能检索查询
   */
  async query(request: AgenticQueryRequest): Promise<AgenticQueryResponse> {
    try {
      const response = await apiClient.post('/rag/agentic/query', request)
      return response.data
    } catch (error) {
      logger.error('Agentic RAG查询失败:', error)
      throw error
    }
  }

  /**
   * 执行流式智能检索查询
   */
  async queryStream(
    request: AgenticQueryRequest,
    onEvent: (event: StreamEvent) => void
  ): Promise<void> {
    try {
      const response = await apiFetch(
        buildApiUrl('/rag/agentic/query/stream'),
        {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            Accept: 'text/event-stream',
          },
          body: JSON.stringify(request),
        }
      )

      await consumeSseJson<StreamEvent>(
        response,
        data => {
          onEvent(data)
        },
        {
          onParseError: error => {
            logger.error('解析流式数据失败:', error)
          },
        }
      )
    } catch (error) {
      logger.error('流式查询失败:', error)
      throw error
    }
  }

  /**
   * 获取检索解释
   */
  async getExplanation(
    queryId?: string,
    pathId?: string
  ): Promise<RetrievalExplanation> {
    try {
      const params = new URLSearchParams()
      if (queryId) params.append('query_id', queryId)
      if (pathId) params.append('path_id', pathId)
      params.append('explanation_level', 'detailed')
      params.append('include_visualization', 'true')

      const response = await apiClient.get(
        `/rag/agentic/explain?${params.toString()}`
      )
      return response.data
    } catch (error) {
      logger.error('获取检索解释失败:', error)
      throw error
    }
  }

  /**
   * 提交用户反馈
   */
  async submitFeedback(feedback: {
    query_id: string
    ratings: Record<string, number>
    comments?: string
    improvements?: string[]
  }): Promise<{ success: boolean; message: string; feedback_id: string }> {
    try {
      const response = await apiClient.post('/rag/agentic/feedback', feedback)
      return response.data
    } catch (error) {
      logger.error('提交反馈失败:', error)
      throw error
    }
  }

  /**
   * 获取统计信息
   */
  async getStats(): Promise<AgenticRagStats> {
    try {
      const response = await apiClient.get('/rag/agentic/stats')
      return response.data
    } catch (error) {
      logger.error('获取统计信息失败:', error)
      throw error
    }
  }

  /**
   * 健康检查
   */
  async healthCheck(): Promise<HealthCheckResponse> {
    try {
      const response = await apiClient.get('/rag/agentic/health')
      return response.data
    } catch (error) {
      logger.error('健康检查失败:', error)
      throw error
    }
  }
}

export default new AgenticRagService()
