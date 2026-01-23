/**
 * RAG系统API服务
 *
 * 封装基础RAG和Agentic RAG的API调用
 */

import { AxiosResponse } from 'axios'
import apiClient from './apiClient'
import { apiFetch, buildApiUrl } from '../utils/apiBase'

import { logger } from '../utils/logger'
// ==================== 基础RAG接口类型 ====================

export interface QueryRequest {
  query: string
  search_type?: 'semantic' | 'keyword' | 'hybrid'
  limit?: number
  score_threshold?: number
  filters?: Record<string, any>
}

export interface QueryResponse {
  success: boolean
  results: KnowledgeItem[]
  query_id: string
  processing_time: number
  total_results: number
  error?: string
}

export interface KnowledgeItem {
  id: string
  content: string
  file_path?: string
  content_type?: string
  metadata: Record<string, any>
  score: number
}

export interface IndexResponse {
  success: boolean
  message: string
  indexed_files: number
  processing_time: number
  error?: string
}

export interface StatsResponse {
  success: boolean
  stats: {
    total_documents: number
    total_vectors: number
    index_size: number
    last_updated: string
  }
  error?: string
}

export interface AddDocumentRequest {
  text: string
  metadata?: Record<string, any>
}

export interface AddDocumentResponse {
  success: boolean
  document_id?: string
  message?: string
  chunks?: number
  text_length?: number
  metadata?: Record<string, any>
  error?: string
}

export interface DeleteDocumentResponse {
  success: boolean
  document_id?: string
  message?: string
  error?: string
}

// ==================== Agentic RAG接口类型 ====================

// 导出枚举类型
export type QueryIntentType =
  | 'factual'
  | 'procedural'
  | 'code'
  | 'creative'
  | 'exploratory'
export type ExpansionStrategyType =
  | 'synonym'
  | 'semantic'
  | 'contextual'
  | 'decomposition'
  | 'multilingual'
export type RetrievalStrategyType = 'semantic' | 'keyword' | 'structured'

export interface AgenticQueryRequest {
  query: string
  context_history?: string[]
  expansion_strategies?: string[]
  retrieval_strategies?: string[]
  max_results?: number
  include_explanation?: boolean
  session_id?: string
}

export interface AgenticQueryResponse {
  success: boolean
  query_id: string
  results: KnowledgeItem[] // 主要结果列表
  confidence: number // 总体置信度
  processing_time: number
  analysis_info?: QueryAnalysisInfo
  expanded_queries?: string[] // 简化为字符串数组
  expansion_strategies?: string[]
  retrieval_results?: RetrievalResultInfo[]
  validation_result?: ValidationResultInfo
  composed_context?: ComposedContext
  explanation?: RetrievalExplanation
  fallback_result?: FallbackResultInfo
  timestamp: string
  session_id?: string
  error?: string
}

export interface QueryAnalysisInfo {
  intent_type: 'factual' | 'procedural' | 'code' | 'creative' | 'exploratory'
  confidence: number
  complexity_score: number
  entities: string[]
  keywords: string[]
  domain?: string
  language: string
}

export interface ExpandedQueryInfo {
  original_query: string
  expanded_queries: string[]
  strategy: string
  confidence: number
  sub_questions?: string[]
  language_variants?: Record<string, string>
  explanation?: string
}

export interface RetrievalResultInfo {
  agent_type: 'semantic' | 'keyword' | 'structured'
  results: KnowledgeItem[]
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
  path_record: any
  confidence_analysis: any
  summary: string
  detailed_explanation: string
  improvement_suggestions: string[]
  visualization_data?: any
}

export interface FallbackResultInfo {
  original_failure: any
  actions_taken: any[]
  user_guidance: any
  success: boolean
  improvement_metrics: Record<string, number>
  total_time: number
}

// ==================== RAG服务类 ====================

export class RagService {
  // ==================== 基础RAG方法 ====================

  /**
   * 执行RAG查询
   */
  async query(request: QueryRequest): Promise<QueryResponse> {
    try {
      const response: AxiosResponse<QueryResponse> = await apiClient.post(
        '/rag/query',
        request
      )
      return response.data
    } catch (error) {
      throw this.handleError(error, 'RAG查询失败')
    }
  }

  /**
   * 添加文本到RAG索引
   */
  async addDocument(request: AddDocumentRequest): Promise<AddDocumentResponse> {
    try {
      const response: AxiosResponse<AddDocumentResponse> = await apiClient.post(
        '/rag/documents',
        request
      )
      return response.data
    } catch (error) {
      throw this.handleError(error, '添加文档失败')
    }
  }

  /**
   * 删除文档
   */
  async deleteDocument(documentId: string): Promise<DeleteDocumentResponse> {
    try {
      const response: AxiosResponse<DeleteDocumentResponse> =
        await apiClient.delete(`/rag/documents/${documentId}`)
      return response.data
    } catch (error) {
      throw this.handleError(error, '删除文档失败')
    }
  }

  /**
   * 索引单个文件
   */
  async indexFile(
    filePath: string,
    force: boolean = false
  ): Promise<IndexResponse> {
    try {
      const response: AxiosResponse<IndexResponse> = await apiClient.post(
        '/rag/index/file',
        {
          file_path: filePath,
          force,
        }
      )
      return response.data
    } catch (error) {
      throw this.handleError(error, '文件索引失败')
    }
  }

  /**
   * 索引目录
   */
  async indexDirectory(
    directory: string,
    recursive: boolean = true,
    force: boolean = false,
    extensions?: string[]
  ): Promise<IndexResponse> {
    try {
      const response: AxiosResponse<IndexResponse> = await apiClient.post(
        '/rag/index/directory',
        {
          directory,
          recursive,
          force,
          extensions,
        }
      )
      return response.data
    } catch (error) {
      throw this.handleError(error, '目录索引失败')
    }
  }

  /**
   * 获取索引统计
   */
  async getIndexStats(): Promise<StatsResponse> {
    try {
      const response: AxiosResponse<any> =
        await apiClient.get('/rag/index/stats')
      const backendData = response.data

      // 转换后端响应格式为前端期望的格式
      if (backendData.success && backendData.stats) {
        const documentsStats = backendData.stats.documents || {}
        const codeStats = backendData.stats.code || {}

        const transformedResponse: StatsResponse = {
          success: backendData.success,
          stats: {
            total_documents:
              (documentsStats.points_count || 0) +
              (codeStats.points_count || 0),
            total_vectors:
              (documentsStats.vectors_count || 0) +
              (codeStats.vectors_count || 0),
            index_size: backendData.total_disk_size || 0, // 使用后端返回的真实存储大小
            last_updated: new Date().toISOString(), // 使用当前时间作为默认值
          },
          error: backendData.error,
        }

        return transformedResponse
      }

      return backendData
    } catch (error) {
      throw this.handleError(error, '获取统计信息失败')
    }
  }

  /**
   * 重置索引
   */
  async resetIndex(collection?: string): Promise<IndexResponse> {
    try {
      const response: AxiosResponse<IndexResponse> = await apiClient.delete(
        '/rag/index/reset',
        {
          data: { collection },
        }
      )
      return response.data
    } catch (error) {
      throw this.handleError(error, '重置索引失败')
    }
  }

  /**
   * 健康检查
   */
  async healthCheck(): Promise<any> {
    try {
      const response = await apiClient.get('/rag/health')
      return response.data
    } catch (error) {
      throw this.handleError(error, '健康检查失败')
    }
  }

  // ==================== Agentic RAG方法 ====================

  /**
   * 执行Agentic RAG智能查询
   */
  async agenticQuery(
    request: AgenticQueryRequest
  ): Promise<AgenticQueryResponse> {
    try {
      const response: AxiosResponse<AgenticQueryResponse> =
        await apiClient.post('/rag/agentic/query', request)
      return response.data
    } catch (error) {
      throw this.handleError(error, 'Agentic RAG查询失败')
    }
  }

  /**
   * 执行Agentic RAG流式查询
   */
  async agenticQueryStream(
    request: AgenticQueryRequest
  ): Promise<ReadableStream> {
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

      if (!response.body) {
        throw new Error('Response body is null')
      }

      return response.body
    } catch (error) {
      throw this.handleError(error, 'Agentic RAG流式查询失败')
    }
  }

  /**
   * 获取检索解释
   */
  async getExplanation(
    queryId?: string,
    pathId?: string,
    explanationLevel: string = 'detailed',
    includeVisualization: boolean = true
  ): Promise<any> {
    try {
      const params = new URLSearchParams()
      if (queryId) params.append('query_id', queryId)
      if (pathId) params.append('path_id', pathId)
      params.append('explanation_level', explanationLevel)
      params.append('include_visualization', includeVisualization.toString())

      const response = await apiClient.get(
        `/rag/agentic/explain?${params.toString()}`
      )
      return response.data
    } catch (error) {
      throw this.handleError(error, '获取检索解释失败')
    }
  }

  /**
   * 提交用户反馈
   */
  async submitFeedback(feedback: {
    query_id: string
    ratings: Record<string, number>
    comments?: string
    helpful_results?: string[]
    problematic_results?: string[]
    suggestions?: string
  }): Promise<any> {
    try {
      const response = await apiClient.post('/rag/agentic/feedback', feedback)
      return response.data
    } catch (error) {
      throw this.handleError(error, '提交反馈失败')
    }
  }

  /**
   * 获取Agentic RAG统计
   */
  async getAgenticStats(): Promise<any> {
    try {
      const response = await apiClient.get('/rag/agentic/stats')
      return response.data
    } catch (error) {
      throw this.handleError(error, '获取Agentic RAG统计失败')
    }
  }

  /**
   * Agentic RAG健康检查
   */
  async agenticHealthCheck(): Promise<any> {
    try {
      const response = await apiClient.get('/rag/agentic/health')
      return response.data
    } catch (error) {
      throw this.handleError(error, 'Agentic RAG健康检查失败')
    }
  }

  // ==================== 辅助方法 ====================

  /**
   * 统一错误处理
   */
  private handleError(error: any, message: string): Error {
    logger.error(message, error)

    if (error.response) {
      // API返回错误
      return new Error(error.response.data?.error || message)
    } else if (error.request) {
      // 网络错误
      return new Error('网络连接失败，请检查网络状态')
    } else {
      // 其他错误
      return new Error(message)
    }
  }

  /**
   * 防抖处理
   */
  debounce<T extends (...args: any[]) => any>(func: T, delay: number): T {
    let timeoutId: ReturnType<typeof setTimeout>
    return ((...args: any[]) => {
      clearTimeout(timeoutId)
      timeoutId = setTimeout(() => func.apply(this, args), delay)
    }) as T
  }
}

// 创建单例实例
export const ragService = new RagService()

// 导出默认实例
export default ragService
