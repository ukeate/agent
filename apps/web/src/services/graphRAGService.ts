/**
 * GraphRAG服务
 *
 * 仅保留后端真实存在的接口调用，不包含任何静态/模拟数据。
 */

import apiClient from './apiClient'

type ApiEnvelope<T> = {
  success: boolean
  data?: T
  error?: string | null
  [k: string]: any
}

type GraphRAGApiRetrievalMode = 'vector_only' | 'graph_only' | 'hybrid' | 'adaptive'
export type GraphRAGRetrievalMode = 'vector' | 'graph' | 'hybrid' | 'adaptive'

const toApiRetrievalMode = (mode?: GraphRAGRetrievalMode): GraphRAGApiRetrievalMode | undefined => {
  if (!mode) return undefined
  if (mode === 'vector') return 'vector_only'
  if (mode === 'graph') return 'graph_only'
  return mode
}

const unwrap = <T,>(payload: ApiEnvelope<T>): T => {
  if (!payload?.success) throw new Error(payload?.error || '请求失败')
  if (payload.data === undefined) throw new Error('后端未返回data')
  return payload.data
}

export interface GraphRAGDocument {
  id?: string
  content?: string
  score?: number
  final_score?: number
  metadata?: any
  source?: string
  [k: string]: any
}

export interface GraphRAGResponse {
  query_id: string
  query: string
  documents: GraphRAGDocument[]
  graph_context: {
    entities: any[]
    relations: any[]
    subgraph?: any
    reasoning_paths?: any[]
    expansion_depth?: number
    confidence_score?: number
    [k: string]: any
  }
  reasoning_results: any[]
  knowledge_sources: Array<{
    source_type: string
    content: string
    confidence: number
    metadata: any
    graph_context?: any
  }>
  fusion_results: Record<string, any>
  performance_metrics: Record<string, any>
  timestamp: string
}

export interface GraphRAGQueryRequest {
  query: string
  retrieval_mode?: GraphRAGRetrievalMode
  max_docs?: number
  include_reasoning?: boolean
  expansion_depth?: number
  confidence_threshold?: number
  query_type?: string
  filters?: Record<string, any> | null
}

export interface GraphRAGQueryAnalysis {
  decomposition: any
  analysis: {
    detected_query_type: string
    complexity_score: number
    sub_queries_count: number
    entities_count: number
    relations_count: number
    [k: string]: any
  }
}

export interface GraphRAGReasoningQueryResult {
  paths: any[]
  statistics: {
    total_paths: number
    avg_hops: number
    avg_score: number
    [k: string]: any
  }
}

export interface GraphRAGMultiSourceFusionRequest {
  vector_results?: any[]
  graph_results?: Record<string, any>
  reasoning_results?: any[]
  confidence_threshold?: number
}

export interface GraphRAGMultiSourceFusionResult {
  fusion_results: any
  statistics: any
}

export interface GraphRAGPerformanceStats {
  query_statistics: {
    total_queries: number
    successful_queries: number
    failed_queries: number
    average_response_time: number
    queries_by_type: Record<string, number>
  }
  performance_metrics: {
    average_retrieval_time: number
    average_reasoning_time: number
    average_fusion_time: number
    cache_hit_rate: number
  }
  time_period: {
    start_time: string | null
    end_time: string | null
  }
}

export interface GraphRAGPerformanceComparison {
  comparison_summary: {
    performance_change: number
    response_time_change: number
    accuracy_change: number
    throughput_change: number
  }
  detailed_comparison: {
    baseline_metrics: any
    comparison_metrics: any
    differences: Record<string, number>
  }
  recommendations: string[]
}

export interface GraphRAGBenchmarkRequest {
  test_queries: string[]
  config?: {
    max_docs?: number
    include_reasoning?: boolean
    confidence_threshold?: number
  }
}

export interface GraphRAGBenchmarkResult {
  results: any[]
  statistics: any
}

export interface GraphRAGConfigResponse {
  config: any
  status: string
}

export interface GraphRAGConsistencyCheck {
  query_id: string
  consistency_score: number
  documents_checked: number
}

export interface GraphRAGExplainResult {
  explanation: string
  reasoning_paths_count: number
  query: string
}

export interface GraphRAGTraceResult {
  query_id: string
  steps: any[]
  results: any
  total_duration_ms: number
}

class GraphRAGService {
  private baseUrl = '/graphrag'

  async query(request: GraphRAGQueryRequest): Promise<GraphRAGResponse> {
    const response = await apiClient.post(`${this.baseUrl}/query`, {
      ...request,
      retrieval_mode: toApiRetrievalMode(request.retrieval_mode),
    })
    return unwrap<GraphRAGResponse>(response.data)
  }

  async analyzeQuery(query: string, queryType?: string): Promise<GraphRAGQueryAnalysis> {
    const response = await apiClient.post(`${this.baseUrl}/query/analyze`, {
      query,
      query_type: queryType,
    })
    return unwrap<GraphRAGQueryAnalysis>(response.data)
  }

  async queryReasoning(
    entity1: string,
    entity2: string,
    maxHops: number = 3,
    maxPaths: number = 10
  ): Promise<GraphRAGReasoningQueryResult> {
    const response = await apiClient.post(`${this.baseUrl}/query/reasoning`, {
      entity1,
      entity2,
      max_hops: maxHops,
      max_paths: maxPaths,
    })
    return unwrap<GraphRAGReasoningQueryResult>(response.data)
  }

  async getQueryResult(queryId: string): Promise<any> {
    const response = await apiClient.get(`${this.baseUrl}/query/${queryId}`)
    return unwrap<any>(response.data)
  }

  async multiSourceFusion(request: GraphRAGMultiSourceFusionRequest): Promise<GraphRAGMultiSourceFusionResult> {
    const response = await apiClient.post(`${this.baseUrl}/fusion/multi-source`, request)
    return unwrap<GraphRAGMultiSourceFusionResult>(response.data)
  }

  async conflictResolution(request: {
    knowledge_sources: Array<{
      source_type?: string
      content: string
      confidence?: number
      metadata?: Record<string, any>
    }>
    conflicts?: any[]
    strategy?: string
  }): Promise<any> {
    const response = await apiClient.post(`${this.baseUrl}/fusion/conflict-resolution`, request)
    return unwrap<any>(response.data)
  }

  async getConsistencyCheck(queryId?: string): Promise<GraphRAGConsistencyCheck> {
    const response = await apiClient.get(`${this.baseUrl}/fusion/consistency`, {
      params: queryId ? { query_id: queryId } : undefined,
    })
    return unwrap<GraphRAGConsistencyCheck>(response.data)
  }

  async getPerformanceStats(startTime?: string, endTime?: string): Promise<GraphRAGPerformanceStats> {
    const params: any = {}
    if (startTime) params.start_time = startTime
    if (endTime) params.end_time = endTime
    const response = await apiClient.get(`${this.baseUrl}/performance/stats`, { params })
    return unwrap<GraphRAGPerformanceStats>(response.data)
  }

  async getPerformanceComparison(baselineDate: string, comparisonDate: string): Promise<GraphRAGPerformanceComparison> {
    const response = await apiClient.get(`${this.baseUrl}/performance/comparison`, {
      params: {
        baseline_date: baselineDate,
        comparison_date: comparisonDate,
      },
    })
    return unwrap<GraphRAGPerformanceComparison>(response.data)
  }

  async runBenchmark(request: GraphRAGBenchmarkRequest): Promise<GraphRAGBenchmarkResult> {
    const response = await apiClient.post(`${this.baseUrl}/performance/benchmark`, request)
    return unwrap<GraphRAGBenchmarkResult>(response.data)
  }

  async getConfig(): Promise<GraphRAGConfigResponse> {
    const response = await apiClient.get(`${this.baseUrl}/config`)
    return unwrap<GraphRAGConfigResponse>(response.data)
  }

  async updateConfig(request: {
    max_expansion_depth?: number
    confidence_threshold?: number
    max_reasoning_paths?: number
  }): Promise<{ updated_config: any; message: string }> {
    const response = await apiClient.put(`${this.baseUrl}/config`, request)
    return unwrap<{ updated_config: any; message: string }>(response.data)
  }

  async explainResult(request: { query: string; reasoning_paths: any[] }): Promise<GraphRAGExplainResult> {
    const response = await apiClient.post(`${this.baseUrl}/debug/explain`, request)
    return unwrap<GraphRAGExplainResult>(response.data)
  }

  async getDebugTrace(queryId: string): Promise<GraphRAGTraceResult> {
    const response = await apiClient.get(`${this.baseUrl}/debug/trace/${queryId}`)
    return unwrap<GraphRAGTraceResult>(response.data)
  }
}

export const graphRAGService = new GraphRAGService()

