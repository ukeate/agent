/**
 * 多模态RAG服务层
 * 与后端API交互（不返回任何静态假数据）
 */

import { apiFetch, buildApiUrl } from '../utils/apiBase'
import apiClient from './apiClient'

export interface MultimodalQueryResponse {
  answer: string
  sources: string[]
  confidence: number
  processing_time: number
  context_used: Record<string, number>
}

export interface DocumentUploadResponse {
  doc_id: string
  source_file: string
  content_type: string
  num_text_chunks: number
  num_images: number
  num_tables: number
  processing_time: number
}

export interface MultimodalRagStats {
  vector_store_type: string
  embedding_model: string
  embedding_dimension: number
  collections: Array<{ name: string; count: number }>
  text_documents: number
  image_documents: number
  table_documents: number
  total_documents: number
  cache: {
    enabled: boolean
    size: number
    hits: number
    misses: number
    hit_rate: number
  }
  retrieval: {
    top_k: number
    similarity_threshold: number
    rerank_enabled: boolean
  }
}

export interface MultimodalQueryWithDetailsResponse {
  query_analysis: {
    query_type: 'text' | 'visual' | 'document' | 'mixed'
    requires_image_search: boolean
    requires_table_search: boolean
    filters: Record<string, any>
    top_k: number
    similarity_threshold: number
  }
  retrieval_strategy: {
    strategy: 'text' | 'visual' | 'document' | 'hybrid'
    weights: { text: number; image: number; table: number }
    reranking: boolean
    top_k: number
    similarity_threshold: number
  }
  retrieval_results: {
    texts: Array<{ content: string; score: number; source: string; metadata: Record<string, any> }>
    images: Array<{ content: string; score: number; source: string; metadata: Record<string, any> }>
    tables: Array<{ content: string; score: number; source: string; metadata: Record<string, any> }>
    sources: string[]
    total_results: number
    retrieval_time_ms: number
  }
  qa_response: {
    answer: string
    confidence: number
    processing_time_ms: number
    context_used: Record<string, number>
  }
}

class MultimodalRagService {
  private baseUrl = '/multimodal-rag'

  async getSystemStatus(): Promise<MultimodalRagStats> {
    const response = await apiClient.get<MultimodalRagStats>(`${this.baseUrl}/stats`)
    return response.data
  }

  async query(query: string, options?: {
    topK?: number
    temperature?: number
    includeImages?: boolean
    includeTables?: boolean
    maxTokens?: number
    stream?: boolean
  }): Promise<MultimodalQueryResponse> {
    const response = await apiClient.post<MultimodalQueryResponse>(`${this.baseUrl}/query`, {
      query,
      stream: options?.stream ?? false,
      max_tokens: options?.maxTokens ?? 1000,
      temperature: options?.temperature ?? 0.7,
      include_images: options?.includeImages ?? true,
      include_tables: options?.includeTables ?? true,
      top_k: options?.topK ?? 5,
    })
    return response.data
  }

  async queryWithFiles(query: string, files: File[]): Promise<MultimodalQueryResponse> {
    const formData = new FormData()
    formData.append('query', query)
    files.forEach((file) => formData.append('files', file))

    const response = await apiClient.post<MultimodalQueryResponse>(`${this.baseUrl}/query-with-files`, formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    })
    return response.data
  }

  async queryWithDetails(query: string, files?: File[]): Promise<MultimodalQueryWithDetailsResponse> {
    const formData = new FormData()
    formData.append('query', query)
    files?.forEach((file) => formData.append('files', file))
    const response = await apiClient.post<MultimodalQueryWithDetailsResponse>(`${this.baseUrl}/query-with-details`, formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    })
    return response.data
  }

  async streamQuery(query: string, options?: {
    topK?: number
    temperature?: number
    includeImages?: boolean
    includeTables?: boolean
    maxTokens?: number
  }): Promise<ReadableStream<Uint8Array>> {
    const response = await apiFetch(buildApiUrl('/multimodal-rag/stream-query'), {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Accept': 'text/event-stream',
      },
      body: JSON.stringify({
        query,
        stream: true,
        max_tokens: options?.maxTokens ?? 1000,
        temperature: options?.temperature ?? 0.7,
        include_images: options?.includeImages ?? true,
        include_tables: options?.includeTables ?? true,
        top_k: options?.topK ?? 5,
      }),
    })

    if (!response.body) {
      throw new Error('No response body')
    }
    return response.body
  }

  async uploadSingleDocument(file: File): Promise<DocumentUploadResponse> {
    const formData = new FormData()
    formData.append('file', file)
    const response = await apiClient.post<DocumentUploadResponse>(`${this.baseUrl}/upload-document`, formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    })
    return response.data
  }

  async batchUploadDocuments(files: File[]): Promise<any> {
    const formData = new FormData()
    files.forEach((file) => formData.append('files', file))
    const response = await apiClient.post(`${this.baseUrl}/batch-upload`, formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    })
    return response.data
  }

  async clearVectorDatabase(): Promise<{ message: string }> {
    const response = await apiClient.delete<{ message: string }>(`${this.baseUrl}/clear`)
    return response.data
  }

  async clearCache(): Promise<{ message: string }> {
    const response = await apiClient.delete<{ message: string }>(`${this.baseUrl}/cache`)
    return response.data
  }
}

export const multimodalRagService = new MultimodalRagService()
