import apiClient from './apiClient'

import { logger } from '../utils/logger'
// ==================== 类型定义 ====================

export interface Document {
  id: string
  title: string
  content: string
  type: 'text' | 'markdown' | 'html' | 'pdf' | 'docx'
  metadata?: {
    author?: string
    created_at: string
    updated_at: string
    tags?: string[]
    category?: string
    version?: number
    size_bytes?: number
    language?: string
  }
  embeddings?: number[]
  chunks?: DocumentChunk[]
}

export interface DocumentChunk {
  id: string
  document_id: string
  content: string
  chunk_index: number
  start_char: number
  end_char: number
  embedding?: number[]
  metadata?: Record<string, any>
}

export interface DocumentUploadRequest {
  title: string
  content?: string
  file?: File
  type: string
  metadata?: Record<string, any>
  auto_chunk?: boolean
  chunk_size?: number
  chunk_overlap?: number
}

export interface DocumentSearchRequest {
  query: string
  limit?: number
  offset?: number
  filters?: {
    type?: string
    tags?: string[]
    category?: string
    author?: string
    date_from?: string
    date_to?: string
  }
  search_type?: 'keyword' | 'semantic' | 'hybrid'
  min_score?: number
}

export interface DocumentSearchResult {
  document: Document
  score: number
  highlights?: string[]
  relevant_chunks?: DocumentChunk[]
}

export interface DocumentAnalysis {
  document_id: string
  summary: string
  key_topics: string[]
  entities: {
    people: string[]
    organizations: string[]
    locations: string[]
    dates: string[]
  }
  sentiment: {
    overall: 'positive' | 'negative' | 'neutral'
    score: number
  }
  language: string
  readability_score: number
}

export interface DocumentProcessingStatus {
  document_id: string
  status: 'pending' | 'processing' | 'completed' | 'failed'
  progress: number
  current_step?: string
  error?: string
  created_at: string
  updated_at: string
}

// ==================== Service Class ====================

class DocumentsService {
  private baseUrl = '/documents'

  // ==================== 文档管理 ====================

  // 单文档上传 - 对应 /upload 端点
  async uploadDocument(
    file: File,
    options: {
      enableOcr?: boolean
      extractImages?: boolean
      autoTag?: boolean
      chunkStrategy?:
        | 'semantic'
        | 'fixed'
        | 'adaptive'
        | 'sliding_window'
        | 'hierarchical'
    } = {}
  ): Promise<any> {
    try {
      const formData = new FormData()
      formData.append('file', file)

      const response = await apiClient.post(
        `${this.baseUrl}/upload`,
        formData,
        {
          headers: {
            'Content-Type': 'multipart/form-data',
          },
          params: {
            enable_ocr: options.enableOcr || false,
            extract_images: options.extractImages !== false,
            auto_tag: options.autoTag !== false,
            chunk_strategy: options.chunkStrategy || 'semantic',
          },
        }
      )
      return response.data
    } catch (error) {
      logger.error('文档上传失败:', error)
      throw error
    }
  }

  // 批量文档上传 - 对应 /batch-upload 端点
  async batchUploadDocuments(
    files: File[],
    options: {
      concurrentLimit?: number
      continueOnError?: boolean
    } = {}
  ): Promise<any> {
    try {
      const formData = new FormData()
      files.forEach(file => {
        formData.append('files', file)
      })

      const response = await apiClient.post(
        `${this.baseUrl}/batch-upload`,
        formData,
        {
          headers: {
            'Content-Type': 'multipart/form-data',
          },
          params: {
            concurrent_limit: options.concurrentLimit || 5,
            continue_on_error: options.continueOnError !== false,
          },
        }
      )
      return response.data
    } catch (error) {
      logger.error('批量上传失败:', error)
      throw error
    }
  }

  // 获取支持的文档格式 - 对应 /supported-formats 端点
  async getSupportedFormats(): Promise<any> {
    try {
      const response = await apiClient.get(`${this.baseUrl}/supported-formats`)
      return response.data
    } catch (error) {
      logger.error('获取支持格式失败:', error)
      throw error
    }
  }

  // 分析文档关系 - 对应 /{doc_id}/analyze-relationships 端点
  async analyzeDocumentRelationships(
    docId: string,
    relatedDocIds: string[] = []
  ): Promise<any> {
    try {
      const response = await apiClient.post(
        `${this.baseUrl}/${docId}/analyze-relationships`,
        null,
        {
          params: {
            related_doc_ids: relatedDocIds,
          },
        }
      )
      return response.data
    } catch (error) {
      logger.error('文档关系分析失败:', error)
      throw error
    }
  }

  // 生成文档标签 - 对应 /{doc_id}/generate-tags 端点
  async generateDocumentTags(
    docId: string,
    content?: string,
    existingTags: string[] = []
  ): Promise<any> {
    try {
      const params: any = { existing_tags: existingTags }
      if (content !== undefined) params.content = content
      const response = await apiClient.post(
        `${this.baseUrl}/${docId}/generate-tags`,
        null,
        {
          params,
        }
      )
      return response.data
    } catch (error) {
      logger.error('标签生成失败:', error)
      throw error
    }
  }

  // 获取文档版本历史 - 对应 /{doc_id}/versions 端点
  async getDocumentVersionHistory(docId: string, limit?: number): Promise<any> {
    try {
      const response = await apiClient.get(
        `${this.baseUrl}/${docId}/versions`,
        {
          params: limit ? { limit } : {},
        }
      )
      return response.data
    } catch (error) {
      logger.error('获取版本历史失败:', error)
      throw error
    }
  }

  // 回滚文档版本 - 对应 /{doc_id}/rollback 端点
  async rollbackDocumentVersion(
    docId: string,
    targetVersionId: string
  ): Promise<any> {
    try {
      const response = await apiClient.post(
        `${this.baseUrl}/${docId}/rollback`,
        null,
        {
          params: {
            target_version_id: targetVersionId,
          },
        }
      )
      return response.data
    } catch (error) {
      logger.error('版本回滚失败:', error)
      throw error
    }
  }

  async getDocument(documentId: string): Promise<Document> {
    const response = await apiClient.get(`${this.baseUrl}/${documentId}`)
    return response.data
  }

  async updateDocument(
    documentId: string,
    updates: Partial<Document>
  ): Promise<Document> {
    const response = await apiClient.put(
      `${this.baseUrl}/${documentId}`,
      updates
    )
    return response.data
  }

  async deleteDocument(
    documentId: string
  ): Promise<{ success: boolean; message: string }> {
    const response = await apiClient.delete(`${this.baseUrl}/${documentId}`)
    return response.data
  }

  async listDocuments(params?: {
    limit?: number
    offset?: number
    type?: string
    category?: string
    tags?: string[]
    sort_by?: string
    order?: 'asc' | 'desc'
  }): Promise<{
    documents: Document[]
    total: number
    page: number
    limit: number
  }> {
    const response = await apiClient.get(this.baseUrl, { params })
    return response.data
  }

  // ==================== 文档搜索 ====================

  async searchDocuments(request: DocumentSearchRequest): Promise<{
    results: DocumentSearchResult[]
    total: number
    query: string
    processing_time_ms: number
  }> {
    const response = await apiClient.post(`${this.baseUrl}/search`, request)
    return response.data
  }

  async similarDocuments(
    documentId: string,
    limit: number = 10
  ): Promise<DocumentSearchResult[]> {
    const response = await apiClient.get(
      `${this.baseUrl}/${documentId}/similar`,
      {
        params: { limit },
      }
    )
    return response.data
  }

  // ==================== 文档处理 ====================

  async processDocument(
    documentId: string,
    operations?: string[]
  ): Promise<DocumentProcessingStatus> {
    const response = await apiClient.post(
      `${this.baseUrl}/${documentId}/process`,
      {
        operations: operations || ['chunk', 'embed', 'analyze'],
      }
    )
    return response.data
  }

  async getProcessingStatus(
    documentId: string
  ): Promise<DocumentProcessingStatus> {
    const response = await apiClient.get(`${this.baseUrl}/${documentId}/status`)
    return response.data
  }

  async chunkDocument(
    documentId: string,
    chunkSize: number = 512,
    overlap: number = 50
  ): Promise<DocumentChunk[]> {
    const response = await apiClient.post(
      `${this.baseUrl}/${documentId}/chunk`,
      {
        chunk_size: chunkSize,
        chunk_overlap: overlap,
      }
    )
    return response.data
  }

  async getDocumentChunks(documentId: string): Promise<DocumentChunk[]> {
    const response = await apiClient.get(`${this.baseUrl}/${documentId}/chunks`)
    return response.data
  }

  // ==================== 文档分析 ====================

  async analyzeDocument(documentId: string): Promise<DocumentAnalysis> {
    const response = await apiClient.post(
      `${this.baseUrl}/${documentId}/analyze`
    )
    return response.data
  }

  async summarizeDocument(
    documentId: string,
    maxLength?: number
  ): Promise<{
    summary: string
    key_points: string[]
  }> {
    const response = await apiClient.post(
      `${this.baseUrl}/${documentId}/summarize`,
      {
        max_length: maxLength,
      }
    )
    return response.data
  }

  async extractEntities(documentId: string): Promise<{
    entities: Record<string, string[]>
    relationships: Array<{
      source: string
      target: string
      type: string
    }>
  }> {
    const response = await apiClient.post(
      `${this.baseUrl}/${documentId}/extract-entities`
    )
    return response.data
  }

  // ==================== 文档导出 ====================

  async exportDocument(
    documentId: string,
    format: 'pdf' | 'docx' | 'html' | 'markdown'
  ): Promise<Blob> {
    const response = await apiClient.get(
      `${this.baseUrl}/${documentId}/export`,
      {
        params: { format },
        responseType: 'blob',
      }
    )
    return response.data
  }

  async downloadDocument(documentId: string): Promise<Blob> {
    const response = await apiClient.get(
      `${this.baseUrl}/${documentId}/download`,
      {
        responseType: 'blob',
      }
    )
    return response.data
  }

  // ==================== 批量操作 ====================

  async batchUpload(files: File[]): Promise<Document[]> {
    const formData = new FormData()
    files.forEach(file => {
      formData.append('files', file)
    })

    const response = await apiClient.post(`${this.baseUrl}/batch`, formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    })
    return response.data
  }

  async batchDelete(documentIds: string[]): Promise<{
    success: number
    failed: number
    errors: Array<{ document_id: string; error: string }>
  }> {
    const response = await apiClient.post(`${this.baseUrl}/batch/delete`, {
      document_ids: documentIds,
    })
    return response.data
  }

  async batchProcess(
    documentIds: string[],
    operations: string[]
  ): Promise<{
    statuses: DocumentProcessingStatus[]
  }> {
    const response = await apiClient.post(`${this.baseUrl}/batch/process`, {
      document_ids: documentIds,
      operations,
    })
    return response.data
  }

  // ==================== 版本控制 ====================

  async getDocumentVersions(documentId: string): Promise<
    Array<{
      version: number
      created_at: string
      created_by: string
      changes: string
      size_bytes: number
    }>
  > {
    const response = await apiClient.get(
      `${this.baseUrl}/${documentId}/versions`
    )
    return response.data
  }

  async getDocumentVersion(
    documentId: string,
    version: number
  ): Promise<Document> {
    const response = await apiClient.get(
      `${this.baseUrl}/${documentId}/versions/${version}`
    )
    return response.data
  }

  async restoreDocumentVersion(
    documentId: string,
    version: number
  ): Promise<Document> {
    const response = await apiClient.post(
      `${this.baseUrl}/${documentId}/restore`,
      {
        version,
      }
    )
    return response.data
  }

  // ==================== 文档标签 ====================

  async addTags(documentId: string, tags: string[]): Promise<Document> {
    const response = await apiClient.post(
      `${this.baseUrl}/${documentId}/tags`,
      {
        tags,
      }
    )
    return response.data
  }

  async removeTags(documentId: string, tags: string[]): Promise<Document> {
    const response = await apiClient.delete(
      `${this.baseUrl}/${documentId}/tags`,
      {
        data: { tags },
      }
    )
    return response.data
  }

  async getAllTags(): Promise<
    Array<{
      tag: string
      count: number
    }>
  > {
    const response = await apiClient.get(`${this.baseUrl}/tags`)
    return response.data
  }
}

// ==================== 导出 ====================

export const documentsService = new DocumentsService()
export default documentsService
