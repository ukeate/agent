import apiClient from './apiClient'

// ==================== 类型定义 ====================

export enum MemoryType {
  WORKING = 'working',
  EPISODIC = 'episodic',
  SEMANTIC = 'semantic',
}

export enum MemoryStatus {
  ACTIVE = 'active',
  ARCHIVED = 'archived',
  COMPRESSED = 'compressed',
  DELETED = 'deleted',
}

export interface MemoryCreateRequest {
  type: MemoryType
  content: string
  metadata?: Record<string, any>
  importance?: number
  tags?: string[]
  source?: string
}

export interface MemoryUpdateRequest {
  content?: string
  metadata?: Record<string, any>
  importance?: number
  tags?: string[]
  status?: MemoryStatus
}

export interface MemoryResponse {
  id: string
  type: MemoryType
  content: string
  metadata: Record<string, any>
  importance: number
  access_count: number
  created_at: string
  last_accessed: string
  status: MemoryStatus
  tags: string[]
  relevance_score?: number
}

export interface MemoryFilters {
  memory_types?: MemoryType[]
  status?: MemoryStatus[]
  min_importance?: number
  max_importance?: number
  created_after?: string
  created_before?: string
  tags?: string[]
  session_id?: string
  user_id?: string
}

export interface MemoryAnalytics {
  total_memories: number
  memories_by_type: Record<string, number>
  memories_by_status: Record<string, number>
  avg_importance: number
  total_access_count: number
  avg_access_count: number
  most_accessed_memories: MemoryResponse[]
  recent_memories: MemoryResponse[]
  memory_growth_rate: number
  storage_usage_mb: number
}

export interface MemoryPatterns {
  time_patterns: {
    hourly_distribution: Record<string, number>
    daily_distribution: Record<string, number>
  }
  content_patterns: {
    tag_frequency: Record<string, number>
    type_distribution: Record<string, number>
  }
  usage_patterns: {
    peak_hours: Array<[string, number]>
    most_active_days: Array<[string, number]>
  }
}

export interface MemoryTrends {
  period: {
    start_date: string
    end_date: string
    total_days: number
  }
  daily_trends: Record<
    string,
    {
      memory_count: number
      avg_importance: number
      total_access: number
      type_distribution: Record<string, number>
    }
  >
  summary: {
    total_memories: number
    avg_daily_creation: number
    growth_rate: number
  }
}

export interface MemoryGraphStats {
  graph_overview: {
    total_nodes: number
    total_edges: number
    density: number
    connected_components: number
  }
  node_statistics: {
    isolated_nodes: number
    connected_nodes: number
    max_connections: number
    avg_connections: number
  }
  connectivity_distribution: Record<string, number>
  memory_types_in_graph: Record<string, number>
}

// ==================== Service Class ====================

class MemoryManagementService {
  private baseUrl = '/memories'

  // ==================== 记忆管理 ====================

  async createMemory(
    request: MemoryCreateRequest,
    sessionId?: string
  ): Promise<MemoryResponse> {
    const headers: Record<string, string> = {}
    if (sessionId) {
      headers['session-id'] = sessionId
    }

    const response = await apiClient.post(this.baseUrl, request, { headers })
    return response.data
  }

  async getMemory(memoryId: string): Promise<MemoryResponse> {
    const response = await apiClient.get(`${this.baseUrl}/${memoryId}`)
    return response.data
  }

  async updateMemory(
    memoryId: string,
    request: MemoryUpdateRequest
  ): Promise<MemoryResponse> {
    const response = await apiClient.put(`${this.baseUrl}/${memoryId}`, request)
    return response.data
  }

  async deleteMemory(memoryId: string): Promise<{ message: string }> {
    const response = await apiClient.delete(`${this.baseUrl}/${memoryId}`)
    return response.data
  }

  // ==================== 记忆搜索 ====================

  async searchMemories(params: {
    query: string
    memory_types?: MemoryType[]
    status?: MemoryStatus[]
    min_importance?: number
    max_importance?: number
    tags?: string[]
    limit?: number
    session_id?: string
  }): Promise<MemoryResponse[]> {
    const headers: Record<string, string> = {}
    if (params.session_id) {
      headers['session-id'] = params.session_id
    }

    const response = await apiClient.get(`${this.baseUrl}/search`, {
      params: {
        query: params.query,
        memory_types: params.memory_types,
        status: params.status,
        min_importance: params.min_importance,
        max_importance: params.max_importance,
        tags: params.tags,
        limit: params.limit || 10,
      },
      headers,
    })
    return response.data
  }

  async getSessionMemories(
    sessionId: string,
    memoryType?: MemoryType,
    limit: number = 100
  ): Promise<MemoryResponse[]> {
    const response = await apiClient.get(
      `${this.baseUrl}/session/${sessionId}`,
      {
        params: { memory_type: memoryType, limit },
      }
    )
    return response.data
  }

  // ==================== 记忆关联 ====================

  async associateMemories(
    memoryId: string,
    targetMemoryId: string,
    weight: number = 0.5,
    associationType: string = 'related'
  ): Promise<{ message: string }> {
    const response = await apiClient.post(
      `${this.baseUrl}/${memoryId}/associate`,
      null,
      {
        params: {
          target_memory_id: targetMemoryId,
          weight,
          association_type: associationType,
        },
      }
    )
    return response.data
  }

  async getRelatedMemories(
    memoryId: string,
    depth: number = 2,
    limit: number = 10
  ): Promise<MemoryResponse[]> {
    const response = await apiClient.get(
      `${this.baseUrl}/${memoryId}/related`,
      {
        params: { depth, limit },
      }
    )
    return response.data
  }

  // ==================== 记忆巩固 ====================

  async consolidateSessionMemories(
    sessionId: string
  ): Promise<{ message: string }> {
    const response = await apiClient.post(
      `${this.baseUrl}/consolidate/${sessionId}`
    )
    return response.data
  }

  // ==================== 分析统计 ====================

  async getMemoryAnalytics(
    daysBack: number = 7,
    sessionId?: string
  ): Promise<MemoryAnalytics> {
    const response = await apiClient.get(`${this.baseUrl}/analytics`, {
      params: { days_back: daysBack, session_id: sessionId },
    })
    return response.data
  }

  async getMemoryPatterns(daysBack: number = 7): Promise<MemoryPatterns> {
    const response = await apiClient.get(`${this.baseUrl}/analytics/patterns`, {
      params: { days_back: daysBack },
    })
    return response.data
  }

  async getMemoryTrends(days: number = 30): Promise<MemoryTrends> {
    const response = await apiClient.get(`${this.baseUrl}/analytics/trends`, {
      params: { days },
    })
    return response.data
  }

  async getMemoryGraphStats(): Promise<MemoryGraphStats> {
    const response = await apiClient.get(
      `${this.baseUrl}/analytics/graph/stats`
    )
    return response.data
  }

  // ==================== 记忆清理 ====================

  async cleanupOldMemories(
    daysOld: number = 30,
    minImportance: number = 0.3
  ): Promise<{ message: string }> {
    const response = await apiClient.post(`${this.baseUrl}/cleanup`, null, {
      params: { days_old: daysOld, min_importance: minImportance },
    })
    return response.data
  }
}

// ==================== 导出 ====================

export const memoryManagementService = new MemoryManagementService()
export default memoryManagementService
