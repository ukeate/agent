/**
 * 记忆系统API服务
 */
import { apiClient } from './apiClient'
import type {
  Memory,
  MemoryCreateRequest,
  MemoryUpdateRequest,
  MemoryFilters,
  MemoryAnalytics,
  MemoryPattern,
  MemoryTrend,
} from '@/types/memory'

class MemoryService {
  private baseUrl = '/memories'

  /**
   * 创建新记忆
   */
  async createMemory(
    request: MemoryCreateRequest,
    sessionId?: string
  ): Promise<Memory> {
    const headers = sessionId ? { 'session-id': sessionId } : {}
    const response = await apiClient.post<Memory>(this.baseUrl, request, {
      headers,
    })
    return response.data
  }

  /**
   * 获取单个记忆
   */
  async getMemory(memoryId: string): Promise<Memory> {
    const response = await apiClient.get<Memory>(`${this.baseUrl}/${memoryId}`)
    return response.data
  }

  /**
   * 更新记忆
   */
  async updateMemory(
    memoryId: string,
    request: MemoryUpdateRequest
  ): Promise<Memory> {
    const response = await apiClient.put<Memory>(
      `${this.baseUrl}/${memoryId}`,
      request
    )
    return response.data
  }

  /**
   * 删除记忆
   */
  async deleteMemory(memoryId: string): Promise<void> {
    await apiClient.delete(`${this.baseUrl}/${memoryId}`)
  }

  /**
   * 搜索记忆
   */
  async searchMemories(
    query: string,
    filters?: MemoryFilters,
    limit: number = 10
  ): Promise<Memory[]> {
    const params = {
      query,
      limit,
      ...filters,
    }
    const response = await apiClient.get<Memory[]>(`${this.baseUrl}/search`, {
      params,
    })
    return response.data
  }

  /**
   * 获取会话记忆
   */
  async getSessionMemories(
    sessionId: string,
    memoryType?: string,
    limit: number = 100
  ): Promise<Memory[]> {
    const params = {
      memory_type: memoryType,
      limit,
    }
    const response = await apiClient.get<Memory[]>(
      `${this.baseUrl}/session/${sessionId}`,
      { params }
    )
    return response.data
  }

  /**
   * 关联两个记忆
   */
  async associateMemories(
    memoryId: string,
    targetMemoryId: string,
    weight: number = 0.5,
    associationType: string = 'related'
  ): Promise<void> {
    const params = {
      target_memory_id: targetMemoryId,
      weight,
      association_type: associationType,
    }
    await apiClient.post(`${this.baseUrl}/${memoryId}/associate`, null, {
      params,
    })
  }

  /**
   * 获取相关记忆
   */
  async getRelatedMemories(
    memoryId: string,
    depth: number = 2,
    limit: number = 10
  ): Promise<Memory[]> {
    const params = { depth, limit }
    const response = await apiClient.get<Memory[]>(
      `${this.baseUrl}/${memoryId}/related`,
      { params }
    )
    return response.data
  }

  /**
   * 巩固会话记忆
   */
  async consolidateMemories(sessionId: string): Promise<void> {
    await apiClient.post(`${this.baseUrl}/consolidate/${sessionId}`)
  }

  /**
   * 清理旧记忆
   */
  async cleanupOldMemories(
    daysOld: number = 30,
    minImportance: number = 0.3
  ): Promise<void> {
    const params = {
      days_old: daysOld,
      min_importance: minImportance,
    }
    await apiClient.post(`${this.baseUrl}/cleanup`, null, { params })
  }

  /**
   * 获取记忆分析
   */
  async getMemoryAnalytics(
    sessionId?: string,
    daysBack: number = 7
  ): Promise<MemoryAnalytics> {
    const params = {
      session_id: sessionId,
      days_back: daysBack,
    }
    const response = await apiClient.get<MemoryAnalytics>(
      `${this.baseUrl}/analytics`,
      { params }
    )
    return response.data
  }

  /**
   * 获取记忆模式
   */
  async getMemoryPatterns(sessionId?: string): Promise<MemoryPattern> {
    const params = sessionId ? { session_id: sessionId } : {}
    const response = await apiClient.get<any>(
      `${this.baseUrl}/analytics/patterns`,
      { params }
    )
    return response.data
  }

  /**
   * 获取记忆趋势
   */
  async getMemoryTrends(
    days: number = 30,
    sessionId?: string
  ): Promise<MemoryTrend> {
    const params = {
      days,
      session_id: sessionId,
    }
    const response = await apiClient.get<MemoryTrend>(
      `${this.baseUrl}/analytics/trends`,
      { params }
    )
    return response.data
  }

  /**
   * 获取图统计
   */
  async getGraphStatistics(): Promise<any> {
    const response = await apiClient.get(
      `${this.baseUrl}/analytics/graph/stats`
    )
    return response.data
  }

  /**
   * 获取系统健康状态
   */
  async getSystemHealth(): Promise<any> {
    const response = await apiClient.get(`${this.baseUrl}/analytics/health`)
    return response.data
  }

  /**
   * 导入记忆
   */
  async importMemories(memories: any[], sessionId?: string): Promise<any> {
    const params = sessionId ? { session_id: sessionId } : {}
    const response = await apiClient.post(
      `${this.baseUrl}/import`,
      { memories },
      { params }
    )
    return response.data
  }

  /**
   * 导出记忆
   */
  async exportMemories(
    sessionId?: string,
    memoryTypes?: string[]
  ): Promise<any[]> {
    const params = {
      session_id: sessionId,
      memory_types: memoryTypes,
    }
    const response = await apiClient.get<any[]>(`${this.baseUrl}/export`, {
      params,
    })
    return response.data
  }
}

export const memoryService = new MemoryService()
