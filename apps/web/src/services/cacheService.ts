import apiClient from './apiClient'

import { logger } from '../utils/logger'
// 缓存统计接口
export interface CacheStats {
  total_size: number
  total_items: number
  hit_rate: number
  miss_rate: number
  total_hits: number
  total_misses: number
  memory_usage_mb: number
  max_memory_mb: number
  evictions: number
  expired_items: number
  cache_efficiency: number
  nodes: Record<
    string,
    {
      hits: number
      misses: number
      size: number
      items: number
      last_accessed: string
    }
  >
}

// 缓存健康状态接口
export interface CacheHealth {
  status: 'healthy' | 'degraded' | 'unhealthy'
  checks: {
    connectivity: boolean
    memory_usage: boolean
    hit_rate: boolean
    response_time: boolean
  }
  metrics: {
    memory_usage_percent: number
    hit_rate_percent: number
    avg_response_time_ms: number
  }
  issues: string[]
  recommendations: string[]
}

// 缓存性能指标接口
export interface CachePerformance {
  response_times: {
    avg_ms: number
    p50_ms: number
    p95_ms: number
    p99_ms: number
  }
  throughput: {
    reads_per_second: number
    writes_per_second: number
  }
  memory: {
    used_mb: number
    available_mb: number
    fragmentation_ratio: number
  }
  operations: {
    total_reads: number
    total_writes: number
    total_deletes: number
    failed_operations: number
  }
}

// 缓存条目接口
export interface CacheEntry {
  key: string
  value?: any
  size_bytes: number
  ttl_seconds?: number
  created_at: string
  accessed_at: string
  access_count: number
  node_name?: string
}

// 缓存策略接口
export interface CacheStrategy {
  eviction_policy: 'LRU' | 'LFU' | 'FIFO' | 'TTL' | 'NOEVICTION'
  max_size_mb: number
  default_ttl_seconds: number
  compression_enabled: boolean
  warming_enabled: boolean
}

// 缓存清理结果接口
export interface CacheClearResult {
  success: boolean
  cleared_count: number
  pattern: string
  message: string
}

// 缓存失效结果接口
export interface CacheInvalidateResult {
  success: boolean
  node_name: string
  invalidated_count: number
  message: string
}

class CacheService {
  private baseUrl = '/cache'

  // 获取缓存统计信息
  async getStats(): Promise<CacheStats> {
    try {
      const response = await apiClient.get(`${this.baseUrl}/stats`)
      return response.data
    } catch (error) {
      logger.error('获取缓存统计失败:', error)
      throw error
    }
  }

  // 检查缓存健康状态
  async checkHealth(): Promise<CacheHealth> {
    try {
      const response = await apiClient.get(`${this.baseUrl}/health`)
      return response.data
    } catch (error) {
      logger.error('缓存健康检查失败:', error)
      throw error
    }
  }

  // 获取缓存性能指标
  async getPerformance(): Promise<CachePerformance> {
    try {
      const response = await apiClient.get(`${this.baseUrl}/performance`)
      return response.data
    } catch (error) {
      logger.error('获取缓存性能失败:', error)
      throw error
    }
  }

  // 清理缓存
  async clearCache(pattern: string = '*'): Promise<CacheClearResult> {
    try {
      const response = await apiClient.delete(`${this.baseUrl}/clear`, {
        params: { pattern },
      })
      return response.data
    } catch (error) {
      logger.error('清理缓存失败:', error)
      throw error
    }
  }

  // 失效特定节点缓存
  async invalidateNodeCache(
    nodeName: string,
    userId?: string,
    sessionId?: string,
    workflowId?: string
  ): Promise<CacheInvalidateResult> {
    try {
      const response = await apiClient.delete(
        `${this.baseUrl}/invalidate/${nodeName}`,
        {
          params: {
            user_id: userId,
            session_id: sessionId,
            workflow_id: workflowId,
          },
        }
      )
      return response.data
    } catch (error) {
      logger.error('失效节点缓存失败:', error)
      throw error
    }
  }

  // 获取缓存条目
  async getEntry(key: string): Promise<CacheEntry | null> {
    try {
      const response = await apiClient.get(`${this.baseUrl}/entry/${key}`)
      return response.data
    } catch (error) {
      logger.error('获取缓存条目失败:', error)
      return null
    }
  }

  // 设置缓存条目
  async setEntry(key: string, value: any, ttl?: number): Promise<boolean> {
    try {
      const response = await apiClient.post(`${this.baseUrl}/entry`, {
        key,
        value,
        ttl,
      })
      return response.data.success
    } catch (error) {
      logger.error('设置缓存条目失败:', error)
      return false
    }
  }

  // 删除缓存条目
  async deleteEntry(key: string): Promise<boolean> {
    try {
      const response = await apiClient.delete(`${this.baseUrl}/entry/${key}`)
      return response.data.success
    } catch (error) {
      logger.error('删除缓存条目失败:', error)
      throw error
    }
  }

  // 获取缓存策略
  async getStrategy(): Promise<CacheStrategy> {
    try {
      const response = await apiClient.get(`${this.baseUrl}/strategy`)
      return response.data
    } catch (error) {
      logger.error('获取缓存策略失败:', error)
      throw error
    }
  }

  // 更新缓存策略
  async updateStrategy(strategy: Partial<CacheStrategy>): Promise<boolean> {
    try {
      const response = await apiClient.put(`${this.baseUrl}/strategy`, strategy)
      return response.data.success
    } catch (error) {
      logger.error('更新缓存策略失败:', error)
      throw error
    }
  }

  // 预热缓存
  async warmCache(keys: string[]): Promise<{
    success: boolean
    warmed_count: number
    failed_keys: string[]
  }> {
    try {
      const response = await apiClient.post(`${this.baseUrl}/warm`, { keys })
      return response.data
    } catch (error) {
      logger.error('预热缓存失败:', error)
      throw error
    }
  }

  // 获取缓存键列表
  async listKeys(
    pattern: string = '*',
    limit: number = 100
  ): Promise<string[]> {
    try {
      const response = await apiClient.get(`${this.baseUrl}/keys`, {
        params: { pattern, limit },
      })
      return response.data.keys || []
    } catch (error) {
      logger.error('获取缓存键列表失败:', error)
      throw error
    }
  }

  // 批量获取缓存条目
  async getMultiple(keys: string[]): Promise<Record<string, any>> {
    try {
      const response = await apiClient.post(`${this.baseUrl}/mget`, { keys })
      return response.data.entries || {}
    } catch (error) {
      logger.error('批量获取缓存失败:', error)
      throw error
    }
  }

  // 批量设置缓存条目
  async setMultiple(
    entries: Record<string, any>,
    ttl?: number
  ): Promise<boolean> {
    try {
      const response = await apiClient.post(`${this.baseUrl}/mset`, {
        entries,
        ttl,
      })
      return response.data.success
    } catch (error) {
      logger.error('批量设置缓存失败:', error)
      throw error
    }
  }
}

// 导出服务实例
export const cacheService = new CacheService()
export default cacheService
