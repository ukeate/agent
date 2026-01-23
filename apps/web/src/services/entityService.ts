/**
 * 实体管理服务
 * 提供实体CRUD操作的API调用服务
 */

import apiClient from './apiClient'

import { logger } from '../utils/logger'
// 实体类型定义
export interface Entity {
  id: string
  uri: string
  type: string
  label: string
  properties: Record<string, any>
  created: string
  updated: string
  status: 'active' | 'inactive' | 'pending'
}

export interface EntityCreateRequest {
  uri: string
  type: string
  label: string
  properties?: Record<string, any>
}

export interface EntityUpdateRequest {
  uri?: string
  type?: string
  label?: string
  properties?: Record<string, any>
  status?: 'active' | 'inactive' | 'pending'
}

export interface EntitySearchRequest {
  query?: string
  type?: string
  status?: string
  limit?: number
  offset?: number
}

export interface EntityBatchRequest {
  entities: EntityCreateRequest[]
}

export interface EntityResponse {
  entities: Entity[]
  total: number
  limit: number
  offset: number
}

class EntityService {
  private baseUrl = '/entities'

  /**
   * 获取所有实体列表
   */
  async getEntities(params?: {
    type?: string
    status?: string
    limit?: number
    offset?: number
  }): Promise<EntityResponse> {
    try {
      const response = await apiClient.get(this.baseUrl, { params })
      return response.data
    } catch (error) {
      logger.error('获取实体列表失败:', error)
      throw new Error('获取实体列表失败，请检查网络连接或稍后重试')
    }
  }

  /**
   * 获取单个实体详情
   */
  async getEntity(entityId: string): Promise<Entity> {
    try {
      const response = await apiClient.get(`${this.baseUrl}/${entityId}`)
      return response.data
    } catch (error) {
      logger.error('获取实体详情失败:', error)
      throw new Error('获取实体详情失败，请检查网络连接或稍后重试')
    }
  }

  /**
   * 创建新实体
   */
  async createEntity(entityData: EntityCreateRequest): Promise<Entity> {
    try {
      const response = await apiClient.post(this.baseUrl, entityData)
      return response.data
    } catch (error) {
      logger.error('创建实体失败:', error)
      throw new Error('创建实体失败，请检查输入数据或稍后重试')
    }
  }

  /**
   * 更新实体
   */
  async updateEntity(
    entityId: string,
    entityData: EntityUpdateRequest
  ): Promise<Entity> {
    try {
      const response = await apiClient.put(
        `${this.baseUrl}/${entityId}`,
        entityData
      )
      return response.data
    } catch (error) {
      logger.error('更新实体失败:', error)
      throw new Error('更新实体失败，请检查输入数据或稍后重试')
    }
  }

  /**
   * 删除实体
   */
  async deleteEntity(entityId: string): Promise<void> {
    try {
      await apiClient.delete(`${this.baseUrl}/${entityId}`)
    } catch (error) {
      logger.error('删除实体失败:', error)
      throw new Error('删除实体失败，请检查网络连接或稍后重试')
    }
  }

  /**
   * 搜索实体
   */
  async searchEntities(
    searchRequest: EntitySearchRequest
  ): Promise<EntityResponse> {
    try {
      const response = await apiClient.post(
        `${this.baseUrl}/search`,
        searchRequest
      )
      return response.data
    } catch (error) {
      logger.error('搜索实体失败:', error)
      throw new Error('搜索实体失败，请检查搜索条件或稍后重试')
    }
  }

  /**
   * 批量创建实体
   */
  async batchCreateEntities(
    batchRequest: EntityBatchRequest
  ): Promise<EntityResponse> {
    try {
      const response = await apiClient.post(
        `${this.baseUrl}/batch`,
        batchRequest
      )
      return response.data
    } catch (error) {
      logger.error('批量创建实体失败:', error)
      throw new Error('批量创建实体失败，请检查输入数据或稍后重试')
    }
  }

  /**
   * 获取实体类型列表
   */
  async getEntityTypes(): Promise<string[]> {
    try {
      const response = await apiClient.get(`${this.baseUrl}/types`)
      return response.data
    } catch (error) {
      logger.error('获取实体类型失败:', error)
      throw new Error('获取实体类型失败，请检查网络连接或稍后重试')
    }
  }

  /**
   * 获取实体统计信息
   */
  async getEntityStats(): Promise<{
    total: number
    activeCount: number
    inactiveCount: number
    pendingCount: number
    typeDistribution: Record<string, number>
  }> {
    try {
      const response = await apiClient.get(`${this.baseUrl}/stats`)
      return response.data
    } catch (error) {
      logger.error('获取实体统计失败:', error)
      throw new Error('获取实体统计失败，请检查网络连接或稍后重试')
    }
  }

  /**
   * 验证实体数据
   */
  validateEntity(
    entityData: EntityCreateRequest | EntityUpdateRequest
  ): string[] {
    const errors: string[] = []

    if (
      'uri' in entityData &&
      entityData.uri &&
      !this.isValidURI(entityData.uri)
    ) {
      errors.push('URI格式无效')
    }

    if (
      'type' in entityData &&
      entityData.type &&
      !entityData.type.includes(':')
    ) {
      errors.push('实体类型必须包含命名空间前缀')
    }

    if (
      'label' in entityData &&
      entityData.label &&
      entityData.label.trim().length === 0
    ) {
      errors.push('实体标签不能为空')
    }

    return errors
  }

  /**
   * 验证URI格式
   */
  private isValidURI(uri: string): boolean {
    try {
      new URL(uri)
      return true
    } catch {
      return false
    }
  }

  /**
   * 导出实体数据
   */
  async exportEntities(format: 'json' | 'rdf' | 'csv' = 'json'): Promise<Blob> {
    try {
      const response = await apiClient.get(`${this.baseUrl}/export`, {
        params: { format },
        responseType: 'blob',
      })
      return response.data
    } catch (error) {
      logger.error('导出实体数据失败:', error)
      throw new Error('导出实体数据失败，请稍后重试')
    }
  }

  /**
   * 导入实体数据
   */
  async importEntities(
    file: File,
    format: 'json' | 'rdf' | 'csv' = 'json'
  ): Promise<{
    success: number
    failed: number
    errors: string[]
  }> {
    try {
      const formData = new FormData()
      formData.append('file', file)
      formData.append('format', format)

      const response = await apiClient.post(
        `${this.baseUrl}/import`,
        formData,
        {
          headers: {
            'Content-Type': 'multipart/form-data',
          },
        }
      )
      return response.data
    } catch (error) {
      logger.error('导入实体数据失败:', error)
      throw new Error('导入实体数据失败，请检查文件格式或稍后重试')
    }
  }
}

export const entityService = new EntityService()
export default entityService
