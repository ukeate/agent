import apiClient from './apiClient';

// 缓存分配接口
export interface CachedAssignment {
  user_id: string;
  experiment_id: string;
  variant_id: string;
  assigned_at: string;
  assignment_context: Record<string, any>;
}

// 分配创建请求
export interface CreateAssignmentRequest {
  user_id: string;
  experiment_id: string;
  variant_id: string;
  assignment_context?: Record<string, any>;
  ttl?: number;
}

// 批量分配请求
export interface BatchAssignmentRequest {
  assignments: CreateAssignmentRequest[];
}

// 缓存指标
export interface CacheMetrics {
  total_assignments: number;
  total_users: number;
  total_experiments: number;
  cache_hit_rate: number;
  cache_miss_rate: number;
  memory_usage_mb: number;
  expired_assignments: number;
  recent_assignments: number;
}

// 用户分配响应
export interface UserAssignmentResponse {
  user_id: string;
  experiment_id: string;
  variant_id: string | null;
  cache_status: string;
  assignment_context?: Record<string, any>;
  assigned_at?: string;
  message?: string;
}

// 健康检查响应
export interface HealthCheckResponse {
  status: string;
  redis_connection: boolean;
  cache_size: number;
  active_keys: number;
  memory_usage: string;
  timestamp: string;
}

// 缓存信息
export interface CacheInfo {
  cache_strategy: string;
  default_ttl_seconds: number;
  max_cache_size: number;
  batch_size: number;
  batch_timeout_seconds: number;
  key_prefix: string;
  redis_url: string;
}

class AssignmentCacheService {
  private baseUrl = '/assignment-cache';

  // 获取用户在特定实验中的分配
  async getUserAssignment(
    userId: string, 
    experimentId: string
  ): Promise<UserAssignmentResponse> {
    const response = await apiClient.get(
      `${this.baseUrl}/assignments/${userId}/${experimentId}`
    );
    return response.data;
  }

  // 创建用户分配
  async createAssignment(request: CreateAssignmentRequest): Promise<{
    message: string;
    user_id: string;
    experiment_id: string;
    variant_id: string;
    assigned_at: string;
  }> {
    const response = await apiClient.post(`${this.baseUrl}/assignments`, request);
    return response.data;
  }

  // 批量创建用户分配
  async createBatchAssignments(request: BatchAssignmentRequest): Promise<{
    total_requests: number;
    successful_count: number;
    failed_count: number;
    results: Array<{
      user_id: string;
      experiment_id: string;
      success: boolean;
      error?: string;
    }>;
  }> {
    const response = await apiClient.post(`${this.baseUrl}/assignments/batch`, request);
    return response.data;
  }

  // 获取用户的所有分配
  async getUserAllAssignments(userId: string): Promise<{
    user_id: string;
    total_assignments: number;
    assignments: Array<{
      experiment_id: string;
      variant_id: string;
      assigned_at: string;
      assignment_context: Record<string, any>;
    }>;
  }> {
    const response = await apiClient.get(`${this.baseUrl}/users/${userId}/assignments`);
    return response.data;
  }

  // 批量获取分配
  async batchGetAssignments(userExperimentPairs: Array<{
    user_id: string;
    experiment_id: string;
  }>): Promise<{
    total_requests: number;
    found_count: number;
    not_found_count: number;
    results: Array<{
      user_id: string;
      experiment_id: string;
      variant_id: string | null;
      assigned_at?: string;
      found: boolean;
    }>;
  }> {
    const response = await apiClient.post(
      `${this.baseUrl}/assignments/batch-get`, 
      userExperimentPairs
    );
    return response.data;
  }

  // 删除用户分配
  async deleteAssignment(userId: string, experimentId: string): Promise<{
    message: string;
    user_id: string;
    experiment_id: string;
  }> {
    const response = await apiClient.delete(
      `${this.baseUrl}/assignments/${userId}/${experimentId}`
    );
    return response.data;
  }

  // 清除用户的所有分配
  async clearUserAssignments(userId: string): Promise<{
    message: string;
    user_id: string;
    deleted_count: number;
  }> {
    const response = await apiClient.delete(`${this.baseUrl}/users/${userId}/assignments`);
    return response.data;
  }

  // 获取缓存指标
  async getCacheMetrics(): Promise<CacheMetrics> {
    const response = await apiClient.get(`${this.baseUrl}/metrics`);
    return response.data;
  }

  // 健康检查
  async healthCheck(): Promise<HealthCheckResponse> {
    const response = await apiClient.get(`${this.baseUrl}/health`);
    return response.data;
  }

  // 清空所有缓存
  async clearAllCache(): Promise<{
    message: string;
    timestamp: string;
  }> {
    const response = await apiClient.post(`${this.baseUrl}/clear`, {}, {
      params: { confirm: true }
    });
    return response.data;
  }

  // 获取缓存配置信息
  async getCacheInfo(): Promise<CacheInfo> {
    const response = await apiClient.get(`${this.baseUrl}/info`);
    return response.data;
  }

  // 预热缓存
  async warmupCache(userIds: string[]): Promise<{
    message: string;
    user_count: number;
    status: string;
  }> {
    const response = await apiClient.post(`${this.baseUrl}/warmup`, userIds);
    return response.data;
  }
}

export default new AssignmentCacheService();
