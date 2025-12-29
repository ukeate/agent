import apiClient from './apiClient';

export interface QuantizationConfig {
  mode: 'float32' | 'int8' | 'int4' | 'adaptive';
  precision_threshold: number;
  compression_ratio: number;
  enable_dynamic: boolean;
}

export interface SearchConfig {
  query: string;
  top_k: number;
  pg_weight: number;
  qdrant_weight: number;
  use_cache: boolean;
  quantize: boolean;
  search_mode: 'hybrid' | 'pg_only' | 'qdrant_only';
}

export interface SystemStatus {
  pgvector_version: string;
  upgrade_available: boolean;
  quantization_enabled: boolean;
  cache_status: 'healthy' | 'warning' | 'error';
  index_health: 'optimal' | 'needs_optimization' | 'error';
  last_updated: string;
}

export class PgVectorApi {
  private baseUrl = '/pgvector';

  // 系统状态
  async getSystemStatus(): Promise<SystemStatus> {
    const response = await apiClient.get(`${this.baseUrl}/status`);
    return response.data;
  }

  async upgradeToPgVector08(): Promise<{ success: boolean; message: string }> {
    const response = await apiClient.post(`${this.baseUrl}/upgrade`);
    return response.data;
  }

  // 量化配置
  async getQuantizationConfig(): Promise<QuantizationConfig> {
    const response = await apiClient.get(`${this.baseUrl}/quantization/config`);
    return response.data;
  }

  async applyQuantizationConfig(config: QuantizationConfig): Promise<void> {
    await apiClient.post(`${this.baseUrl}/quantization/configure`, config);
  }

  async testQuantization(config: QuantizationConfig): Promise<any[]> {
    const response = await apiClient.post(`${this.baseUrl}/quantization/test`, config);
    return response.data;
  }

  // 性能监控
  async getPerformanceMetrics(timeRange: string): Promise<any[]> {
    const response = await apiClient.get(`${this.baseUrl}/performance/metrics`, {
      params: { time_range: timeRange }
    });
    return response.data;
  }

  async getPerformanceTargets(): Promise<any[]> {
    const response = await apiClient.get(`${this.baseUrl}/performance/targets`);
    return response.data;
  }

  // 混合检索
  async hybridSearch(config: SearchConfig): Promise<{ results: any[]; metrics: any }> {
    const response = await apiClient.post(`${this.baseUrl}/search/hybrid`, config);
    return response.data;
  }

  async benchmarkRetrievalMethods(params: {
    test_queries: string[];
    top_k: number;
  }): Promise<any[]> {
    const response = await apiClient.post(`${this.baseUrl}/search/benchmark`, params);
    return response.data;
  }

  // 数据完整性
  async getIntegritySummary(tableName: string): Promise<any> {
    const response = await apiClient.get(`${this.baseUrl}/integrity/summary`, {
      params: { table_name: tableName }
    });
    return response.data;
  }

  async validateVectorDataIntegrity(params: {
    table_name: string;
    batch_size: number;
  }): Promise<any> {
    const response = await apiClient.post(`${this.baseUrl}/integrity/validate`, params);
    return response.data;
  }

  async repairVectorData(
    integrityReport: any,
    strategy: 'remove_invalid' | 'set_null'
  ): Promise<any> {
    const response = await apiClient.post(`${this.baseUrl}/integrity/repair`, {
      integrity_report: integrityReport,
      repair_strategy: strategy
    });
    return response.data;
  }

  // 索引管理
  async getIndexInfo(tableName: string): Promise<any> {
    const response = await apiClient.get(`${this.baseUrl}/indexes/${tableName}`);
    return response.data;
  }

  async createOptimizedIndex(params: {
    table_name: string;
    vector_column: string;
    index_type: 'hnsw' | 'ivf' | 'hybrid';
    config: any;
  }): Promise<any> {
    const response = await apiClient.post(`${this.baseUrl}/indexes/create`, {
      table_name: params.table_name,
      column_name: params.vector_column,
      index_type: params.index_type,
      distance_metric: 'l2',
      index_options: params.config,
    });
    return response.data;
  }

  // 缓存管理
  async getCacheStats(): Promise<any> {
    const response = await apiClient.get(`${this.baseUrl}/cache/stats`);
    return response.data;
  }

  async clearCache(): Promise<{ success: boolean }> {
    const response = await apiClient.post(`${this.baseUrl}/cache/clear`);
    return response.data;
  }

  async listIndexes(): Promise<any[]> {
    const response = await apiClient.get(`${this.baseUrl}/indexes/list`);
    return response.data.indexes || [];
  }
}

export const pgvectorApi = new PgVectorApi();
