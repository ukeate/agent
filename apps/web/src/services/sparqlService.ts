import apiClient from './apiClient';

export interface SPARQLQueryResult {
  query_id: string;
  success: boolean;
  result_type: string;
  results: {
    head?: {
      vars?: string[];
      link?: string[];
    };
    results?: {
      bindings?: Array<Record<string, { type: string; value: string; datatype?: string }>>;
    };
    boolean?: boolean;
    data?: any[][];
    columns?: string[];
  };
  format: string;
  execution_time: number;
  result_count: number;
  metadata?: {
    cached?: boolean;
    optimized?: boolean;
    query_plan?: any;
  };
}

export interface SPARQLUpdateResult {
  update_id: string;
  success: boolean;
  affected_triples: number;
  execution_time: number;
  message?: string;
}

export interface QueryExplanation {
  query_id: string;
  query_type: string;
  execution_plan: {
    steps: Array<{
      operation: string;
      description: string;
      estimated_cost: number;
      estimated_rows: number;
    }>;
    total_cost: number;
    total_time: number;
  };
  optimization_suggestions?: string[];
  statistics?: {
    triple_patterns: number;
    joins: number;
    filters: number;
    estimated_complexity: string;
  };
}

export interface QueryHistory {
  id: string;
  query: string;
  timestamp: string;
  execution_time: number;
  result_count: number;
  status: 'success' | 'error';
  error?: string;
  cached?: boolean;
}

export interface QueryStatistics {
  total_queries: number;
  success_rate: number;
  average_execution_time: number;
  cache_hit_rate: number;
  most_frequent_patterns: Array<{
    pattern: string;
    count: number;
  }>;
  performance_trend: Array<{
    date: string;
    avg_time: number;
    query_count: number;
  }>;
}

export type ResultFormat = 'JSON' | 'XML' | 'CSV' | 'TSV' | 'TURTLE' | 'N3';
export type OptimizationLevel = 'NONE' | 'BASIC' | 'STANDARD' | 'AGGRESSIVE';

export interface SPARQLQueryRequest {
  query: string;
  default_graph_uri?: string;
  named_graph_uri?: string[];
  timeout?: number;
  format?: ResultFormat;
  use_cache?: boolean;
  optimization_level?: OptimizationLevel;
}

export interface SPARQLUpdateRequest {
  update: string;
  default_graph_uri?: string;
  named_graph_uri?: string[];
  timeout?: number;
}

export interface QueryExplanationRequest {
  query: string;
  include_optimization?: boolean;
  include_statistics?: boolean;
}

export interface SparqlPerformanceReport {
  performance_report: {
    timestamp: number;
    window_minutes: number;
    performance_summary: Record<string, any>;
    active_alerts: any[];
    recent_alerts: any[];
    top_slow_queries: Array<{
      query: string;
      execution_time: number;
      max_time: number;
      timestamp: number;
      frequency: number;
    }>;
    recommendations: string[];
  };
  sparql_engine_stats: Record<string, any>;
  recommendations: string[];
}

class SPARQLService {
  private baseUrl = '/kg/sparql';

  // 查询执行
  async executeQuery(request: SPARQLQueryRequest): Promise<SPARQLQueryResult> {
    const response = await apiClient.post(`${this.baseUrl}/query`, request);
    return response.data;
  }

  async executeUpdate(request: SPARQLUpdateRequest): Promise<SPARQLUpdateResult> {
    const response = await apiClient.post(`${this.baseUrl}/update`, request);
    return response.data;
  }

  // 查询分析
  async explainQuery(request: QueryExplanationRequest): Promise<QueryExplanation> {
    const response = await apiClient.post(`${this.baseUrl}/explain`, request);
    return response.data;
  }

  async validateQuery(query: string): Promise<{ valid: boolean; errors?: string[] }> {
    const response = await apiClient.post(`${this.baseUrl}/validate`, { query });
    return response.data;
  }

  async optimizeQuery(query: string, level?: OptimizationLevel): Promise<{
    original_query: string;
    optimized_query: string;
    improvements: string[];
    estimated_speedup: number;
  }> {
    const response = await apiClient.post(`${this.baseUrl}/optimize`, { 
      query, 
      optimization_level: level || 'STANDARD' 
    });
    return response.data;
  }

  // 历史和统计
  async getQueryHistory(limit?: number, offset?: number): Promise<QueryHistory[]> {
    const params = { limit: limit || 100, offset: offset || 0 };
    const response = await apiClient.get(`${this.baseUrl}/history`, { params });
    return response.data.history || [];
  }

  async getQueryStatistics(period?: string): Promise<QueryStatistics> {
    const params = period ? { period } : {};
    const response = await apiClient.get(`${this.baseUrl}/statistics`, { params });
    return response.data;
  }

  async getPerformanceReport(windowMinutes: number = 60): Promise<SparqlPerformanceReport> {
    const response = await apiClient.get(`${this.baseUrl}/performance`, { params: { window_minutes: windowMinutes } });
    return response.data;
  }

  async getCacheStats(): Promise<{
    cache_stats: Record<string, any>;
    total_queries: number;
    cached_queries: number;
    cache_hit_rate: number;
    timestamp: string;
  }> {
    const response = await apiClient.get(`${this.baseUrl}/cache/stats`);
    return response.data;
  }

  async clearCache(): Promise<{ cleared: number; message: string }> {
    const response = await apiClient.post(`${this.baseUrl}/cache/clear`);
    return response.data;
  }

  // 查询模板
  async getQueryTemplates(category?: string): Promise<Array<{
    id: string;
    name: string;
    description: string;
    query: string;
    category: string;
    parameters?: string[];
  }>> {
    const params = category ? { category } : {};
    const response = await apiClient.get(`${this.baseUrl}/templates`, { params });
    return response.data.templates || [];
  }

  async saveQueryTemplate(template: {
    name: string;
    description: string;
    query: string;
    category: string;
    parameters?: string[];
  }): Promise<{ id: string; success: boolean }> {
    const response = await apiClient.post(`${this.baseUrl}/templates`, template);
    return response.data;
  }

  // 数据集管理
  async listGraphs(): Promise<Array<{
    uri: string;
    triple_count: number;
    last_modified: string;
    size_bytes: number;
  }>> {
    const response = await apiClient.get(`${this.baseUrl}/graphs`);
    return response.data.graphs || [];
  }

  async createGraph(uri: string): Promise<{ success: boolean; message: string }> {
    const response = await apiClient.post(`${this.baseUrl}/graphs`, { uri });
    return response.data;
  }

  async deleteGraph(uri: string): Promise<{ success: boolean; deleted_triples: number }> {
    const response = await apiClient.delete(`${this.baseUrl}/graphs/${encodeURIComponent(uri)}`);
    return response.data;
  }

  // 导入导出
  async importData(file: File, format: string, graphUri?: string): Promise<{
    success: boolean;
    imported_triples: number;
    errors?: string[];
  }> {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('format', format);
    if (graphUri) {
      formData.append('graph_uri', graphUri);
    }

    const response = await apiClient.post(`${this.baseUrl}/import`, formData, {
      headers: {
        'Content-Type': 'multipart/form-data'
      }
    });
    return response.data;
  }

  async exportData(format: ResultFormat, graphUri?: string): Promise<Blob> {
    const params = { format, graph_uri: graphUri };
    const response = await apiClient.get(`${this.baseUrl}/export`, {
      params,
      responseType: 'blob'
    });
    return response.data;
  }

  // 实时监控
  async getPerformanceMetrics(): Promise<{
    current_load: number;
    active_queries: number;
    memory_usage: number;
    cache_size: number;
    uptime: number;
  }> {
    const response = await apiClient.get(`${this.baseUrl}/metrics/performance`);
    return response.data;
  }

  async getActiveQueries(): Promise<Array<{
    query_id: string;
    query: string;
    start_time: string;
    elapsed_time: number;
    user?: string;
  }>> {
    const response = await apiClient.get(`${this.baseUrl}/queries/active`);
    return response.data.queries || [];
  }

  async cancelQuery(queryId: string): Promise<{ success: boolean; message: string }> {
    const response = await apiClient.post(`${this.baseUrl}/queries/${queryId}/cancel`);
    return response.data;
  }
}

export const sparqlService = new SPARQLService();
